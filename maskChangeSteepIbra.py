#!/usr/bin/env python
"""
Filename:  maskChangeSteepIbra.py
Purpose:   Produces a 2-band masked long-term statistics image for IBRA-constrained DRCM processing:
             - Band 1: Long-term bare ground stats masked by clearing and slope (1..101, 0=nodata)
             - Band 2: Rasterized IBRA biogeographic region code (1..89 from OBJECTID, 0=nodata)
"""

import os
import re
import sys
import argparse
import numpy as np
from osgeo import gdal, ogr, osr
from rios import applier, cuiprogress

gdal.UseExceptions()

_STAGE_DIX_RE = re.compile(r"(?:^|[_\-.])dix", re.IGNORECASE)
_STAGE_DP8_RE = re.compile(r"(?:^|[_\-.])dp8", re.IGNORECASE)


def infer_bare_encoding(path):
    """Infer bare ground encoding stage ('dixa2' vs 'dp8a2') from filename."""
    base = os.path.basename(path or "")
    if _STAGE_DP8_RE.search(base):
        return "dp8a2"
    if _STAGE_DIX_RE.search(base):
        return "dixa2"
    return "dixa2"


def validate_ibra_layer_attribute(lyr, attr_name):
    """Verify attr_name exists and all non-null feature values fit within uint8 1..255."""
    lyr_defn = lyr.GetLayerDefn()
    field_idx = lyr_defn.GetFieldIndex(attr_name)
    if field_idx < 0:
        raise ValueError(f"IBRA attribute '{attr_name}' not found in layer schema.")

    lyr.ResetReading()
    seen_any = False
    try:
        for feat in lyr:
            if not feat.IsFieldSetAndNotNull(field_idx):
                continue
            val = int(feat.GetField(field_idx))
            seen_any = True
            if val <= 0 or val > 255:
                raise ValueError(
                    f"IBRA attribute '{attr_name}' value {val} exceeds valid 1-byte region range 1..255; "
                    "cannot rasterize into GDT_Byte without ID corruption."
                )
    finally:
        lyr.ResetReading()

    if not seen_any:
        raise ValueError(f"IBRA layer contains no non-null values for attribute '{attr_name}'.")


def ensure_ibra_albers(ibra_vector_path, target_proj_wkt, output_albers_path=None):
    """
    Ensure the IBRA vector is reprojected to the raster's projection (Australian Albers)
    for fast spatial filtering and rasterization.
    """
    if output_albers_path is None:
        base, _ = os.path.splitext(ibra_vector_path)
        output_albers_path = f"{base}_albers.fgb"

    if os.path.exists(output_albers_path):
        return output_albers_path

    print(f"Reprojecting {ibra_vector_path} to match reference projection -> {output_albers_path}...")
    tmp_path = f"{output_albers_path}.tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    opt = gdal.VectorTranslateOptions(dstSRS=target_proj_wkt, format="FlatGeobuf")
    ds = None
    try:
        ds = gdal.VectorTranslate(tmp_path, ibra_vector_path, options=opt)
        ds = None
        os.replace(tmp_path, output_albers_path)
    except Exception:
        ds = None
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise
    return output_albers_path


def maskStatsIbra(info, inputs, outputs, otherArgs):
    """
    RIOS block callback:
      - Band 1: Bare ground 95th/max stats masked by clearing and slope.
      - Band 2: Rasterized IBRA region code (1..255).
    """
    bare = inputs.bareStats[6]
    (nRows, nCols) = inputs.terrainData.shape[1:]
    outImage = np.zeros((2, nRows, nCols), dtype=np.uint8)

    nullValBare = getattr(otherArgs, "nullValBare", None)
    bare_encoding = getattr(otherArgs, "bare_encoding", "dixa2")

    # Mask valid bare domain strictly before int16 subtraction so invalid uint8 values
    # (e.g. 202..255 under NumPy 2 NEP-50) can never underflow and clip to 101.
    if bare_encoding == "dp8a2":
        valid_bare = bare <= 100
        offset_top = 101
    else:
        valid_bare = (bare >= 100) & (bare <= 200)
        offset_top = 201
    if nullValBare is not None:
        try:
            nd = int(round(float(nullValBare)))
            if float(nullValBare) == nd and 0 <= nd <= 255:
                valid_bare &= bare != np.uint8(nd)
        except (TypeError, ValueError, OverflowError):
            pass

    if np.any(valid_bare):
        # Band 1: Bare ground masked by clearing and slope:
        # 1. NCAS Woody Extent: changeData[0] is Woody Max (0=non-woody, 1=woodland, 2=forest).
        #    Exclude Class 2 (forest, canopy cover >20%).
        # 2. Historical Clearing: changeData[2] is clearing history (1=cleared).
        # 3. Slope Relief: terrainData[0] < 39 keeps slope < 10% (Speight 2009 Table 5).
        # 4. Long-term Bare: scaled to persistent groundcover + 1 (1..101, 0 = NoData).
        valid_mask = (
            (inputs.changeData[0] != 2)
            & (inputs.changeData[2] != 1)
            & (inputs.terrainData[0] < 39)
            & valid_bare
        )
        gc_i16 = offset_top - bare.astype(np.int16)
        outImage[0] = np.where(valid_mask, np.clip(gc_i16, 1, 101), 0).astype(np.uint8)

    # Band 2: Rasterize IBRA region code whenever block intersects vector features
    # (preserves IBRA polygons over land holes where bareStats == 0).
    grid_gt = info.getTransform()
    col_in_grid, row_in_grid = info.getPixColRow(0, 0)
    xRes, yRes = grid_gt[1], grid_gt[5]
    tl_x = grid_gt[0] + col_in_grid * xRes
    tl_y = grid_gt[3] + row_in_grid * yRes
    br_x = tl_x + nCols * xRes
    br_y = tl_y + nRows * yRes

    xMin_b = min(tl_x, br_x)
    xMax_b = max(tl_x, br_x)
    yMin_b = min(tl_y, br_y)
    yMax_b = max(tl_y, br_y)

    ext = otherArgs.extent
    if not (xMax_b < ext[0] or xMin_b > ext[1] or yMax_b < ext[2] or yMin_b > ext[3]):
        otherArgs.lyr.SetSpatialFilterRect(xMin_b, yMin_b, xMax_b, yMax_b)
        if otherArgs.lyr.GetFeatureCount() > 0:
            if nCols == otherArgs.mem_ds.RasterXSize and nRows == otherArgs.mem_ds.RasterYSize:
                mem_ds = otherArgs.mem_ds
                mem_band = otherArgs.mem_band
            else:
                mem_ds = otherArgs.mem_driver.Create("", nCols, nRows, 1, gdal.GDT_Byte)
                mem_ds.SetProjection(otherArgs.proj)
                mem_band = mem_ds.GetRasterBand(1)

            mem_ds.SetGeoTransform((tl_x, xRes, 0, tl_y, 0, yRes))
            mem_band.Fill(0)
            ibra_attr = getattr(otherArgs, "ibra_attr", "OBJECTID")
            gdal.RasterizeLayer(mem_ds, [1], otherArgs.lyr, options=[f"ATTRIBUTE={ibra_attr}"])
            outImage[1] = mem_band.ReadAsArray()

    outputs.dcrmMask = outImage


def parse_args():
    parser = argparse.ArgumentParser(description="Create 2-band masked stats + IBRA GeoTIFF for DRCM.")
    parser.add_argument("--change", default="woodyMaxMinCleared.tif", help="Woody change raster")
    parser.add_argument("--terrain", default="slope_relief_class_3s.tif", help="Slope relief raster")
    parser.add_argument("--bare", default="lztmre_aus_m198912202012_dixa2_bareStats.tif", help="Bare stats raster")
    parser.add_argument("--ibra", default="ibra.fgb", help="IBRA vector dataset")
    parser.add_argument("--ibra-attr", default="OBJECTID", help="Vector attribute for 1..255 region code (default: OBJECTID)")
    parser.add_argument(
        "--bare-encoding",
        choices=("auto", "dixa2", "dp8a2"),
        default="auto",
        help="Bare stats scaling stage (default: infer dp8 vs dix from --bare filename)",
    )
    parser.add_argument("-o", "--output", default="lztmre_aus_m198912202012_dixa2_change_slope2_ibra.tif", help="Output GeoTIFF")
    parser.add_argument("--test-chip", nargs=4, type=int, metavar=("XOFF", "YOFF", "W", "H"), help="Test chip window")
    return parser.parse_args()


def main():
    args = parse_args()

    infiles = applier.FilenameAssociations()
    outfiles = applier.FilenameAssociations()
    otherArgs = applier.OtherInputs()

    infiles.changeData = args.change
    infiles.terrainData = args.terrain
    infiles.bareStats = args.bare
    outfiles.dcrmMask = args.output

    # Inspect reference dataset and release GDAL handle cleanly
    ref_ds = gdal.Open(infiles.bareStats, gdal.GA_ReadOnly)
    if ref_ds.RasterCount < 7:
        ref_count = ref_ds.RasterCount
        ref_ds = None
        raise ValueError(f"Expected >= 7 bands in bare stats raster {infiles.bareStats}, got {ref_count}")
    target_proj = ref_ds.GetProjection()
    ref_gt = ref_ds.GetGeoTransform()
    ref_w, ref_h = ref_ds.RasterXSize, ref_ds.RasterYSize
    null_val_bare = ref_ds.GetRasterBand(7).GetNoDataValue()
    ref_ds = None

    otherArgs.nullValBare = null_val_bare
    otherArgs.bare_encoding = (
        infer_bare_encoding(args.bare) if args.bare_encoding == "auto" else args.bare_encoding
    )
    otherArgs.ibra_attr = args.ibra_attr

    # Reproject IBRA vector to Australian Albers if needed and validate 1..255 attribute bounds
    ibra_albers_path = ensure_ibra_albers(args.ibra, target_proj)
    vec_ds = ogr.Open(ibra_albers_path, 0)
    if vec_ds is None:
        raise ValueError(f"Could not open IBRA vector dataset: {ibra_albers_path}")
    otherArgs.lyr = vec_ds.GetLayer(0)
    validate_ibra_layer_attribute(otherArgs.lyr, args.ibra_attr)
    otherArgs.proj = target_proj
    otherArgs.extent = otherArgs.lyr.GetExtent()

    # Preallocate reusable in-memory rasterization dataset to avoid per-block allocation overhead
    otherArgs.mem_driver = gdal.GetDriverByName("MEM")
    otherArgs.mem_ds = otherArgs.mem_driver.Create("", 512, 512, 1, gdal.GDT_Byte)
    otherArgs.mem_ds.SetProjection(target_proj)
    otherArgs.mem_band = otherArgs.mem_ds.GetRasterBand(1)

    controls = applier.ApplierControls()
    controls.setFootprintType(applier.BOUNDS_FROM_REFERENCE)
    controls.setReferenceImage(infiles.bareStats)
    controls.setResampleMethod("near")
    controls.setStatsIgnore(0)
    controls.setOutputDriverName("GTIFF")
    controls.setCreationOptions(["COMPRESS=DEFLATE",
                                 "ZLEVEL=1",
                                 "BIGTIFF=YES",
                                 "TILED=YES",
                                 "INTERLEAVE=BAND",
                                 "NUM_THREADS=ALL_CPUS",
                                 "BLOCKXSIZE=512",
                                 "BLOCKYSIZE=512"])
    controls.setWindowXsize(512)
    controls.setWindowYsize(512)
    controls.progress = cuiprogress.CUIProgressBar()

    if args.test_chip:
        xoff_t, yoff_t, w_t, h_t = args.test_chip
        if xoff_t < 0 or yoff_t < 0 or w_t <= 0 or h_t <= 0 or (xoff_t + w_t) > ref_w or (yoff_t + h_t) > ref_h:
            vec_ds = None
            raise ValueError(
                f"--test-chip {(xoff_t, yoff_t, w_t, h_t)} out of bounds for {ref_w}x{ref_h}"
            )
        xmin = ref_gt[0] + xoff_t * ref_gt[1]
        ymax = ref_gt[3] + yoff_t * ref_gt[5]
        ymin = ymax + h_t * ref_gt[5]
        xmax = xmin + w_t * ref_gt[1]
        grid = applier.PixelGridDefn(
            xMin=xmin, xMax=xmax, yMin=ymin, yMax=ymax,
            xRes=ref_gt[1], yRes=abs(ref_gt[5]), projection=target_proj
        )
        controls.setReferencePixgrid(grid)

    print(f"Applying maskStatsIbra (bare_encoding={otherArgs.bare_encoding}, ibra_attr={args.ibra_attr}) -> {args.output}...")
    applier.apply(maskStatsIbra, infiles, outfiles, otherArgs, controls=controls)
    vec_ds = None

    # Set band descriptions
    out_ds = gdal.Open(args.output, gdal.GA_Update)
    b1 = out_ds.GetRasterBand(1)
    b1.SetDescription("Masked Reference Groundcover (1..101, 0=NoData)")
    b1.SetNoDataValue(0)
    b2 = out_ds.GetRasterBand(2)
    b2.SetDescription("IBRA Region Code (OBJECTID 1..89, 0=NoData)")
    b2.SetNoDataValue(0)
    out_ds.FlushCache()
    out_ds = None
    print(f"Successfully generated 2-band masked stats + IBRA raster: {args.output}")


if __name__ == "__main__":
    main()
