#!/usr/bin/env python
"""
seasonalDRCMIbra.py

IBRA-Constrained Seasonal DRCM (Deficit / Reference Cover Method) Processing.

Compares seasonal fractional groundcover against long-term reference statistics
strictly constrained to matching IBRA biogeographic regions (1..89).
"""

import os
import argparse
import time
import numpy as np
from osgeo import gdal
import boto3

gdal.UseExceptions()

from .refpix_numba_ibra import refpixelfilter_ibra
from .refpix_numba import refpixelfilternodata

# when they specify --save
SAVE_BUCKET = 'cibo-dat-oregon'
SAVE_DIR = 'DRCM'


def format_time(seconds):
    """Format seconds into human-readable string (e.g. '42s', '3m 15s', '1h 22m 05s')."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours}h {mins:02d}m {secs:02d}s"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute IBRA-constrained Seasonal DRCM (Difference Ground Cover & Reference Cover)."
    )
    parser.add_argument(
        "input_groundcover",
        help="Input seasonal fractional groundcover GeoTIFF (e.g. lztmre_nsw_m202606202608_dp8a2_vinterim.tif)"
    )
    parser.add_argument(
        "--percentile-file",
        default="lztmre_aus_m198912202012_dixa2_change_slope2_ibra.tif",
        help="2-Band reference raster (Band 1: Stats, Band 2: IBRA Code). Default: lztmre_aus_m198912202012_dixa2_change_slope2_ibra.tif"
    )
    parser.add_argument(
        "--ibra-raster",
        default=None,
        help="Optional separate rasterized IBRA raster if percentile-file is single-band."
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output GeoTIFF path. If not specified, derived automatically from the input name."
    )
    parser.add_argument(
        "--window",
        type=int,
        default=1501,
        help="Comparison window size in pixels (must be odd, default: 1501)"
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=2048,
        help="Number of lines to process per chunk (default: 2048)"
    )
    parser.add_argument(
        "--lowerquantile",
        type=float,
        default=0.90,
        help="Lower quantile bound for finding high persistent cover reference sites (default: 0.90)"
    )
    parser.add_argument(
        "--upperquantile",
        type=float,
        default=0.95,
        help="Upper quantile bound for finding high persistent cover reference sites (default: 0.95)"
    )
    parser.add_argument(
        "--minvalpercent",
        type=float,
        default=0.10,
        help="Minimum fraction of valid pixels within comparison window in matching IBRA (default: 0.10)"
    )
    parser.add_argument(
        "--test-chip",
        nargs=4,
        type=int,
        metavar=("XOFF", "YOFF", "XSIZE", "YSIZE"),
        help="Optional test chip window [xoff, yoff, xsize, ysize] for testing/micro-subsetting"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of Numba parallel worker threads to use (default: all available cores)"
    )
    parser.add_argument(
        "--overviews",
        nargs="+",
        type=int,
        default=[4, 8, 16, 32],
        help="Overview levels to calculate using average resampling (default: 4 8 16 32)"
    )
    parser.add_argument(
        "--no-overviews",
        action="store_true",
        help="Skip calculating overviews on completion"
    )
    parser.add_argument(
        "--no-ibra",
        action="store_true",
        help="Do not constrain by IBRA regions; calculate reference statistics using a single global histogram."
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save to S3 and delete local output file on completion"
    )
    return parser.parse_args()


def derive_output_name(input_path, no_ibra=False):
    base = os.path.basename(input_path)
    root, ext = os.path.splitext(base)
    suffix = "doda2" if no_ibra else "doda2_ibra"
    generic = "_drcm" if no_ibra else "_drcm_ibra"
    if "dixa2" in root:
        out_root = root.replace("dixa2", suffix)
    elif "dp8a2" in root:
        out_root = root.replace("dp8a2", suffix)
    else:
        out_root = f"{root}{generic}"
    return f"{out_root}{ext}"


def convert_to_groundcover(bare_data, nodata_val=255):
    """
    Convert raw bare ground array to persistent groundcover + 1 (values 1..101, 0 is nodata).

    Auto-detects and supports both remote sensing standards:
    - Modern DEA (dp8a2): Bare stored as 0..100% with 255 NoData.
      Groundcover = 100 - bare% -> mapped to 101 - bare (1..101 range).
    - Legacy JRSRP/RSC (dixa2): Bare stored with +100 offset (100 = 0% bare, 200 = 100% bare).
      Groundcover mapped to 201 - bare (1..101 range).
    """
    max_val = np.max(bare_data)
    if max_val > 100 and max_val != 255:
        # Legacy JRSRP offset format: 100 = 0% bare, 200 = 100% bare
        gc = np.where((bare_data >= 100) & (bare_data <= 200) & (bare_data != nodata_val),
                      201 - bare_data, 0)
    else:
        # Modern DEA format: 0..100% bare, 255 = nodata
        gc = np.where((bare_data <= 100) & (bare_data != nodata_val),
                      101 - bare_data, 0)
    return np.clip(gc, 0, 101).astype(np.int8)


def run_seasonal_drcm_ibra(
    input_groundcover,
    percentile_file,
    ibra_raster=None,
    output_file=None,
    window=1501,
    chunksize=2048,
    lowerquantile=0.90,
    upperquantile=0.95,
    minvalpercent=0.15,
    test_chip=None,
    threads=None,
    overview_levels=(4, 8, 16, 32),
    no_overviews=False,
    no_ibra=False,
    save=False
):
    if threads is not None:
        import numba
        numba.set_num_threads(threads)

    if window % 2 == 0:
        window += 1

    if output_file is None:
        output_file = derive_output_name(input_groundcover, no_ibra=no_ibra)

    mode_str = "Single Global Histogram (IBRA disabled)" if no_ibra else "IBRA-Constrained Regional Histograms"
    print(f"Processing Mode: {mode_str}", flush=True)

    print(f"Opening input groundcover: {input_groundcover}", flush=True)
    in_ds = gdal.Open(input_groundcover, gdal.GA_ReadOnly)
    in_gt = in_ds.GetGeoTransform()
    in_proj = in_ds.GetProjection()
    in_band = in_ds.GetRasterBand(1)
    nodata_in = in_band.GetNoDataValue()
    if nodata_in is None:
        nodata_in = 255

    print(f"Opening reference stats file: {percentile_file}", flush=True)
    ref_ds = gdal.Open(percentile_file, gdal.GA_ReadOnly)
    ref_gt = ref_ds.GetGeoTransform()
    ref_band1 = ref_ds.GetRasterBand(1)

    if not no_ibra:
        if ref_ds.RasterCount >= 2:
            ref_ibra_band = ref_ds.GetRasterBand(2)
            ibra_ds = None
        elif ibra_raster:
            print(f"Opening separate IBRA raster: {ibra_raster}", flush=True)
            ibra_ds = gdal.Open(ibra_raster, gdal.GA_ReadOnly)
            ref_ibra_band = ibra_ds.GetRasterBand(1)
        else:
            raise ValueError(
                f"{percentile_file} has only 1 band. Please specify --ibra-raster, provide a 2-band masked stats+IBRA raster, or use --no-ibra."
            )
    else:
        ref_ibra_band = None
        ibra_ds = None

    # Determine spatial extent
    if test_chip:
        x_origin_off, y_origin_off, full_width, full_height = test_chip
        print(f"Running on test chip: xoff={x_origin_off}, yoff={y_origin_off}, w={full_width}, h={full_height}", flush=True)
        chip_gt = (
            in_gt[0] + x_origin_off * in_gt[1],
            in_gt[1],
            in_gt[2],
            in_gt[3] + y_origin_off * in_gt[5],
            in_gt[4],
            in_gt[5]
        )
    else:
        x_origin_off, y_origin_off = 0, 0
        full_width = in_ds.RasterXSize
        full_height = in_ds.RasterYSize
        chip_gt = in_gt

    # Calculate offset of groundcover in the reference dataset
    col_offset = int(round((in_gt[0] - ref_gt[0]) / ref_gt[1])) + x_origin_off
    row_offset = int(round((in_gt[3] - ref_gt[3]) / ref_gt[5])) + y_origin_off
    print(f"Spatial alignment offset in reference grid: col={col_offset}, row={row_offset}", flush=True)

    # Create output GeoTIFF
    print(f"Creating output dataset: {output_file} ({full_width} x {full_height})", flush=True)
    driver = gdal.GetDriverByName("GTiff")
    options = [
        "COMPRESS=DEFLATE",
        "ZLEVEL=1",
        "PREDICTOR=2",
        "TILED=YES",
        "INTERLEAVE=BAND",
        "BIGTIFF=YES",
        "NUM_THREADS=ALL_CPUS",
        "BLOCKXSIZE=512",
        "BLOCKYSIZE=512"
    ]
    out_ds = driver.Create(output_file, full_width, full_height, 2, gdal.GDT_Byte, options=options)
    out_ds.SetGeoTransform(chip_gt)
    out_ds.SetProjection(in_proj)

    dgc_band = out_ds.GetRasterBand(1)
    dgc_desc = "Difference Ground Cover (Unconstrained, 100+gc-ref, 0=NoData)" if no_ibra else "Difference Ground Cover (IBRA-Constrained, 100+gc-ref, 0=NoData)"
    dgc_band.SetDescription(dgc_desc)
    dgc_band.SetNoDataValue(0)

    ref_out_band = out_ds.GetRasterBand(2)
    ref_desc = "Reference Ground Cover (Unconstrained, 100+ref, 0=NoData)" if no_ibra else "Reference Ground Cover (IBRA-Constrained, 100+ref, 0=NoData)"
    ref_out_band.SetDescription(ref_desc)
    ref_out_band.SetNoDataValue(0)

    halfwindow = window // 2
    nchunks = (full_height + chunksize - 1) // chunksize
    print(f"Processing {full_height} lines in {nchunks} chunks (chunksize={chunksize}, window={window})...\n", flush=True)

    t_start = time.perf_counter()
    total_pixels_processed = 0

    for chunk in range(nchunks):
        t_chunk_start = time.perf_counter()
        datastartline = chunk * chunksize
        datastopline = min((chunk + 1) * chunksize, full_height)
        chunk_height = datastopline - datastartline

        pct_start = (datastartline / full_height) * 100
        print(f"[Chunk {chunk + 1}/{nchunks} - {pct_start:5.1f}%] Starting lines {datastartline}:{datastopline}...", flush=True)

        startline = max(0, datastartline - halfwindow)
        stopline = min(full_height, datastopline + halfwindow)
        win_ysize = stopline - startline

        # Read bare ground data from input
        bare_ground = in_band.ReadAsArray(
            x_origin_off, y_origin_off + startline, full_width, win_ysize
        )
        groundcover_data = convert_to_groundcover(bare_ground, nodata_in)

        # Read reference percentile data
        ref_start_y = row_offset + startline
        ref_read_y = max(0, min(ref_start_y, ref_ds.RasterYSize - win_ysize))
        percentile_data = ref_band1.ReadAsArray(
            col_offset, ref_read_y, full_width, win_ysize
        ).astype(np.int8)

        # Slice bounds within the buffered read window
        y_slice_start = datastartline - startline
        y_slice_stop = datastopline - startline

        if not no_ibra:
            # Read IBRA region codes
            ibra_data = ref_ibra_band.ReadAsArray(
                col_offset, ref_read_y, full_width, win_ysize
            ).astype(np.uint8)

            # Execute pure Python / Numba IBRA-constrained filter
            chunk_ref = refpixelfilter_ibra(
                percentile_data,
                groundcover_data,
                ibra_data,
                window=window,
                lowerquantile=lowerquantile,
                upperquantile=upperquantile,
                minvalpercent=minvalpercent,
                r_start=y_slice_start,
                r_stop=y_slice_stop
            )
        else:
            # Execute pure Python / Numba single global histogram filter
            chunk_ref = refpixelfilternodata(
                percentile_data,
                groundcover_data,
                window=window,
                lowerquantile=lowerquantile,
                upperquantile=upperquantile,
                minvalpercent=minvalpercent,
                r_start=y_slice_start,
                r_stop=y_slice_stop
            )

        chunk_gc = groundcover_data[y_slice_start:y_slice_stop]

        # Calculate DGC: 100 + groundcover - refpixel (0 is nodata)
        dgc_data = np.where(
            (chunk_ref < 0) | (chunk_gc == 0),
            0,
            100 + chunk_gc - chunk_ref
        ).astype(np.uint8)

        # Calculate Ref Cover Scaled: 100 + refpixel (0 is nodata)
        ref_data_scaled = np.where(
            chunk_ref < 0,
            0,
            100 + chunk_ref
        ).astype(np.uint8)

        # Write output bands
        dgc_band.WriteArray(dgc_data, 0, datastartline)
        ref_out_band.WriteArray(ref_data_scaled, 0, datastartline)

        dgc_band.FlushCache()
        ref_out_band.FlushCache()

        chunk_pixels = chunk_height * full_width
        total_pixels_processed += chunk_pixels
        t_chunk = time.perf_counter() - t_chunk_start
        t_elapsed = time.perf_counter() - t_start
        rate = chunk_pixels / t_chunk if t_chunk > 0 else 0
        overall_rate = total_pixels_processed / t_elapsed if t_elapsed > 0 else 0

        pct_done = (datastopline / full_height) * 100
        remaining_lines = full_height - datastopline
        remaining_pixels = remaining_lines * full_width
        eta_seconds = remaining_pixels / overall_rate if overall_rate > 0 else 0

        eta_str = format_time(eta_seconds)
        elapsed_str = format_time(t_elapsed)

        print(
            f"[Chunk {chunk + 1}/{nchunks} - {pct_done:5.1f}%] Lines {datastartline}:{datastopline} done in {t_chunk:.1f}s ({rate/1e6:.2f}M px/s) | "
            f"Elapsed: {elapsed_str} | Remaining: ~{eta_str}\n",
            flush=True
        )

    out_ds.FlushCache()

    # Calculate internal overviews using average resampling
    if not no_overviews and overview_levels:
        ovr_str = ", ".join(str(lvl) for lvl in overview_levels)
        print(f"Calculating internal overviews (average, levels: {ovr_str})...", flush=True)
        t_ovr_start = time.perf_counter()
        gdal.SetConfigOption("COMPRESS_OVERVIEW", "DEFLATE")
        gdal.SetConfigOption("PREDICTOR_OVERVIEW", "1")
        gdal.SetConfigOption("NUM_THREADS", "ALL_CPUS")
        out_ds.BuildOverviews("AVERAGE", list(overview_levels), callback=gdal.TermProgress_nocb)
        out_ds.FlushCache()
        t_ovr = time.perf_counter() - t_ovr_start
        print(f"\nOverviews completed in {t_ovr:.1f}s", flush=True)

    dgc_band = None
    ref_out_band = None
    out_ds = None
    in_ds = None
    ref_ds = None
    ibra_ds = None

    t_total = time.perf_counter() - t_start
    overall_rate = total_pixels_processed / t_total if t_total > 0 else 0
    total_str = format_time(t_total)
    print(
        f"\n[100.0%] Finished generating {output_file}\n"
        f"Total: {total_pixels_processed:,} pixels in {total_str} (avg {overall_rate/1e6:.2f}M px/s)",
        flush=True
    )

    # save to S3 if they asked for it
    if save:
        s3 = bobo3.client('s3')
        s3.upload_file(output_file, SAVE_BUCKET, f'{SAVE_DIR}/{os.path.basename(output_file)}')
        os.remove(output_file)

    return output_file


def main():
    args = parse_args()
    run_seasonal_drcm_ibra(
        input_groundcover=args.input_groundcover,
        percentile_file=args.percentile_file,
        ibra_raster=args.ibra_raster,
        output_file=args.output,
        window=args.window,
        chunksize=args.chunksize,
        lowerquantile=args.lowerquantile,
        upperquantile=args.upperquantile,
        minvalpercent=args.minvalpercent,
        test_chip=args.test_chip,
        threads=args.threads,
        overview_levels=args.overviews,
        no_overviews=args.no_overviews,
        no_ibra=args.no_ibra,
        save=args.save
    )


if __name__ == "__main__":
    main()
