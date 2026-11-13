"""
test_ibra_drcm.py

Cheat-proof verification test suite for the Dynamic Reference Cover Method (DRCM) pipeline.
Covers an independent histogram-free mathematical oracle, core Numba kernels,
I/O engines (FastS3 / FastLocal / GDAL fallbacks), halo-padded AOI subsetting,
chunk overlap reuse, portable real-satellite GeoTIFF chips (test_data/),
AWS Batch manifest/worker orchestration, and remote /vsis3/ streaming parity:
  - Portable real satellite chips (committed in test_data/, ~0.5 MB total):
      * test_data/lztmre_aus_m202606202608_dp8a2_vinterim.tif (440x440, EPSG:3577)
      * test_data/lztmre_aus_m198912202012_dixa2_change_slope_ibra.tif (520x520, offset 40,40)
      * test_data/aoi_nsw_ibra_boundary.geojson (EPSG:4326 AOI spanning IBRA 17 & 58)
  - Remote /vsis3/ Cibo Labs production datasets (skipped only under --offline):
      * /vsis3/cibo-dat-oregon/DRCM/lztmre_aus_m198912202012_dixa2_change_slope_ibra.tif
      * /vsis3/cibo-dat-oregon/GC/lztmre_aus_m202606202608_dp8a2_vinterim.tif

Run full suite (local + portable satellite chips + remote /vsis3/ parity):
  python test_ibra_drcm.py

Run completely offline (local + portable real satellite chips in test_data/, zero AWS creds required):
  python test_ibra_drcm.py --offline
"""

import argparse
import json
import os
import sys
import tempfile
import time
import types

import numpy as np
from osgeo import gdal, ogr, osr

gdal.UseExceptions()
ogr.UseExceptions()

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import refpix_numba
import refpix_numba_ibra
import run_aus_batch_worker
import seasonalDRCMIbra
import submit_batch_jobs
from seasonalDRCMIbra import (
    _SequentialBandReader,
    _parse_s3_uri,
    _parse_vsis3_uri,
    _to_gdal_path,
    convert_to_groundcover,
    derive_output_name,
    format_time,
    run_seasonal_drcm_ibra,
)

TEST_DATA_DIR = os.path.join(REPO_ROOT, "test_data")
LOCAL_REF_CHIP = os.path.join(TEST_DATA_DIR, "lztmre_aus_m198912202012_dixa2_change_slope_ibra.tif")
LOCAL_GC_CHIP = os.path.join(TEST_DATA_DIR, "lztmre_aus_m202606202608_dp8a2_vinterim.tif")
LOCAL_AOI_GEOJSON = os.path.join(TEST_DATA_DIR, "aoi_nsw_ibra_boundary.geojson")

S3_REF_TIF = "/vsis3/cibo-dat-oregon/DRCM/lztmre_aus_m198912202012_dixa2_change_slope_ibra.tif"
S3_GC_TIF = "/vsis3/cibo-dat-oregon/GC/lztmre_aus_m202606202608_dp8a2_vinterim.tif"


def _oracle_refpixel(
    cover,
    ground,
    ibra=None,
    window=15,
    lowerquantile=0.90,
    upperquantile=0.95,
    minvalpercent=0.15,
    r_slice=None,
    c_slice=None,
):
    """
    Independent histogram-free brute-force oracle built directly from 2D numpy slices
    and sorted value arrays (shares zero code with refpix_numba / refpix_numba_ibra).
    """
    nrow, ncol = cover.shape
    hw = window // 2
    r0_out, r1_out = (0, nrow) if r_slice is None else r_slice
    c0_out, c1_out = (0, ncol) if c_slice is None else c_slice
    out = np.empty((r1_out - r0_out, c1_out - c0_out), dtype=np.int8)

    for oi, r in enumerate(range(r0_out, r1_out)):
        wt = max(0, r - hw)
        wb = min(nrow - 1, r + hw)
        top_f = max(1, (r + 1) - hw)
        bot_f = min(nrow, (r + 1) + hw)
        for oj, c in enumerate(range(c0_out, c1_out)):
            wl = max(0, c - hw)
            wr = min(ncol - 1, c + hw)
            sub_cov = cover[wt : wb + 1, wl : wr + 1]
            sub_gnd = ground[wt : wb + 1, wl : wr + 1]

            if ibra is not None:
                reg = int(ibra[r, c])
                if reg == 0:
                    out[oi, oj] = -1
                    continue
                mask_reg = ibra[wt : wb + 1, wl : wr + 1] == reg
                n_reg = int(np.sum(mask_reg))
                if n_reg == 0:
                    out[oi, oj] = -1
                    continue
                vc = sub_cov[mask_reg]
                vg = sub_gnd[mask_reg]
                n_valid = int(np.sum((vc > 0) & (vg > 0)))
                if n_valid == 0 or n_valid < float(np.float32(minvalpercent)) * n_reg:
                    out[oi, oj] = -1
                    continue
                pop_cov = vc[vc > 0]
            else:
                right_f = (c + 1) + hw
                if (c + 1) < hw:
                    npix = (bot_f - top_f) * right_f
                elif right_f > ncol:
                    npix = (bot_f - top_f) * (ncol - (c + 1))
                else:
                    npix = (bot_f - top_f) * window
                vc = sub_cov.ravel()
                vg = sub_gnd.ravel()
                n_valid = int(np.sum((vc > 0) & (vc <= 101) & (vg > 0)))
                if n_valid < float(np.float32(minvalpercent)) * npix:
                    out[oi, oj] = -1
                    continue
                pop_cov = vc[(vc > 0) & (vc <= 101)]

            n_pop = len(pop_cov)
            if n_pop == 0:
                out[oi, oj] = 0
                continue
            s_vals = np.sort(pop_cov.astype(np.int64))
            lv = int(lowerquantile * n_pop)
            uv = int(upperquantile * n_pop)
            b0 = b1 = 0
            uvfound = False
            for cand in range(101, 0, -1):
                cnt_le = int(np.searchsorted(s_vals, cand, side="right"))
                if cnt_le < uv and not uvfound:
                    b1 = cand
                    uvfound = True
                if cnt_le < lv:
                    b0 = cand
                    break
            in_band = (vc >= b0 + 1) & (vc <= b1 + 1) & (vg > 0)
            nsum = int(np.sum(in_band))
            out[oi, oj] = 0 if nsum == 0 else int(np.sum(vg[in_band].astype(np.int64))) // nsum
    return out


def _write_test_geotiff(
    path, bands_u8, gt, proj_wkt, nodata=None, predictor=1, block_size=128, interleave="BAND"
):
    """Helper to write a multi-band tiled DEFLATE uint8 GeoTIFF."""
    h, w = bands_u8[0].shape
    drv = gdal.GetDriverByName("GTiff")
    opts = [
        "COMPRESS=DEFLATE",
        f"PREDICTOR={predictor}",
        "TILED=YES",
        f"BLOCKXSIZE={block_size}",
        f"BLOCKYSIZE={block_size}",
        f"INTERLEAVE={interleave}",
    ]
    ds = drv.Create(path, w, h, len(bands_u8), gdal.GDT_Byte, options=opts)
    ds.SetGeoTransform(gt)
    if proj_wkt:
        ds.SetProjection(proj_wkt)
    for idx, arr in enumerate(bands_u8, start=1):
        b = ds.GetRasterBand(idx)
        if nodata is not None:
            b.SetNoDataValue(nodata)
        b.WriteArray(arr)
    ds.FlushCache()
    ds = None


def test_1_utility_functions():
    print("\n--- Test 1: Utility & URI Functions (format_time, derive_output_name, S3 URIs) ---")
    assert format_time(0) == "0s"
    assert format_time(42) == "42s"
    assert format_time(59) == "59s"
    assert format_time(60) == "1m 00s"
    assert format_time(125) == "2m 05s"
    assert format_time(3599) == "59m 59s"
    assert format_time(3600) == "1h 00m 00s"
    assert format_time(3665) == "1h 01m 05s"

    assert derive_output_name(S3_GC_TIF) == "lztmre_aus_m202606202608_doda2_ibra_vinterim.tif"
    assert (
        derive_output_name("/vsis3/cibo-dat-oregon/GC/lztmre_qld_m201609201611_dixa2_vinterim.tif")
        == "lztmre_qld_m201609201611_doda2_ibra_vinterim.tif"
    )
    assert derive_output_name(S3_GC_TIF, no_ibra=True) == "lztmre_aus_m202606202608_doda2_vinterim.tif"
    assert (
        derive_output_name("/vsis3/cibo-dat-oregon/GC/lztmre_qld_m201609201611_dixa2_vinterim.tif", no_ibra=True)
        == "lztmre_qld_m201609201611_doda2_vinterim.tif"
    )
    assert derive_output_name("custom_mosaic.tif") == "custom_mosaic_drcm_ibra.tif"
    assert derive_output_name("custom_mosaic.tif", no_ibra=True) == "custom_mosaic_drcm.tif"

    assert _to_gdal_path("s3://cibo-dat-oregon/GC/test.tif") == "/vsis3/cibo-dat-oregon/GC/test.tif"
    assert _to_gdal_path("/tmp/local.tif") == "/tmp/local.tif"
    assert _parse_s3_uri("s3://cibo-dat-oregon/DRCM/out.tif") == ("cibo-dat-oregon", "DRCM/out.tif")
    assert _parse_vsis3_uri("/vsis3/cibo-dat-oregon/DRCM/out.tif") == ("cibo-dat-oregon", "DRCM/out.tif")

    for bad in ("s3://cibo-dat-oregon/prefix/", "s3://only-bucket", "s3:///missing-bucket"):
        raised = False
        try:
            _parse_s3_uri(bad)
        except ValueError:
            raised = True
        assert raised, f"Expected ValueError for invalid S3 URI: {bad}"

    print(">>> PASS: Utility and URI helpers verified!")


def test_2_groundcover_conversion():
    print("\n--- Test 2: Groundcover Encoding Conversion (LUT, in-place, fallback, empty) ---")
    dea_bare = np.array([[0, 25, 50], [75, 100, 255]], dtype=np.uint8)
    expected_dea = np.array([[101, 76, 51], [26, 1, 0]], dtype=np.int8)

    dea_gc = convert_to_groundcover(dea_bare, nodata_val=255)
    assert np.array_equal(dea_gc, expected_dea)

    dea_copy = dea_bare.copy()
    dea_inplace = convert_to_groundcover(dea_copy, nodata_val=255, inplace=True)
    assert dea_inplace.dtype == np.int8 and np.array_equal(dea_inplace, expected_dea)
    assert np.shares_memory(dea_copy, dea_inplace)

    dea_u16 = convert_to_groundcover(dea_bare.astype(np.uint16), nodata_val=255)
    assert np.array_equal(dea_u16, expected_dea)

    jrsrp_bare = np.array([[100, 125, 150], [175, 200, 0]], dtype=np.uint8)
    expected_jrsrp = np.array([[101, 76, 51], [26, 1, 0]], dtype=np.int8)
    assert np.array_equal(convert_to_groundcover(jrsrp_bare, nodata_val=0), expected_jrsrp)
    jrsrp_copy = jrsrp_bare.copy()
    assert np.array_equal(convert_to_groundcover(jrsrp_copy, nodata_val=0, inplace=True), expected_jrsrp)
    assert np.array_equal(convert_to_groundcover(jrsrp_bare.astype(np.uint16), nodata_val=0), expected_jrsrp)

    assert np.all(convert_to_groundcover(np.full((10, 10), 255, dtype=np.uint8), nodata_val=255) == 0)
    empty_gc = convert_to_groundcover(np.empty((0, 0), dtype=np.uint8), nodata_val=255, inplace=True)
    assert empty_gc.shape == (0, 0) and empty_gc.dtype == np.int8
    print(">>> PASS: Groundcover LUT, in-place, fallback, and edge cases verified!")


def test_3_quantile_bounds_and_mean():
    print("\n--- Test 3: Kernel Micro-Functions (_bounds, _bounds_fast, _calcmean) ---")
    hist = np.zeros(104, dtype=np.int32)
    hist[50] = 10
    hist[80] = 10

    b0_hi, b1_hi = refpix_numba_ibra._bounds(hist[:103], 101, 0.90, 0.95)
    b0_hi_f, b1_hi_f = refpix_numba._bounds_fast(hist, 20, 101, 0.90, 0.95)
    assert (b0_hi, b1_hi) == (79, 79) and (b0_hi_f, b1_hi_f) == (79, 79)

    b0_mid, b1_mid = refpix_numba_ibra._bounds(hist[:103], 101, 0.40, 0.60)
    b0_mid_f, b1_mid_f = refpix_numba._bounds_fast(hist, 20, 101, 0.40, 0.60)
    assert (b0_mid, b1_mid) == (49, 79) and (b0_mid_f, b1_mid_f) == (49, 79)

    # Multi-bin weighted average inside [lowerb+1 .. upperb+1] + integer floor rounding
    dryhist = np.zeros(103, dtype=np.int32)
    drysum = np.zeros(103, dtype=np.int32)
    dryhist[50] = 999  # Excluded (below lowerb+1 = 51)
    drysum[50] = 99900
    dryhist[51] = 4
    drysum[51] = 200
    dryhist[52] = 6
    drysum[52] = 407  # Total in [51, 52] = 607 // 10 = 60
    dryhist[53] = 999  # Excluded (above upperb+1 = 52)
    drysum[53] = 99900
    assert refpix_numba_ibra._calcmean(drysum, dryhist, 50, 51) == 60
    assert refpix_numba._calcmean(drysum, dryhist, 50, 51) == 60

    dryhist.fill(0)
    drysum.fill(0)
    assert refpix_numba_ibra._calcmean(drysum, dryhist, 10, 20) == 0
    assert refpix_numba._calcmean(drysum, dryhist, 10, 20) == 0
    assert refpix_numba_ibra._bounds(np.zeros(103, dtype=np.int32), 101, 0.90, 0.95) == (0, 0)
    assert refpix_numba._bounds_fast(np.zeros(104, dtype=np.int32), 0, 101, 0.90, 0.95) == (0, 0)
    print(">>> PASS: Quantile bounds, multi-bin mean, and edge cases verified!")


def test_4_oracle_invariance_and_slicing():
    print("\n--- Test 4: Independent Oracle Parity, Single-Region Invariance & Tiled Kernels ---")
    rng = np.random.default_rng(2026)
    oh, ow = 47, 61
    cov_o = rng.integers(0, 102, size=(oh, ow), dtype=np.int8)
    gnd_o = rng.integers(0, 102, size=(oh, ow), dtype=np.int8)
    cov_o[5:12, 10:20] = 0
    gnd_o[20:28, 30:40] = 0
    ibra_o = rng.choice([0, 17, 58, 89], size=(oh, ow)).astype(np.uint8)

    for win in (11, 15):
        oracle_ib = _oracle_refpixel(cov_o, gnd_o, ibra_o, window=win, minvalpercent=0.15)
        numba_ib = refpix_numba_ibra.refpixelfilter_ibra(cov_o, gnd_o, ibra_o, window=win, minvalpercent=0.15)
        assert np.array_equal(oracle_ib, numba_ib), f"IBRA kernel vs independent oracle mismatch (window={win})"

        oracle_gl = _oracle_refpixel(cov_o, gnd_o, None, window=win, minvalpercent=0.15)
        numba_gl = refpix_numba.refpixelfilternodata(cov_o, gnd_o, window=win, minvalpercent=0.15)
        assert np.array_equal(oracle_gl, numba_gl), f"Global kernel vs independent oracle mismatch (window={win})"

    # Large non-square multi-tile & U16/I32 invariance
    np.random.seed(42)
    nrow, ncol = 190, 215
    cover = np.random.randint(0, 102, size=(nrow, ncol), dtype=np.int8)
    ground = np.random.randint(0, 102, size=(nrow, ncol), dtype=np.int8)
    cover[:30, :30] = 0
    ground[:30, :30] = 0
    ibra = np.full((nrow, ncol), 17, dtype=np.uint8)

    window, lq, uq, minval = 31, 0.90, 0.95, 0.15
    res_global = refpix_numba.refpixelfilternodata(cover, ground, window, lq, uq, minval)
    res_ibra = refpix_numba_ibra.refpixelfilter_ibra(cover, ground, ibra, window, lq, uq, minval)
    assert int(np.sum(res_global > 0)) > 10000 and int(np.sum(res_global == -1)) > 100
    assert np.array_equal(res_global, res_ibra), "res_global and res_ibra must be bit-for-bit identical"

    r0, r1, c0, c1 = 23, 169, 19, 191
    sliced_small_tiles = refpix_numba.refpixelfilternodata(
        cover, ground, window, lq, uq, minval,
        r_start=r0, r_stop=r1, c_start=c0, c_stop=c1,
        strip_h=32, col_tile_w=48,
    )
    assert np.array_equal(sliced_small_tiles, res_global[r0:r1, c0:c1]), "Sub-window or tile-size mismatch!"

    res_i32 = refpix_numba._KERNEL_I32(
        cover, ground, window, lq, uq, minval, 0, nrow, 0, ncol, 64, 64
    )
    assert np.array_equal(res_i32, res_global), "_KERNEL_I32 vs _KERNEL_U16 mismatch!"
    print(">>> PASS: Independent brute-force oracle, sub-window slicing, tiles, and U16/I32 match 100%!")


def test_5_boundary_isolation():
    print("\n--- Test 5: Row & Column Boundary Isolation Test ---")
    size = 100
    cover = np.zeros((size, size), dtype=np.int8)
    ground = np.zeros((size, size), dtype=np.int8)
    ibra = np.zeros((size, size), dtype=np.uint8)

    # Quadrant-specific IBRA regions + gradients so axis swaps cannot hide
    cover[:50, :50] = 80
    ground[:50, :50] = 70
    ibra[:50, :50] = 10
    cover[:50:6, :50] = 95
    ground[:50:6, :50] = 90

    cover[:50, 50:] = 20
    ground[:50, 50:] = 15
    ibra[:50, 50:] = 20
    cover[:50:6, 50:] = 35
    ground[:50:6, 50:] = 30

    cover[50:, :] = 55
    ground[50:, :] = 45
    ibra[50:, :] = 30
    cover[50::6, :] = 65
    ground[50::6, :] = 60

    window, lq, uq, minval = 21, 0.90, 0.95, 0.05
    res_ibra = refpix_numba_ibra.refpixelfilter_ibra(cover, ground, ibra, window, lq, uq, minval)
    res_global = refpix_numba.refpixelfilternodata(cover, ground, window, lq, uq, minval)

    # Vertical split check (IBRA 10 vs 20 across col 49/50)
    assert np.all(res_ibra[10:40, 48:50] == 90)
    assert np.all(res_ibra[10:40, 50:52] == 30)
    assert np.all(res_global[10:40, 50:52] > 60)

    # Horizontal split check (IBRA 20 vs 30 across row 49/50)
    assert np.all(res_ibra[48:50, 60:90] == 30)
    assert np.all(res_ibra[50:52, 60:90] == 60)
    print(">>> PASS: Zero horizontal or vertical cross-boundary contamination proven!")


def test_6_ocean_and_valid_fraction():
    print("\n--- Test 6: Ocean Masking, Empty-Block Skipping & minvalpercent Thresholds ---")
    size = 50
    cover = np.full((size, size), 60, dtype=np.int8)
    ground = np.full((size, size), 50, dtype=np.int8)
    ibra = np.full((size, size), 15, dtype=np.uint8)

    ibra[:25, :25] = 0
    res = refpix_numba_ibra.refpixelfilter_ibra(cover, ground, ibra, window=11)
    assert np.all(res[:25, :25] == -1)
    assert np.all(res[35:, 35:] == 50)

    cover.fill(0)
    ground.fill(0)
    ibra.fill(15)
    cover[20:25, 20:26] = 60  # 30 / 121 = 24.8% valid
    ground[20:25, 20:26] = 50

    res_insuff = refpix_numba_ibra.refpixelfilter_ibra(cover, ground, ibra, window=11, minvalpercent=0.30)
    res_suff = refpix_numba_ibra.refpixelfilter_ibra(cover, ground, ibra, window=11, minvalpercent=0.15)
    assert res_insuff[22, 22] == -1 and res_suff[22, 22] == 50
    assert res_insuff[5, 5] == -1 and res_suff[5, 5] == -1

    sparse_h, sparse_w = 600, 600
    sp_cov = np.zeros((sparse_h, sparse_w), dtype=np.int8)
    sp_gnd = np.zeros((sparse_h, sparse_w), dtype=np.int8)
    sp_cov[300:360, 300:360] = 72
    sp_gnd[300:360, 300:360] = 64

    blocks = refpix_numba._build_valid_blocks(sp_cov, sp_gnd, 256)
    assert blocks[0, 0] == 0 and blocks[1, 1] == 1

    sp_fast = refpix_numba.refpixelfilternodata(sp_cov, sp_gnd, window=31, minvalpercent=0.15)
    sp_cov_unskipped = sp_cov.copy()
    sp_cov_unskipped[0, 0] = 50
    sp_gnd_unskipped = sp_gnd.copy()
    sp_gnd_unskipped[0, 0] = 50
    sp_unskipped = refpix_numba.refpixelfilternodata(
        sp_cov_unskipped, sp_gnd_unskipped, window=31, minvalpercent=0.15
    )
    assert np.all(sp_fast[:250, :250] == -1)
    assert np.all(sp_fast[:, -1] == 0)
    assert np.array_equal(sp_fast[32:, :], sp_unskipped[32:, :])
    assert np.array_equal(sp_fast[:, 32:], sp_unskipped[:, 32:])
    assert np.all(sp_fast[300:360, 300:360] == 64)

    # Check boundary of sparse island against independent brute-force oracle
    oracle_edge = _oracle_refpixel(
        sp_cov, sp_gnd, None, window=31, minvalpercent=0.15, r_slice=(280, 320), c_slice=(290, 340)
    )
    assert np.array_equal(sp_fast[280:320, 290:340], oracle_edge)
    print(">>> PASS: Ocean masking, empty-block skip, and validity thresholds verified!")


def test_7_local_satellite_multi_ibra():
    print("\n--- Test 7: Portable Real Satellite Multi-IBRA Test (test_data/ chips, offset 40,40) ---")
    assert os.path.exists(LOCAL_GC_CHIP) and os.path.exists(LOCAL_REF_CHIP)
    with tempfile.TemporaryDirectory(prefix="drcm_test7_") as tmpdir:
        out_tif = os.path.join(tmpdir, "local_sat_multi_ibra.tif")
        run_seasonal_drcm_ibra(
            input_groundcover=LOCAL_GC_CHIP,
            percentile_file=LOCAL_REF_CHIP,
            output_file=out_tif,
            window=61,
            chunksize=200,
        )

        ds_res = gdal.Open(out_tif)
        ds_gc = gdal.Open(LOCAL_GC_CHIP)
        ds_ref = gdal.Open(LOCAL_REF_CHIP)
        assert ds_res.RasterCount == 2
        assert ds_res.RasterXSize == ds_gc.RasterXSize == 440
        assert ds_res.RasterYSize == ds_gc.RasterYSize == 440
        assert ds_res.GetGeoTransform() == ds_gc.GetGeoTransform()

        b1, b2 = ds_res.GetRasterBand(1), ds_res.GetRasterBand(2)
        assert "IBRA-Constrained" in b1.GetDescription()
        assert "IBRA-Constrained" in b2.GetDescription()
        assert b1.GetOverviewCount() == 4 and b2.GetOverviewCount() == 4

        dgc = b1.ReadAsArray()
        ref_scaled = b2.ReadAsArray()
        assert int(np.sum(dgc > 0)) > 120000 and int(np.sum(ref_scaled > 0)) > 120000

        # Offset alignment check: GC starts at pixel (40, 40) inside REF chip
        perc = ds_ref.GetRasterBand(1).ReadAsArray(40, 40, 440, 440).astype(np.int8)
        ibra = ds_ref.GetRasterBand(2).ReadAsArray(40, 40, 440, 440)
        bare = ds_gc.GetRasterBand(1).ReadAsArray()
        gc = convert_to_groundcover(bare)

        # Verify against independent histogram-free oracle across real IBRA boundary (rows 150..290, cols 150..290)
        oracle_sub = _oracle_refpixel(
            perc, gc, ibra, window=61, r_slice=(150, 290), c_slice=(150, 290)
        )
        oracle_ref_sub = np.where(oracle_sub < 0, 0, 100 + oracle_sub).astype(np.uint8)
        oracle_dgc_sub = np.where((oracle_sub < 0) | (gc[150:290, 150:290] == 0), 0, 100 + gc[150:290, 150:290] - oracle_sub).astype(np.uint8)
        assert np.array_equal(ref_scaled[150:290, 150:290], oracle_ref_sub)
        assert np.array_equal(dgc[150:290, 150:290], oracle_dgc_sub)

        # Verify real Darling Riverine Plains (17) vs Cobar Peneplain (58) statistical divergence
        assert 17 in np.unique(ibra) and 58 in np.unique(ibra)
        mean_17 = float(np.mean(ref_scaled[(ibra == 17) & (ref_scaled > 0)]))
        mean_58 = float(np.mean(ref_scaled[(ibra == 58) & (ref_scaled > 0)]))
        assert abs(mean_17 - mean_58) > 0.1
        ds_res = ds_gc = ds_ref = None
    print(">>> PASS: Real satellite multi-IBRA chip + spatial offset (40,40) + oracle parity verified!")


def test_8_local_satellite_no_ibra_and_geojson():
    print("\n--- Test 8: Portable Real Satellite --no-ibra & Halo-Padded GeoJSON Test ---")
    assert os.path.exists(LOCAL_AOI_GEOJSON)
    with tempfile.TemporaryDirectory(prefix="drcm_test8_") as tmpdir:
        full_no_ibra = os.path.join(tmpdir, "local_sat_no_ibra.tif")
        full_ibra = os.path.join(tmpdir, "local_sat_ibra.tif")
        aoi_no_ibra = os.path.join(tmpdir, "aoi_no_ibra.tif")
        aoi_ibra = os.path.join(tmpdir, "aoi_ibra.tif")

        run_seasonal_drcm_ibra(
            LOCAL_GC_CHIP, LOCAL_REF_CHIP, output_file=full_no_ibra,
            window=61, chunksize=200, no_ibra=True, no_overviews=True,
        )
        run_seasonal_drcm_ibra(
            LOCAL_GC_CHIP, LOCAL_REF_CHIP, output_file=full_ibra,
            window=61, chunksize=200, no_ibra=False, no_overviews=True,
        )
        run_seasonal_drcm_ibra(
            LOCAL_GC_CHIP, LOCAL_REF_CHIP, output_file=aoi_no_ibra,
            window=61, chunksize=140, geojson=LOCAL_AOI_GEOJSON, no_ibra=True, no_overviews=True,
        )
        run_seasonal_drcm_ibra(
            LOCAL_GC_CHIP, LOCAL_REF_CHIP, output_file=aoi_ibra,
            window=61, chunksize=140, geojson=LOCAL_AOI_GEOJSON, no_ibra=False, no_overviews=True,
        )

        ds_full_ni = gdal.Open(full_no_ibra)
        assert "Unconstrained" in ds_full_ni.GetRasterBand(1).GetDescription()
        dgc_ni = ds_full_ni.GetRasterBand(1).ReadAsArray()
        ref_ni = ds_full_ni.GetRasterBand(2).ReadAsArray()

        ds_gc = gdal.Open(LOCAL_GC_CHIP)
        ds_ref = gdal.Open(LOCAL_REF_CHIP)
        gt = ds_gc.GetGeoTransform()
        perc = ds_ref.GetRasterBand(1).ReadAsArray(40, 40, 440, 440).astype(np.int8)
        gc = convert_to_groundcover(ds_gc.GetRasterBand(1).ReadAsArray())

        # Independent oracle parity on interior real satellite pixels
        oracle_sub = _oracle_refpixel(perc, gc, None, window=61, r_slice=(160, 280), c_slice=(160, 280))
        exp_ref = np.where(oracle_sub < 0, 0, 100 + oracle_sub).astype(np.uint8)
        exp_dgc = np.where((oracle_sub < 0) | (gc[160:280, 160:280] == 0), 0, 100 + gc[160:280, 160:280] - oracle_sub).astype(np.uint8)
        assert np.array_equal(ref_ni[160:280, 160:280], exp_ref)
        assert np.array_equal(dgc_ni[160:280, 160:280], exp_dgc)

        # Verify halo-padded EPSG:4326 GeoJSON AOI actually subsetted tightly and matched 0-diff
        ds_full_ib = gdal.Open(full_ibra)
        for full_ds, aoi_path in ((ds_full_ni, aoi_no_ibra), (ds_full_ib, aoi_ibra)):
            sub_ds = gdal.Open(aoi_path)
            sgt = sub_ds.GetGeoTransform()
            sw, sh = sub_ds.RasterXSize, sub_ds.RasterYSize
            assert 275 <= sw <= 290 and 275 <= sh <= 290, f"Unexpected AOI subset size: {sw}x{sh}"
            x0 = int(round((sgt[0] - gt[0]) / gt[1]))
            y0 = int(round((sgt[3] - gt[3]) / gt[5]))
            assert 75 <= x0 <= 85 and 75 <= y0 <= 85
            assert np.array_equal(sub_ds.GetRasterBand(1).ReadAsArray(), full_ds.GetRasterBand(1).ReadAsArray()[y0 : y0 + sh, x0 : x0 + sw])
            assert np.array_equal(sub_ds.GetRasterBand(2).ReadAsArray(), full_ds.GetRasterBand(2).ReadAsArray()[y0 : y0 + sh, x0 : x0 + sw])
            sub_ds = None
        ds_full_ni = ds_full_ib = ds_gc = ds_ref = None
    print(">>> PASS: Real satellite --no-ibra + GeoJSON halo AOI verified offline!")


def test_9_local_fast_tiled_reader_and_predictor_fallback():
    print("\n--- Test 9: FastLocal (os.pread + zlib) Multi-Band & Fallback Verification ---")
    with tempfile.TemporaryDirectory(prefix="drcm_test9_") as tmpdir:
        local_in = os.path.join(tmpdir, "in_pred1.tif")
        local_pred2 = os.path.join(tmpdir, "in_pred2.tif")
        local_ref_band = os.path.join(tmpdir, "ref_band.tif")
        local_ref_pixel = os.path.join(tmpdir, "ref_pixel.tif")
        local_out = os.path.join(tmpdir, "out.tif")

        rng = np.random.default_rng(12345)
        h, w = 300, 300
        bare = rng.integers(0, 101, size=(h, w), dtype=np.uint8)
        bare[20:40, 20:40] = 255
        ref_arr = rng.integers(1, 102, size=(h, w), dtype=np.uint8)
        ref_arr[128:256, 128:256] = 0
        ibra_arr = np.full((h, w), 17, dtype=np.uint8)
        gt = (0.0, 30.0, 0.0, 0.0, 0.0, -30.0)

        _write_test_geotiff(local_in, [bare], gt, "", nodata=255, predictor=1, block_size=128, interleave="BAND")
        _write_test_geotiff(local_pred2, [bare], gt, "", nodata=255, predictor=2, block_size=128, interleave="BAND")
        _write_test_geotiff(local_ref_band, [ref_arr, ibra_arr], gt, "", nodata=0, predictor=1, block_size=128, interleave="BAND")
        _write_test_geotiff(local_ref_pixel, [ref_arr, ibra_arr], gt, "", nodata=0, predictor=1, block_size=128, interleave="PIXEL")

        # 1. INTERLEAVE=BAND 2-band FastLocal activates cleanly on input and both ref/ibra bands
        rdr = _SequentialBandReader(local_in, local_ref_band, ibra_band_idx=2, io_workers=4)
        try:
            assert rdr.in_fast is not None and rdr.in_fast.local_path == local_in
            assert rdr.ref_fast is not None and rdr.ref_fast.local_path == local_ref_band
            assert rdr.ibra_fast is not None and rdr.ibra_fast.local_path == local_ref_band
            assert np.array_equal(rdr.read_in_window((0, 0, w, h)), bare)
            got_ref, got_ib = rdr.read_ref_windows((0, 0, w, h), need_ibra=True)
            assert np.array_equal(got_ref, ref_arr.view(np.int8))
            assert np.array_equal(got_ib, ibra_arr)
        finally:
            rdr.close()

        # 2. PREDICTOR=2 input and INTERLEAVE=PIXEL 2-band ref cleanly fall back to GDAL
        rdr2 = _SequentialBandReader(local_pred2, local_ref_pixel, ibra_band_idx=2, io_workers=4)
        try:
            assert rdr2.in_fast is None, "PREDICTOR=2 must fall back to GDAL"
            assert rdr2.ref_fast is None, "INTERLEAVE=PIXEL 2-band must fall back to GDAL"
            assert rdr2.ibra_fast is None
            assert np.array_equal(rdr2.read_in_window((0, 0, w, h)), bare)
            got_ref2, got_ib2 = rdr2.read_ref_windows((0, 0, w, h), need_ibra=True)
            assert np.array_equal(got_ref2, ref_arr.view(np.int8)) and np.array_equal(got_ib2, ibra_arr)
        finally:
            rdr2.close()

        # 3. End-to-end local FastLocal pipeline vs direct kernel
        run_seasonal_drcm_ibra(
            input_groundcover=local_in,
            percentile_file=local_ref_band,
            output_file=local_out,
            window=21,
            chunksize=150,
            no_ibra=True,
            no_overviews=True,
        )
        ds_out = gdal.Open(local_out)
        dgc = ds_out.GetRasterBand(1).ReadAsArray()
        ref_scaled = ds_out.GetRasterBand(2).ReadAsArray()
        ds_out = None

        gc = convert_to_groundcover(bare, 255)
        direct_ref = refpix_numba.refpixelfilternodata(ref_arr.view(np.int8), gc, window=21)
        exp_ref = np.where(direct_ref < 0, 0, 100 + direct_ref).astype(np.uint8)
        exp_dgc = np.where((direct_ref < 0) | (gc == 0), 0, 100 + gc - direct_ref).astype(np.uint8)
        assert np.array_equal(dgc, exp_dgc) and np.array_equal(ref_scaled, exp_ref)
    print(">>> PASS: FastLocal BAND activation + PREDICTOR=2/PIXEL fallback verified!")


def test_10_multichunk_and_halo_aoi_parity():
    print("\n--- Test 10: Multi-Chunk Overlap, Spatial Offset Alignment & Strict Halo AOI Parity ---")
    with tempfile.TemporaryDirectory(prefix="drcm_test10_") as tmpdir:
        srs_3577 = osr.SpatialReference()
        srs_3577.ImportFromEPSG(3577)
        srs_3577.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        wkt_3577 = srs_3577.ExportToWkt()

        srs_4326 = osr.SpatialReference()
        srs_4326.ImportFromEPSG(4326)
        srs_4326.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        # Larger reference grid so input_groundcover has non-zero alignment offset (col=40, row=35)
        ref_h, ref_w = 500, 520
        off_x, off_y = 40, 35
        h, w = 420, 440
        ref_gt = (1000000.0, 30.0, 0.0, -3200000.0, 0.0, -30.0)
        in_gt = (ref_gt[0] + off_x * 30.0, 30.0, 0.0, ref_gt[3] - off_y * 30.0, 0.0, -30.0)

        rng = np.random.default_rng(777)
        bare = rng.integers(5, 95, size=(h, w), dtype=np.uint8)
        bare[:40, :40] = 255
        perc = rng.integers(10, 101, size=(ref_h, ref_w), dtype=np.uint8)
        ibra = np.where(np.arange(ref_w)[None, :] < (off_x + 220), 17, 58).astype(np.uint8)

        in_tif = os.path.join(tmpdir, "aus_synth_dp8a2.tif")
        ref_tif = os.path.join(tmpdir, "aus_synth_ref_ibra.tif")
        _write_test_geotiff(in_tif, [bare], in_gt, wkt_3577, nodata=255, predictor=1, block_size=128, interleave="BAND")
        _write_test_geotiff(ref_tif, [perc, ibra], ref_gt, wkt_3577, nodata=0, predictor=1, block_size=128, interleave="BAND")

        for no_ibra in (True, False):
            tag = "noibra" if no_ibra else "ibra"
            full_1 = os.path.join(tmpdir, f"full_single_{tag}.tif")
            full_n = os.path.join(tmpdir, f"full_multi_{tag}.tif")
            run_seasonal_drcm_ibra(
                in_tif, ref_tif, output_file=full_1, window=31, chunksize=1024,
                no_ibra=no_ibra, no_overviews=True,
            )
            run_seasonal_drcm_ibra(
                in_tif, ref_tif, output_file=full_n, window=31, chunksize=96,
                no_ibra=no_ibra, no_overviews=True,
            )
            d1 = gdal.Open(full_1)
            dn = gdal.Open(full_n)
            for b in (1, 2):
                assert np.array_equal(d1.GetRasterBand(b).ReadAsArray(), dn.GetRasterBand(b).ReadAsArray())
            d1 = dn = None

        # AOI cols 150..310 (160 px wide), rows 120..300 (180 px tall) crossing IBRA col 220
        minx = in_gt[0] + 150 * 30.0
        maxx = in_gt[0] + 310 * 30.0
        maxy = in_gt[3] - 120 * 30.0
        miny = in_gt[3] - 300 * 30.0

        # Prove halo padding IS genuinely required (unpadded test_chip misses border pixels!)
        nopad_tif = os.path.join(tmpdir, "nopad_chip.tif")
        run_seasonal_drcm_ibra(
            in_tif, ref_tif, output_file=nopad_tif, window=31, chunksize=200,
            test_chip=[150, 120, 160, 180], no_ibra=True, no_overviews=True,
        )
        np_ds = gdal.Open(nopad_tif)
        full_ds_tmp = gdal.Open(os.path.join(tmpdir, "full_single_noibra.tif"))
        assert not np.array_equal(
            np_ds.GetRasterBand(1).ReadAsArray()[0, :],
            full_ds_tmp.GetRasterBand(1).ReadAsArray()[120, 150:310],
        ), "Unpadded crop unexpectedly matched haloed interior!"
        np_ds = full_ds_tmp = None

        to_4326 = osr.CoordinateTransformation(srs_3577, srs_4326)
        ring_coords = []
        for px, py in ((minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy), (minx, miny)):
            lon, lat, _ = to_4326.TransformPoint(px, py)
            ring_coords.append([lon, lat])
        geojson_path = os.path.join(tmpdir, "aoi_4326.geojson")
        with open(geojson_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "type": "FeatureCollection",
                    "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
                    "features": [
                        {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [ring_coords]}}
                    ],
                },
                f,
            )

        for no_ibra in (True, False):
            tag = "noibra" if no_ibra else "ibra"
            ref_full_ds = gdal.Open(os.path.join(tmpdir, f"full_single_{tag}.tif"))
            full_b1 = ref_full_ds.GetRasterBand(1).ReadAsArray()
            full_b2 = ref_full_ds.GetRasterBand(2).ReadAsArray()
            ref_full_ds = None

            out_bbox = os.path.join(tmpdir, f"bbox_{tag}.tif")
            out_gj = os.path.join(tmpdir, f"gj_{tag}.tif")
            run_seasonal_drcm_ibra(
                in_tif, ref_tif, output_file=out_bbox, window=31, chunksize=80,
                bbox=[minx, miny, maxx, maxy], no_ibra=no_ibra, no_overviews=True,
            )
            run_seasonal_drcm_ibra(
                in_tif, ref_tif, output_file=out_gj, window=31, chunksize=80,
                geojson=geojson_path, no_ibra=no_ibra, no_overviews=True,
            )

            for p in (out_bbox, out_gj):
                sub_ds = gdal.Open(p)
                sgt = sub_ds.GetGeoTransform()
                sw, sh = sub_ds.RasterXSize, sub_ds.RasterYSize
                # Strict anti-cheat checks: actually subsetted + strictly covers requested AOI + tight fit
                assert 159 <= sw <= 165 and 179 <= sh <= 185, f"Subset dims {sw}x{sh} out of expected tight AOI range"
                out_minx, out_maxx = sgt[0], sgt[0] + sw * sgt[1]
                out_maxy, out_miny = sgt[3], sgt[3] + sh * sgt[5]
                assert out_minx <= minx + 1e-6 and out_maxx >= maxx - 1e-6
                assert out_maxy >= maxy - 1e-6 and out_miny <= miny + 1e-6
                x0 = int(round((sgt[0] - in_gt[0]) / in_gt[1]))
                y0 = int(round((sgt[3] - in_gt[3]) / in_gt[5]))
                assert np.array_equal(sub_ds.GetRasterBand(1).ReadAsArray(), full_b1[y0 : y0 + sh, x0 : x0 + sw])
                assert np.array_equal(sub_ds.GetRasterBand(2).ReadAsArray(), full_b2[y0 : y0 + sh, x0 : x0 + sw])
                sub_ds = None
    print(">>> PASS: Multi-chunk overlap, offset (40,35) alignment & strict halo AOI parity verified!")


def test_11_batch_worker_and_submission():
    print("\n--- Test 11: Batch Submission, S3 --save/-o Upload & Worker Content Verification ---")
    with tempfile.TemporaryDirectory(prefix="drcm_test11_") as tmpdir:
        # 1. Verify load_env_file parsing (spaces around =, quotes, comments, no-clobber)
        env_p = os.path.join(tmpdir, "test.env")
        with open(env_p, "w", encoding="utf-8") as f:
            f.write('# comment\nTEST_DRCM_K1 = "val_quoted"\nTEST_DRCM_K2=plain_val\nTEST_DRCM_KEEP = "override"\n')
        try:
            os.environ.pop("TEST_DRCM_K1", None)
            os.environ.pop("TEST_DRCM_K2", None)
            os.environ["TEST_DRCM_KEEP"] = "existing"
            submit_batch_jobs.load_env_file(env_p)
            assert os.environ["TEST_DRCM_K1"] == "val_quoted" and os.environ["TEST_DRCM_K2"] == "plain_val"
            assert os.environ["TEST_DRCM_KEEP"] == "existing"
        finally:
            os.environ.pop("TEST_DRCM_K1", None)
            os.environ.pop("TEST_DRCM_K2", None)
            os.environ.pop("TEST_DRCM_KEEP", None)

        # 2. Exact round-robin sharding, filtering, command construction & Graviton env
        sample_items = [
            ("GC/lztmre_aus_m200909200911_dp8a2.tif", 10_000_000_000),
            ("GC/lztmre_aus_m200912201002_dp8a2.tif", 12_000_000_000),
            ("GC/lztmre_aus_m201003201005_dp8a2.tif", 13_000_000_000),
            ("GC/lztmre_aus_m202606202608_dp8a2_vinterim.tif", 14_000_000_000),
            ("GC/unrelated_file.tif", 1_000_000),
        ]
        filtered = submit_batch_jobs.filter_season_items(sample_items, start_ym="200912", end_ym="202608")
        assert [k for k, _ in filtered] == [
            "GC/lztmre_aus_m200912201002_dp8a2.tif",
            "GC/lztmre_aus_m201003201005_dp8a2.tif",
            "GC/lztmre_aus_m202606202608_dp8a2_vinterim.tif",
        ]

        manifest, shard_bytes = submit_batch_jobs.build_shard_manifest(
            filtered, num_shards=2, geojson="field.geojson", output_prefix="DRCM/nsw_field"
        )
        assert manifest["shards"][0] == [
            "GC/lztmre_aus_m200912201002_dp8a2.tif",
            "GC/lztmre_aus_m202606202608_dp8a2_vinterim.tif",
        ]
        assert manifest["shards"][1] == ["GC/lztmre_aus_m201003201005_dp8a2.tif"]
        assert shard_bytes == [26_000_000_000, 13_000_000_000]

        cmd_single = submit_batch_jobs.build_worker_command(
            "s3://cibo-dat-oregon/DRCM/_batch_scripts/v4", explicit_shard_index=6
        )
        assert "--shard-index 6" in cmd_single[2]
        assert "--shard-index" not in submit_batch_jobs.build_worker_command("s3://cibo-dat-oregon/DRCM/_batch_scripts/v4")[2]

        env_map = {e["name"]: e["value"] for e in submit_batch_jobs.build_batch_environment()}
        assert env_map["NUMBA_CPU_NAME"] == "generic" and env_map["NUMBA_CPU_FEATURES"] == "+neon"
        assert "0" in env_map["GDAL_HTTP_RETRY_CODES"].split(",")

        # 3. Verify --save upload + local file cleanup & -o s3:// upload with mocked boto3
        gt = (0.0, 30.0, 0.0, 0.0, 0.0, -30.0)
        bare1 = np.full((128, 128), 30, dtype=np.uint8)
        bare2 = np.full((128, 128), 60, dtype=np.uint8)
        ref = np.full((128, 128), 80, dtype=np.uint8)
        bare2[:10, :10] = 255
        bare2[30:40, 30:40] = 10  # Higher groundcover patch for non-constant output

        in1 = os.path.join(tmpdir, "lztmre_aus_m202503202505_dp8a2.tif")
        in2 = os.path.join(tmpdir, "lztmre_aus_m202506202508_dp8a2.tif")
        ref_tif = os.path.join(tmpdir, "ref.tif")
        _write_test_geotiff(in1, [bare1], gt, "", nodata=255, predictor=1, block_size=128)
        _write_test_geotiff(in2, [bare2], gt, "", nodata=255, predictor=1, block_size=128)
        _write_test_geotiff(ref_tif, [ref], gt, "", nodata=0, predictor=1, block_size=128)

        saved_copies = {}

        class _FakeBotoS3:
            def upload_file(self, Filename, Bucket, Key, Config=None):
                dst = os.path.join(tmpdir, f"captured_{len(saved_copies)}.tif")
                with open(Filename, "rb") as src, open(dst, "wb") as out:
                    out.write(src.read())
                saved_copies[(Bucket, Key)] = dst

        fake_boto = types.ModuleType("boto3")
        fake_boto.client = lambda svc: _FakeBotoS3()
        fake_s3_pkg = types.ModuleType("boto3.s3")
        fake_xfer = types.ModuleType("boto3.s3.transfer")
        fake_xfer.TransferConfig = lambda **kw: kw
        orig_mods = {k: sys.modules.get(k) for k in ("boto3", "boto3.s3", "boto3.s3.transfer")}
        sys.modules["boto3"] = fake_boto
        sys.modules["boto3.s3"] = fake_s3_pkg
        sys.modules["boto3.s3.transfer"] = fake_xfer
        try:
            save_local = os.path.join(tmpdir, "save_target_doda2.tif")
            ret_uri = run_seasonal_drcm_ibra(
                in2, ref_tif, output_file=save_local, window=11, chunksize=128,
                no_ibra=True, no_overviews=True, save=True,
            )
            assert ret_uri == "s3://cibo-dat-oregon/DRCM/save_target_doda2.tif"
            assert not os.path.exists(save_local), "--save must delete local output after S3 upload"
            assert ("cibo-dat-oregon", "DRCM/save_target_doda2.tif") in saved_copies
        finally:
            for k, v in orig_mods.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

        # 4. Run run_aus_batch_worker end-to-end and verify uploaded GeoTIFF pixel contents!
        manifest_path = os.path.join(tmpdir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "bucket": "test-bucket",
                    "ref_key": "DRCM/ref.tif",
                    "output_prefix": "DRCM/aus/",
                    "shards": [[in1, in2], [in2]],
                },
                f,
            )

        uploaded_bytes = {}

        class _FakeWorkerS3:
            def head_object(self, Bucket, Key):
                if Key.endswith("202503202505_doda2.tif"):
                    return {"ContentLength": 500_000_000}
                if Key in uploaded_bytes:
                    return {"ContentLength": uploaded_bytes[Key][0]}
                raise FileNotFoundError(Key)

            def upload_file(self, Filename, Bucket, Key, Config=None):
                captured = os.path.join(tmpdir, os.path.basename(Key) + ".uploaded")
                with open(Filename, "rb") as src, open(captured, "wb") as out:
                    data = src.read()
                    out.write(data)
                uploaded_bytes[Key] = (len(data), captured)

        fake_s3 = _FakeWorkerS3()
        orig_build_s3 = run_aus_batch_worker._build_s3_client
        orig_cache_fn = run_aus_batch_worker._ensure_local_ref_cache
        orig_run = seasonalDRCMIbra.run_seasonal_drcm_ibra
        orig_argv = sys.argv
        try:
            run_aus_batch_worker._build_s3_client = lambda: fake_s3
            run_aus_batch_worker._ensure_local_ref_cache = lambda s3, b, k, d: (ref_tif, False)

            def _wrapped_run(**kw):
                ig = kw["input_groundcover"]
                if ig.startswith("s3://test-bucket/"):
                    kw["input_groundcover"] = ig[len("s3://test-bucket/"):]
                return orig_run(**kw)

            seasonalDRCMIbra.run_seasonal_drcm_ibra = _wrapped_run
            target_key = "DRCM/aus/lztmre_aus_m202506202508_doda2.tif"
            sys.argv = [
                "run_aus_batch_worker.py",
                "--manifest", manifest_path,
                "--shard-index", "0",
                "--window", "11",
                "--chunksize", "128",
                "--minvalpercent", "0.10",
                "--no-ibra",
                "--no-overviews",
                "--work-dir", os.path.join(tmpdir, "work"),
                "--min-output-bytes", "100",
            ]
            run_aus_batch_worker.main()

            assert list(uploaded_bytes.keys()) == [target_key]
            captured_tif = uploaded_bytes[target_key][1]
            ds_up = gdal.Open(captured_tif)
            assert ds_up.RasterCount == 2 and "Unconstrained" in ds_up.GetRasterBand(1).GetDescription()
            up_dgc = ds_up.GetRasterBand(1).ReadAsArray()
            up_ref = ds_up.GetRasterBand(2).ReadAsArray()
            ds_up = None

            gc2 = convert_to_groundcover(bare2, 255)
            oracle_w = _oracle_refpixel(ref.view(np.int8), gc2, None, window=11, minvalpercent=0.10)
            exp_ref = np.where(oracle_w < 0, 0, 100 + oracle_w).astype(np.uint8)
            exp_dgc = np.where((oracle_w < 0) | (gc2 == 0), 0, 100 + gc2 - oracle_w).astype(np.uint8)
            assert np.array_equal(up_dgc, exp_dgc) and np.array_equal(up_ref, exp_ref)
        finally:
            sys.argv = orig_argv
            run_aus_batch_worker._build_s3_client = orig_build_s3
            run_aus_batch_worker._ensure_local_ref_cache = orig_cache_fn
            seasonalDRCMIbra.run_seasonal_drcm_ibra = orig_run
    print(">>> PASS: Batch submission, --save, and worker GeoTIFF output contents verified!")


def test_12_remote_vsis3_and_chip_authenticity():
    print("\n--- Test 12: Remote /vsis3/ Streaming & test_data/ Chip Authenticity Parity ---")
    with tempfile.TemporaryDirectory(prefix="drcm_test12_") as tmpdir:
        out_ibra = os.path.join(tmpdir, "s3_ibra.tif")
        out_no_ibra = os.path.join(tmpdir, "s3_no_ibra.tif")
        run_seasonal_drcm_ibra(
            S3_GC_TIF, S3_REF_TIF, output_file=out_ibra,
            window=101, test_chip=[90000, 80000, 440, 440],
        )
        run_seasonal_drcm_ibra(
            S3_GC_TIF, S3_REF_TIF, output_file=out_no_ibra,
            window=101, test_chip=[90000, 80000, 440, 440], no_ibra=True, no_overviews=True,
        )

        # Verify committed test_data/ chips match live S3 windows 100% bit-for-bit!
        ds_s3_gc = gdal.Open(S3_GC_TIF)
        ds_loc_gc = gdal.Open(LOCAL_GC_CHIP)
        s3_gc = ds_s3_gc.GetRasterBand(1).ReadAsArray(90000, 80000, 440, 440)
        loc_gc = ds_loc_gc.GetRasterBand(1).ReadAsArray()
        assert np.array_equal(s3_gc, loc_gc), "test_data GC chip does not match S3 window!"

        ds_s3_ref = gdal.Open(S3_REF_TIF)
        ds_loc_ref = gdal.Open(LOCAL_REF_CHIP)
        for b in (1, 2):
            assert np.array_equal(
                ds_s3_ref.GetRasterBand(b).ReadAsArray(89960, 79960, 520, 520),
                ds_loc_ref.GetRasterBand(b).ReadAsArray(),
            ), f"test_data ref band {b} does not match S3 window!"
        ds_s3_gc = ds_loc_gc = ds_s3_ref = ds_loc_ref = None

        # Verify remote /vsis3/ runs produce 0 mismatches against offline test_data/ runs
        loc_ibra_out = os.path.join(tmpdir, "loc_ibra.tif")
        loc_noibra_out = os.path.join(tmpdir, "loc_noibra.tif")
        run_seasonal_drcm_ibra(LOCAL_GC_CHIP, LOCAL_REF_CHIP, output_file=loc_ibra_out, window=101)
        run_seasonal_drcm_ibra(LOCAL_GC_CHIP, LOCAL_REF_CHIP, output_file=loc_noibra_out, window=101, no_ibra=True, no_overviews=True)

        for s3_p, loc_p in ((out_ibra, loc_ibra_out), (out_no_ibra, loc_noibra_out)):
            a, b = gdal.Open(s3_p), gdal.Open(loc_p)
            assert np.array_equal(a.GetRasterBand(1).ReadAsArray(), b.GetRasterBand(1).ReadAsArray())
            assert np.array_equal(a.GetRasterBand(2).ReadAsArray(), b.GetRasterBand(2).ReadAsArray())
            a = b = None
    print(">>> PASS: Remote /vsis3/ FastS3 streaming & test_data/ chip authenticity verified (0 mismatches)!")


def main():
    parser = argparse.ArgumentParser(description="Run DRCM verification test suite.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip remote /vsis3/ streaming test (Test 12); still runs all local + test_data/ real satellite tests (Tests 1-11).",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    test_1_utility_functions()
    test_2_groundcover_conversion()
    test_3_quantile_bounds_and_mean()
    test_4_oracle_invariance_and_slicing()
    test_5_boundary_isolation()
    test_6_ocean_and_valid_fraction()
    test_7_local_satellite_multi_ibra()
    test_8_local_satellite_no_ibra_and_geojson()
    test_9_local_fast_tiled_reader_and_predictor_fallback()
    test_10_multichunk_and_halo_aoi_parity()
    test_11_batch_worker_and_submission()

    if args.offline:
        print("\n--- Test 12: SKIPPED (--offline specified) ---")
    else:
        test_12_remote_vsis3_and_chip_authenticity()

    t_total = time.perf_counter() - t0
    ran_label = (
        "11 OFFLINE TESTS (incl. test_data/ real satellite chips)"
        if args.offline
        else "ALL 12 CHEAT-PROOF TESTS (incl. test_data/ + /vsis3/ parity)"
    )
    print("\n========================================================")
    print(f"{ran_label} PASSED in {t_total:.2f}s!")
    print("========================================================")


if __name__ == "__main__":
    main()
