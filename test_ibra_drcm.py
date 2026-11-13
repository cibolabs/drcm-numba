"""
test_ibra_drcm.py

Cheat-proof, rigorous test suite for the Dynamic Reference Cover Method (DRCM) pipeline.
Covers core kernels, I/O engines, halo-padded AOI subsetting, chunk overlap reuse,
AWS Batch manifest/worker orchestration, and remote /vsis3/ cloud datasets:
  - Reference: /vsis3/cibo-dat-oregon/DRCM/lztmre_aus_m198912202012_dixa2_change_slope_ibra.tif
  - Fractional Cover: /vsis3/cibo-dat-oregon/GC/lztmre_aus_m202606202608_dp8a2_vinterim.tif

Run full suite (including remote /vsis3/ tests 7 & 8):
  python test_ibra_drcm.py

Run offline/local suite only (e.g., CI or Docker builds without AWS credentials):
  python test_ibra_drcm.py --offline
"""

import argparse
import json
import os
import sys
import tempfile
import time

import numpy as np
from osgeo import gdal, ogr, osr

gdal.UseExceptions()
ogr.UseExceptions()

# Ensure repository root is on sys.path and set as working directory
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

# Remote cloud-native datasets accessed via GDAL Virtual File System (/vsis3/)
S3_REF_TIF = "/vsis3/cibo-dat-oregon/DRCM/lztmre_aus_m198912202012_dixa2_change_slope_ibra.tif"
S3_GC_TIF = "/vsis3/cibo-dat-oregon/GC/lztmre_aus_m202606202608_dp8a2_vinterim.tif"


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

    # In-place LUT path must share exact semantics and return int8 view
    dea_copy = dea_bare.copy()
    dea_inplace = convert_to_groundcover(dea_copy, nodata_val=255, inplace=True)
    assert dea_inplace.dtype == np.int8 and np.array_equal(dea_inplace, expected_dea)
    assert np.shares_memory(dea_copy, dea_inplace)

    # Non-uint8 fallback path must match LUT output bit-for-bit
    dea_u16 = convert_to_groundcover(dea_bare.astype(np.uint16), nodata_val=255)
    assert np.array_equal(dea_u16, expected_dea)

    # Legacy JRSRP (dixa2): 100..200 bare -> 101..1 gc, 0 -> 0
    jrsrp_bare = np.array([[100, 125, 150], [175, 200, 0]], dtype=np.uint8)
    expected_jrsrp = np.array([[101, 76, 51], [26, 1, 0]], dtype=np.int8)
    assert np.array_equal(convert_to_groundcover(jrsrp_bare, nodata_val=0), expected_jrsrp)
    jrsrp_copy = jrsrp_bare.copy()
    assert np.array_equal(convert_to_groundcover(jrsrp_copy, nodata_val=0, inplace=True), expected_jrsrp)
    assert np.array_equal(convert_to_groundcover(jrsrp_bare.astype(np.uint16), nodata_val=0), expected_jrsrp)

    # Degenerate: All-NoData and empty (0, 0)
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

    dryhist = np.zeros(103, dtype=np.int32)
    drysum = np.zeros(103, dtype=np.int32)
    dryhist[51] = 10
    drysum[51] = 700
    assert refpix_numba_ibra._calcmean(drysum, dryhist, 50, 50) == 70
    assert refpix_numba._calcmean(drysum, dryhist, 50, 50) == 70

    dryhist.fill(0)
    drysum.fill(0)
    assert refpix_numba_ibra._calcmean(drysum, dryhist, 10, 20) == 0
    assert refpix_numba._calcmean(drysum, dryhist, 10, 20) == 0
    assert refpix_numba_ibra._bounds(np.zeros(103, dtype=np.int32), 101, 0.90, 0.95) == (0, 0)
    assert refpix_numba._bounds_fast(np.zeros(104, dtype=np.int32), 0, 101, 0.90, 0.95) == (0, 0)
    print(">>> PASS: Quantile bounds and mean kernels verified across IBRA and Fast engines!")


def test_4_single_region_invariance_and_slicing():
    print("\n--- Test 4: Single-Region Invariance, Tiled Slicing & Kernel Width Equivalence ---")
    np.random.seed(42)
    size = 200
    cover = np.random.randint(0, 102, size=(size, size), dtype=np.int8)
    ground = np.random.randint(0, 102, size=(size, size), dtype=np.int8)
    cover[:30, :30] = 0
    ground[:30, :30] = 0
    ibra = np.full((size, size), 17, dtype=np.uint8)

    window, lq, uq, minval = 31, 0.90, 0.95, 0.15

    res_global = refpix_numba.refpixelfilternodata(cover, ground, window, lq, uq, minval)
    res_ibra = refpix_numba_ibra.refpixelfilter_ibra(cover, ground, ibra, window, lq, uq, minval)

    valid_count = int(np.sum(res_global > 0))
    nodata_count = int(np.sum(res_global == -1))
    assert valid_count > 10000 and nodata_count > 100
    assert np.array_equal(res_global, res_ibra), "res_global and res_ibra must be bit-for-bit identical"

    # Verify sub-window slicing (used by halo-padded AOIs and chunking) and alternate tile sizes
    r0, r1, c0, c1 = 23, 173, 19, 181
    sliced_small_tiles = refpix_numba.refpixelfilternodata(
        cover, ground, window, lq, uq, minval,
        r_start=r0, r_stop=r1, c_start=c0, c_stop=c1,
        strip_h=32, col_tile_w=48,
    )
    assert np.array_equal(sliced_small_tiles, res_global[r0:r1, c0:c1]), "Sub-window or tile-size mismatch!"

    # Verify int32 column-histogram kernel matches uint16 kernel bit-for-bit
    res_i32 = refpix_numba._KERNEL_I32(
        cover, ground, window, lq, uq, minval, 0, size, 0, size, 64, 64
    )
    assert np.array_equal(res_i32, res_global), "_KERNEL_I32 vs _KERNEL_U16 mismatch!"
    print(f">>> PASS: {size * size:,} pixels, sub-window slicing, tile sizes, and U16/I32 kernels match 100%!")


def test_5_boundary_isolation():
    print("\n--- Test 5: Cheat-Proof Boundary Isolation Test ---")
    size = 100
    cover = np.zeros((size, size), dtype=np.int8)
    ground = np.zeros((size, size), dtype=np.int8)
    ibra = np.zeros((size, size), dtype=np.uint8)

    cover[:, :50] = 80
    ground[:, :50] = 70
    ibra[:, :50] = 10
    cover[::6, :50] = 95
    ground[::6, :50] = 90

    cover[:, 50:] = 20
    ground[:, 50:] = 15
    ibra[:, 50:] = 20
    cover[::6, 50:] = 35
    ground[::6, 50:] = 30

    window, lq, uq, minval = 21, 0.90, 0.95, 0.05
    res_ibra = refpix_numba_ibra.refpixelfilter_ibra(cover, ground, ibra, window, lq, uq, minval)
    res_global = refpix_numba.refpixelfilternodata(cover, ground, window, lq, uq, minval)

    assert ground[55, 49] == 70 and res_ibra[55, 49] == 90
    assert ground[55, 50] == 15 and res_ibra[55, 50] == 30
    assert np.all(res_ibra[20:80, 48:50] == 90)
    assert np.all(res_ibra[20:80, 50:52] == 30)
    assert np.all(res_global[20:80, 50:52] > 60)
    print(">>> PASS: Zero cross-boundary contamination strictly proven!")


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

    # Verify refpix_numba coarse 256x256 empty-block skipping preserves exact edge/interior semantics
    sparse_h, sparse_w = 600, 600
    sp_cov = np.zeros((sparse_h, sparse_w), dtype=np.int8)
    sp_gnd = np.zeros((sparse_h, sparse_w), dtype=np.int8)
    sp_cov[300:360, 300:360] = 72
    sp_gnd[300:360, 300:360] = 64

    blocks = refpix_numba._build_valid_blocks(sp_cov, sp_gnd, 256)
    assert blocks[0, 0] == 0 and blocks[1, 1] == 1

    # Compare empty-block-skipped run against unskipped run (forced via distant corner pixel)
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

    # Verify parity with IBRA engine where minvalpercent (0.001 -> >=1 valid px) is invariant to 30*31 vs 31*31 denom
    sp_fast_all = refpix_numba.refpixelfilternodata(sp_cov, sp_gnd, window=31, minvalpercent=0.001)
    sp_ibra_all = refpix_numba_ibra.refpixelfilter_ibra(
        sp_cov, sp_gnd, np.ones((sparse_h, sparse_w), dtype=np.uint8), window=31, minvalpercent=0.001
    )
    assert np.array_equal(sp_fast_all[:, :-1], sp_ibra_all[:, :-1])
    print(">>> PASS: Ocean masking, empty-block skip, and validity thresholds verified!")


def test_7_satellite_multi_ibra_end_to_end():
    print("\n--- Test 7: Real Satellite Multi-IBRA End-to-End Test (via /vsis3/) ---")
    with tempfile.TemporaryDirectory(prefix="drcm_test7_") as tmpdir:
        out_drcm_tif = os.path.join(tmpdir, "test_full_suite_drcm.tif")
        run_seasonal_drcm_ibra(
            input_groundcover=S3_GC_TIF,
            percentile_file=S3_REF_TIF,
            output_file=out_drcm_tif,
            window=101,
            test_chip=[90000, 80000, 500, 500],
        )

        ds_res = gdal.Open(out_drcm_tif)
        assert ds_res.RasterCount == 2
        b1_res = ds_res.GetRasterBand(1)
        b2_res = ds_res.GetRasterBand(2)
        assert "IBRA-Constrained" in b1_res.GetDescription()
        assert "IBRA-Constrained" in b2_res.GetDescription()
        assert b1_res.GetOverviewCount() == 4
        assert b2_res.GetOverviewCount() == 4

        dgc = b1_res.ReadAsArray()
        ref_scaled = b2_res.ReadAsArray()
        assert np.sum(dgc > 0) > 200000 and np.sum(ref_scaled > 0) > 200000

        ds_ref = gdal.Open(S3_REF_TIF)
        perc = ds_ref.GetRasterBand(1).ReadAsArray(90000, 80000, 500, 500).astype(np.int8)
        ibra = ds_ref.GetRasterBand(2).ReadAsArray(90000, 80000, 500, 500)
        ds_gc = gdal.Open(S3_GC_TIF)
        bare = ds_gc.GetRasterBand(1).ReadAsArray(90000, 80000, 500, 500)
        gc = convert_to_groundcover(bare)

        direct_ref = refpix_numba_ibra.refpixelfilter_ibra(perc, gc, ibra, window=101)
        direct_ref_scaled = np.where(direct_ref < 0, 0, 100 + direct_ref).astype(np.uint8)
        direct_dgc = np.where((direct_ref < 0) | (gc == 0), 0, 100 + gc - direct_ref).astype(np.uint8)

        assert np.array_equal(ref_scaled, direct_ref_scaled)
        assert np.array_equal(dgc, direct_dgc)

        unique_regions = np.unique(ibra)
        assert 17 in unique_regions and 58 in unique_regions
        mean_17 = np.mean(ref_scaled[(ibra == 17) & (ref_scaled > 0)])
        mean_58 = np.mean(ref_scaled[(ibra == 58) & (ref_scaled > 0)])
        assert abs(mean_17 - mean_58) > 0.1
        ds_res = ds_ref = ds_gc = None
    print(">>> PASS: Real satellite multi-IBRA DRCM run (via /vsis3/) verified!")


def test_8_satellite_no_ibra_mode():
    print("\n--- Test 8: Single Global Histogram Mode (--no-ibra via /vsis3/) ---")
    with tempfile.TemporaryDirectory(prefix="drcm_test8_") as tmpdir:
        out_no_ibra = os.path.join(tmpdir, "test_no_ibra_eval.tif")
        run_seasonal_drcm_ibra(
            input_groundcover=S3_GC_TIF,
            percentile_file=S3_REF_TIF,
            output_file=out_no_ibra,
            window=101,
            test_chip=[90000, 80000, 500, 500],
            no_ibra=True,
            no_overviews=True,
        )

        ds1 = gdal.Open(out_no_ibra)
        assert "Unconstrained" in ds1.GetRasterBand(1).GetDescription()
        assert "Unconstrained" in ds1.GetRasterBand(2).GetDescription()
        dgc1 = ds1.GetRasterBand(1).ReadAsArray()
        ref1 = ds1.GetRasterBand(2).ReadAsArray()

        in_ds = gdal.Open(S3_GC_TIF)
        ref_ds = gdal.Open(S3_REF_TIF)
        bare = in_ds.GetRasterBand(1).ReadAsArray(90000, 80000, 500, 500)
        gc = convert_to_groundcover(bare)
        perc = ref_ds.GetRasterBand(1).ReadAsArray(90000, 80000, 500, 500).astype(np.int8)

        direct_ref = refpix_numba.refpixelfilternodata(perc, gc, window=101)
        direct_ref_scaled = np.where(direct_ref < 0, 0, 100 + direct_ref).astype(np.uint8)
        direct_dgc = np.where((direct_ref < 0) | (gc == 0), 0, 100 + gc - direct_ref).astype(np.uint8)

        assert np.array_equal(dgc1, direct_dgc)
        assert np.array_equal(ref1, direct_ref_scaled)
        ds1 = in_ds = ref_ds = None
    print(">>> PASS: --no-ibra unconstrained mode (via /vsis3/) matches baseline 100%!")


def _write_test_geotiff(path, bands_u8, gt, proj_wkt, nodata=None, predictor=1, block_size=128):
    """Helper to write a multi-band tiled DEFLATE uint8 GeoTIFF."""
    h, w = bands_u8[0].shape
    drv = gdal.GetDriverByName("GTiff")
    opts = [
        "COMPRESS=DEFLATE",
        f"PREDICTOR={predictor}",
        "TILED=YES",
        f"BLOCKXSIZE={block_size}",
        f"BLOCKYSIZE={block_size}",
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


def test_9_local_fast_tiled_reader_and_predictor_fallback():
    print("\n--- Test 9: FastLocal (os.pread + zlib) Tiled Reader & PREDICTOR=2 Fallback ---")
    with tempfile.TemporaryDirectory(prefix="drcm_test9_") as tmpdir:
        local_in = os.path.join(tmpdir, "in_pred1.tif")
        local_pred2 = os.path.join(tmpdir, "in_pred2.tif")
        local_ref = os.path.join(tmpdir, "ref.tif")
        local_out = os.path.join(tmpdir, "out.tif")

        rng = np.random.default_rng(12345)
        h, w = 300, 300
        bare = rng.integers(0, 101, size=(h, w), dtype=np.uint8)
        bare[20:40, 20:40] = 255
        ref_arr = rng.integers(1, 102, size=(h, w), dtype=np.uint8)
        ref_arr[128:256, 128:256] = 0
        gt = (0.0, 30.0, 0.0, 0.0, 0.0, -30.0)

        _write_test_geotiff(local_in, [bare], gt, "", nodata=255, predictor=1, block_size=128)
        _write_test_geotiff(local_pred2, [bare], gt, "", nodata=255, predictor=2, block_size=128)
        _write_test_geotiff(local_ref, [ref_arr], gt, "", nodata=0, predictor=1, block_size=128)

        # 1. Verify FastLocal activation and exact pixel reads
        rdr = _SequentialBandReader(local_in, local_ref, io_workers=4)
        try:
            assert rdr.in_fast is not None and rdr.in_fast.local_path == local_in
            assert rdr.ref_fast is not None and rdr.ref_fast.local_path == local_ref
            assert np.array_equal(rdr.read_in_window((0, 0, w, h)), bare)
            got_ref, _ = rdr.read_ref_windows((0, 0, w, h), need_ibra=False)
            assert np.array_equal(got_ref, ref_arr.view(np.int8))
        finally:
            rdr.close()

        # 2. Verify PREDICTOR=2 self-verification fallback cleanly drops back to GDAL
        rdr2 = _SequentialBandReader(local_pred2, local_ref, io_workers=4)
        try:
            assert rdr2.in_fast is None, "PREDICTOR=2 input must safely fall back to GDAL reader"
            assert rdr2.ref_fast is not None
            assert np.array_equal(rdr2.read_in_window((0, 0, w, h)), bare)
        finally:
            rdr2.close()

        # 3. Verify end-to-end local pipeline matches direct Numba kernel
        run_seasonal_drcm_ibra(
            input_groundcover=local_in,
            percentile_file=local_ref,
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
    print(">>> PASS: FastLocal os.pread + PREDICTOR=2 GDAL fallback verified!")


def test_10_multichunk_and_halo_aoi_parity():
    print("\n--- Test 10: Multi-Chunk Overlap Reuse & Halo-Padded --bbox / --geojson Parity ---")
    with tempfile.TemporaryDirectory(prefix="drcm_test10_") as tmpdir:
        srs_3577 = osr.SpatialReference()
        srs_3577.ImportFromEPSG(3577)
        srs_3577.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        wkt_3577 = srs_3577.ExportToWkt()

        srs_4326 = osr.SpatialReference()
        srs_4326.ImportFromEPSG(4326)
        srs_4326.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

        h, w = 420, 440
        gt = (1000000.0, 30.0, 0.0, -3200000.0, 0.0, -30.0)
        rng = np.random.default_rng(777)
        bare = rng.integers(5, 95, size=(h, w), dtype=np.uint8)
        bare[:40, :40] = 255
        perc = rng.integers(10, 101, size=(h, w), dtype=np.uint8)
        ibra = np.where(np.arange(w)[None, :] < 220, 17, 58).astype(np.uint8)

        in_tif = os.path.join(tmpdir, "aus_synth_dp8a2.tif")
        ref_tif = os.path.join(tmpdir, "aus_synth_ref_ibra.tif")
        _write_test_geotiff(in_tif, [bare], gt, wkt_3577, nodata=255, predictor=1, block_size=128)
        _write_test_geotiff(ref_tif, [perc, ibra], gt, wkt_3577, nodata=0, predictor=1, block_size=128)

        # 1. Verify multi-chunk overlap reuse (chunksize=96 -> 5 chunks) vs single-chunk (chunksize=1024)
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

        # 2. Verify halo-padded --bbox and EPSG:4326 --geojson match full-grid slice 100% bit-for-bit
        # Choose AOI spanning columns 150..310 and rows 120..300 (crosses IBRA boundary at col 220!)
        minx = gt[0] + 150 * gt[1]
        maxx = gt[0] + 310 * gt[1]
        maxy = gt[3] + 120 * gt[5]
        miny = gt[3] + 300 * gt[5]

        # Build EPSG:4326 GeoJSON polygon corresponding to exact projected envelope corners
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
                x0 = int(round((sgt[0] - gt[0]) / gt[1]))
                y0 = int(round((sgt[3] - gt[3]) / gt[5]))
                sw, sh = sub_ds.RasterXSize, sub_ds.RasterYSize
                assert np.array_equal(sub_ds.GetRasterBand(1).ReadAsArray(), full_b1[y0:y0 + sh, x0:x0 + sw]), (
                    f"Band 1 halo mismatch for {p}"
                )
                assert np.array_equal(sub_ds.GetRasterBand(2).ReadAsArray(), full_b2[y0:y0 + sh, x0:x0 + sw]), (
                    f"Band 2 halo mismatch for {p}"
                )
                sub_ds = None
    print(">>> PASS: Multi-chunk overlap reuse & halo-padded --bbox/--geojson parity verified (0 mismatches)!")


def test_11_batch_worker_and_submission():
    print("\n--- Test 11: Batch Submission Manifest/Command & Worker Shard Execution ---")
    # 1. Test season filtering, round-robin sharding, and command construction
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
    assert manifest["num_shards"] == 2
    assert manifest["output_prefix"] == "DRCM/nsw_field/"
    assert manifest["geojson"] == "field.geojson"
    assert len(manifest["shards"][0]) == 2 and len(manifest["shards"][1]) == 1
    assert sum(shard_bytes) == 39_000_000_000

    # Single-shard / retry commands must pass explicit --shard-index
    cmd_single = submit_batch_jobs.build_worker_command(
        "s3://cibo-dat-oregon/DRCM/_batch_scripts/v4", explicit_shard_index=6
    )
    assert "--shard-index 6" in cmd_single[2]
    cmd_array = submit_batch_jobs.build_worker_command("s3://cibo-dat-oregon/DRCM/_batch_scripts/v4")
    assert "--shard-index" not in cmd_array[2]

    env_map = {e["name"]: e["value"] for e in submit_batch_jobs.build_batch_environment()}
    assert env_map["NUMBA_CPU_NAME"] == "generic"
    assert env_map["NUMBA_CPU_FEATURES"] == "+neon"
    assert "0" in env_map["GDAL_HTTP_RETRY_CODES"].split(",")

    # 2. Test run_aus_batch_worker manifest parsing, skip-existing, and shard execution locally
    with tempfile.TemporaryDirectory(prefix="drcm_test11_") as tmpdir:
        gt = (0.0, 30.0, 0.0, 0.0, 0.0, -30.0)
        bare1 = np.full((128, 128), 30, dtype=np.uint8)
        bare2 = np.full((128, 128), 60, dtype=np.uint8)
        ref = np.full((128, 128), 80, dtype=np.uint8)

        in1 = os.path.join(tmpdir, "lztmre_aus_m202503202505_dp8a2.tif")
        in2 = os.path.join(tmpdir, "lztmre_aus_m202506202508_dp8a2.tif")
        ref_tif = os.path.join(tmpdir, "ref.tif")
        _write_test_geotiff(in1, [bare1], gt, "", nodata=255, predictor=1, block_size=128)
        _write_test_geotiff(in2, [bare2], gt, "", nodata=255, predictor=1, block_size=128)
        _write_test_geotiff(ref_tif, [ref], gt, "", nodata=0, predictor=1, block_size=128)

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

        class _FakeS3:
            def __init__(self):
                self.uploaded = {}

            def head_object(self, Bucket, Key):
                if Key.endswith("202503202505_doda2.tif"):
                    return {"ContentLength": 500_000_000}  # Simulate already-completed Season 1
                raise FileNotFoundError(Key)

            def upload_file(self, Filename, Bucket, Key, Config=None):
                self.uploaded[Key] = os.path.getsize(Filename)

        fake_s3 = _FakeS3()
        orig_build_s3 = run_aus_batch_worker._build_s3_client
        orig_cache_fn = run_aus_batch_worker._ensure_local_ref_cache
        orig_argv = sys.argv
        try:
            run_aus_batch_worker._build_s3_client = lambda: fake_s3
            run_aus_batch_worker._ensure_local_ref_cache = lambda s3, b, k, d: (ref_tif, False)

            # Local inputs in manifest: override path resolution just for local paths
            orig_run = seasonalDRCMIbra.run_seasonal_drcm_ibra

            def _wrapped_run(**kw):
                ig = kw["input_groundcover"]
                if ig.startswith("s3://test-bucket/"):
                    kw["input_groundcover"] = ig[len("s3://test-bucket/"):]
                return orig_run(**kw)

            seasonalDRCMIbra.run_seasonal_drcm_ibra = _wrapped_run
            fake_s3_upload_target = "DRCM/aus/lztmre_aus_m202506202508_doda2.tif"

            # Allow upload verification head_object to pass once uploaded
            orig_head = fake_s3.head_object

            def _head_after_upload(Bucket, Key):
                if Key in fake_s3.uploaded:
                    return {"ContentLength": fake_s3.uploaded[Key]}
                return orig_head(Bucket, Key)

            fake_s3.head_object = _head_after_upload
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

            # Season 1 was skipped, Season 2 was computed & uploaded
            assert list(fake_s3.uploaded.keys()) == [fake_s3_upload_target]
            assert fake_s3.uploaded[fake_s3_upload_target] > 100
        finally:
            sys.argv = orig_argv
            run_aus_batch_worker._build_s3_client = orig_build_s3
            run_aus_batch_worker._ensure_local_ref_cache = orig_cache_fn
            seasonalDRCMIbra.run_seasonal_drcm_ibra = orig_run
    print(">>> PASS: Batch submission manifest/command and worker skip-existing execution verified!")


def main():
    parser = argparse.ArgumentParser(description="Run DRCM verification test suite.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Skip remote /vsis3/ tests (Tests 7 & 8) for local/CI verification without AWS access.",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    test_1_utility_functions()
    test_2_groundcover_conversion()
    test_3_quantile_bounds_and_mean()
    test_4_single_region_invariance_and_slicing()
    test_5_boundary_isolation()
    test_6_ocean_and_valid_fraction()

    if args.offline:
        print("\n--- Tests 7 & 8: SKIPPED (--offline specified) ---")
    else:
        test_7_satellite_multi_ibra_end_to_end()
        test_8_satellite_no_ibra_mode()

    test_9_local_fast_tiled_reader_and_predictor_fallback()
    test_10_multichunk_and_halo_aoi_parity()
    test_11_batch_worker_and_submission()

    t_total = time.perf_counter() - t0
    ran_label = "9 OFFLINE TESTS" if args.offline else "ALL 11 CHEAT-PROOF TESTS (100% /vsis3/ + FastLocal + AOI + Batch)"
    print("\n========================================================")
    print(f"{ran_label} PASSED in {t_total:.2f}s!")
    print("========================================================")


if __name__ == "__main__":
    main()
