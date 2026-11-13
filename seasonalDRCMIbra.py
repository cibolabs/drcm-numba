#!/usr/bin/env python
"""
seasonalDRCMIbra.py

IBRA-Constrained Seasonal DRCM (Deficit / Reference Cover Method) Processing.

Compares seasonal fractional groundcover against long-term reference statistics
strictly constrained to matching IBRA biogeographic regions (1..89), or using a
single global histogram (--no-ibra).
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import http.client
import math
import os
import random
import resource
import tempfile
import time
import zlib

import numpy as np
from numba import njit, prange
from osgeo import gdal, ogr, osr

gdal.UseExceptions()
ogr.UseExceptions()

from refpix_numba_ibra import refpixelfilter_ibra
from refpix_numba import refpixelfilternodata

# Default S3 destination when --save is passed without an s3:// -o URI
SAVE_BUCKET = "cibo-dat-oregon"
SAVE_DIR = "DRCM"


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
        help="Output GeoTIFF path or s3:// URI (or s3://bucket/prefix/ ending with '/'). If not specified, derived automatically from the input name."
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
        default=4096,
        help="Number of lines to process per chunk (default: 4096)"
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
        default=0.15,
        help="Minimum fraction of valid pixels within comparison window in matching IBRA (default: 0.15)"
    )
    parser.add_argument(
        "--test-chip",
        nargs=4,
        type=int,
        metavar=("XOFF", "YOFF", "XSIZE", "YSIZE"),
        help="Optional test chip window [xoff, yoff, xsize, ysize] for unpadded testing/micro-subsetting"
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("MINX", "MINY", "MAXX", "MAXY"),
        help="Projected bounding box [minx miny maxx maxy] in input raster CRS (halo-padded reads so edge pixels match continental runs)"
    )
    parser.add_argument(
        "--geojson",
        default=None,
        help="Path or s3:// URI to GeoJSON AOI whose envelope subsets output cleanly with halo padding"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of Numba parallel worker threads to use (default: container cgroup quota)"
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
        help="Upload local output to s3://cibo-dat-oregon/DRCM/ and delete the local file on completion"
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


@njit(parallel=True)
def _max_u8_2d(arr):
    h, w = arr.shape
    row_max = np.zeros(h, dtype=np.uint8)
    for r in prange(h):
        m = np.uint8(0)
        for c in range(w):
            v = arr[r, c]
            if v > m:
                m = v
        row_max[r] = m
    m = np.uint8(0)
    for r in range(h):
        if row_max[r] > m:
            m = row_max[r]
    return int(m)


@njit(parallel=True)
def _apply_lut_u8_2d(src, lut, dst):
    h, w = src.shape
    for r in prange(h):
        for c in range(w):
            dst[r, c] = lut[src[r, c]]


@njit(parallel=True)
def _apply_lut_u8_inplace(arr_u8, lut_u8):
    h, w = arr_u8.shape
    for r in prange(h):
        for c in range(w):
            arr_u8[r, c] = lut_u8[arr_u8[r, c]]


@njit(parallel=True)
def _encode_outputs_2d(chunk_gc, chunk_ref):
    """Compute uint8 DGC and scaled reference cover without int64 intermediate arrays."""
    h, w = chunk_ref.shape
    dgc_data = np.empty((h, w), dtype=np.uint8)
    ref_data_scaled = np.empty((h, w), dtype=np.uint8)
    for r in prange(h):
        for c in range(w):
            rv = chunk_ref[r, c]
            if rv < 0:
                ref_data_scaled[r, c] = 0
                dgc_data[r, c] = 0
            else:
                ref_data_scaled[r, c] = np.uint8(100 + rv)
                gv = chunk_gc[r, c]
                if gv == 0:
                    dgc_data[r, c] = 0
                else:
                    dgc_data[r, c] = np.uint8(100 + gv - rv)
    return dgc_data, ref_data_scaled


def convert_to_groundcover(bare_data, nodata_val=255, inplace=False):
    """
    Convert raw bare ground array to persistent groundcover + 1 (values 1..101, 0 is nodata).

    Auto-detects and supports both remote sensing standards:
    - Modern DEA (dp8a2): Bare stored as 0..100% with 255 NoData.
      Groundcover = 100 - bare% -> mapped to 101 - bare (1..101 range).
    - Legacy JRSRP/RSC (dixa2): Bare stored with +100 offset (100 = 0% bare, 200 = 100% bare).
      Groundcover mapped to 201 - bare (1..101 range).
    """
    if bare_data.size == 0:
        return np.empty(bare_data.shape, dtype=np.int8)

    if bare_data.dtype == np.uint8 and bare_data.ndim == 2:
        max_val = _max_u8_2d(bare_data)
        lut = np.zeros(256, dtype=np.int8)
        if max_val > 100 and max_val != 255:
            lut[100:201] = np.arange(101, 0, -1, dtype=np.int8)
        else:
            lut[0:101] = np.arange(101, 0, -1, dtype=np.int8)
        if nodata_val is not None:
            try:
                nd = int(nodata_val)
                if 0 <= nd <= 255 and nd == nodata_val:
                    lut[nd] = 0
            except (TypeError, ValueError):
                pass
        if inplace and bare_data.flags.writeable:
            # Output range 0..101 is bit-identical across uint8 and int8 views.
            _apply_lut_u8_inplace(bare_data, lut.view(np.uint8))
            return bare_data.view(np.int8)
        out = np.empty(bare_data.shape, dtype=np.int8)
        _apply_lut_u8_2d(bare_data, lut, out)
        return out

    max_val = np.max(bare_data)
    if max_val > 100 and max_val != 255:
        gc = np.where((bare_data >= 100) & (bare_data <= 200) & (bare_data != nodata_val),
                      201 - bare_data, 0)
    else:
        gc = np.where((bare_data <= 100) & (bare_data != nodata_val),
                      101 - bare_data, 0)
    return np.clip(gc, 0, 101).astype(np.int8)


def _to_gdal_path(path):
    """Convert s3:// URI to GDAL /vsis3/ virtual filesystem path if needed."""
    if path and isinstance(path, str) and path.startswith("s3://"):
        return f"/vsis3/{path[5:]}"
    return path


def _parse_s3_uri(uri):
    without_scheme = uri[5:]
    bucket, _, key = without_scheme.partition("/")
    if not bucket or not key or key.endswith("/"):
        raise ValueError(f"Invalid S3 object URI: {uri}")
    return bucket, key


def _parse_vsis3_uri(path):
    without = path[len("/vsis3/"):]
    bucket, _, key = without.partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid /vsis3/ path: {path}")
    return bucket, key


def _get_effective_cpus():
    """Detect effective CPU count honoring container cgroup v1/v2 quotas."""
    host_cpus = os.cpu_count() or 1
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            parts = f.read().strip().split()
        if len(parts) == 2 and parts[0] != "max":
            quota = int(parts[0])
            period = int(parts[1])
            if quota > 0 and period > 0:
                return min(host_cpus, max(1, int(round(quota / period))))
    except OSError:
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fq, open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fp:
            quota = int(fq.read().strip())
            period = int(fp.read().strip())
        if quota > 0 and period > 0:
            return min(host_cpus, max(1, int(round(quota / period))))
    except OSError:
        pass
    return host_cpus


def _configure_vsis3_defaults():
    """Apply safe high-throughput GDAL defaults for /vsis3/ GeoTIFF streaming."""
    if os.environ.get("CPL_AWS_WEB_IDENTITY_ENABLE") is None:
        os.environ["CPL_AWS_WEB_IDENTITY_ENABLE"] = "NO"
    defaults = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_USE_HEAD": "NO",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
        "GDAL_HTTP_MULTIPLEX": "YES",
        "GDAL_HTTP_MAX_RETRY": "20",
        "GDAL_HTTP_RETRY_DELAY": "1",
        "GDAL_HTTP_RETRY_CODES": "400,429,500,502,503,504,0",
        "CPL_ZIP_ENCODING": "UTF-8",
        "CPL_AWS_WEB_IDENTITY_ENABLE": "NO",
    }
    for k, v in defaults.items():
        if gdal.GetConfigOption(k) is None:
            gdal.SetConfigOption(k, v)


def _is_transient_io_error(exc):
    msg = str(exc).lower()
    if isinstance(exc, (http.client.HTTPException, OSError, TimeoutError)):
        return True
    tokens = (
        "response_code=0",
        "response_code=429",
        "response_code=500",
        "response_code=502",
        "response_code=503",
        "response_code=504",
        "timed out",
        "timeout",
        "connection",
        "reset by peer",
        "slowdown",
        "incompleteread",
        "curl",
        "temporarily unavailable",
    )
    return any(t in msg for t in tokens)


def _read_gdal_band_with_retry(band, win, retries=6):
    """Read a GDAL window with exponential backoff and curl cache reset on transient errors."""
    last_exc = None
    for attempt in range(retries):
        try:
            return band.ReadAsArray(*win)
        except Exception as exc:
            last_exc = exc
            if not _is_transient_io_error(exc) or attempt + 1 >= retries:
                raise
            try:
                gdal.VSICurlClearCache()
            except Exception:
                pass
            delay = min(20.0, (1.5 ** attempt) + random.uniform(0.2, 1.0))
            print(f"Transient GDAL read error ({exc}); retrying in {delay:.1f}s ({attempt + 1}/{retries})...", flush=True)
            time.sleep(delay)
    raise last_exc


class _FastS3TiledBandReader:
    """
    Parallel coalesced range reader for uncompressed-predictor DEFLATE tiled Byte GeoTIFFs.

    Supports both S3 (/vsis3/) coalesced byte-range GETs and thread-safe local POSIX pread,
    with 1-tile GDAL self-verification and automatic fallback on any mismatch.
    """

    def __init__(self, s3_client, bucket, key, band_idx, ds, sparse_fill=0, max_workers=16, local_path=None):
        self.s3 = s3_client
        self.bucket = bucket
        self.key = key
        self.local_path = local_path
        self._local_fd = os.open(local_path, os.O_RDONLY) if local_path else None
        self.label = local_path if local_path else f"s3://{bucket}/{key}"
        try:
            self.width = ds.RasterXSize
            self.height = ds.RasterYSize
            band = ds.GetRasterBand(band_idx)
            self.bw, self.bh = band.GetBlockSize()
            self.ntx = (self.width + self.bw - 1) // self.bw
            self.nty = (self.height + self.bh - 1) // self.bh
            self.sparse_fill = int(sparse_fill)
            self.max_workers = max(1, int(max_workers))

            offsets = np.zeros((self.nty, self.ntx), dtype=np.int64)
            sizes = np.zeros((self.nty, self.ntx), dtype=np.int64)
            sample_tile = None
            fallback_tile = None

            for ty in range(self.nty):
                for tx in range(self.ntx):
                    off_s = band.GetMetadataItem(f"BLOCK_OFFSET_{tx}_{ty}", "TIFF")
                    sz_s = band.GetMetadataItem(f"BLOCK_SIZE_{tx}_{ty}", "TIFF")
                    if off_s and sz_s:
                        off = int(off_s)
                        sz = int(sz_s)
                        offsets[ty, tx] = off
                        sizes[ty, tx] = sz
                        if sz > 512 and sample_tile is None:
                            sample_tile = (tx, ty, off, sz)
                        elif sz > 0 and fallback_tile is None:
                            fallback_tile = (tx, ty, off, sz)

            self.offsets = offsets
            self.sizes = sizes

            # Self-verify against GDAL on a non-empty tile to guarantee raw zlib == GDAL ReadAsArray.
            chk = sample_tile or fallback_tile
            if chk is None:
                raise RuntimeError(f"No non-empty TIFF block offsets found for {self.label} band {band_idx}")
            tx, ty, off, sz = chk
            rem_w = min(self.bw, self.width - tx * self.bw)
            rem_h = min(self.bh, self.height - ty * self.bh)
            raw = self._fetch_range_with_retry(off, off + sz - 1)
            dec = np.frombuffer(zlib.decompress(raw), dtype=np.uint8).reshape(self.bh, self.bw)[:rem_h, :rem_w]
            ref = _read_gdal_band_with_retry(band, (tx * self.bw, ty * self.bh, rem_w, rem_h))
            if not np.array_equal(dec, ref):
                raise RuntimeError(f"Tile self-verification mismatch at ({tx}, {ty}) for {self.label} band {band_idx}")
        except Exception:
            self.close()
            raise

    def close(self):
        if getattr(self, "_local_fd", None) is not None:
            try:
                os.close(self._local_fd)
            except OSError:
                pass
            self._local_fd = None

    def _fetch_range_with_retry(self, start_byte, end_byte, retries=8):
        expected = end_byte - start_byte + 1
        if self._local_fd is not None:
            chunks = []
            pos = start_byte
            rem = expected
            while rem > 0:
                chunk = os.pread(self._local_fd, rem, pos)
                if not chunk:
                    raise OSError(f"Unexpected EOF reading {self.label} at offset {pos} ({rem} remaining)")
                chunks.append(chunk)
                pos += len(chunk)
                rem -= len(chunk)
            return chunks[0] if len(chunks) == 1 else b"".join(chunks)

        range_hdr = f"bytes={start_byte}-{end_byte}"
        last_exc = None
        for attempt in range(retries):
            try:
                resp = self.s3.get_object(Bucket=self.bucket, Key=self.key, Range=range_hdr)
                data = resp["Body"].read()
                if len(data) != expected:
                    raise OSError(f"Short S3 range read: got {len(data)} of {expected} bytes")
                return data
            except Exception as exc:
                last_exc = exc
                if not _is_transient_io_error(exc) or attempt + 1 >= retries:
                    raise
                delay = min(20.0, (1.5 ** attempt) + random.uniform(0.1, 0.8))
                time.sleep(delay)
        raise last_exc

    def read_window(self, win, executor):
        xoff, yoff, xsize, ysize = map(int, win)
        out = np.empty((ysize, xsize), dtype=np.uint8)
        if xsize <= 0 or ysize <= 0:
            return out

        bw, bh = self.bw, self.bh
        tx0 = xoff // bw
        tx1 = (xoff + xsize - 1) // bw
        ty0 = yoff // bh
        ty1 = (yoff + ysize - 1) // bh

        jobs = []
        offsets = self.offsets
        sizes = self.sizes
        for ty in range(ty0, ty1 + 1):
            gy0 = max(yoff, ty * bh)
            gy1 = min(yoff + ysize, (ty + 1) * bh)
            sy0 = gy0 - ty * bh
            sy1 = gy1 - ty * bh
            dy0 = gy0 - yoff
            dy1 = gy1 - yoff
            for tx in range(tx0, tx1 + 1):
                gx0 = max(xoff, tx * bw)
                gx1 = min(xoff + xsize, (tx + 1) * bw)
                dx0 = gx0 - xoff
                dx1 = gx1 - xoff
                sz = int(sizes[ty, tx])
                if sz <= 0:
                    out[dy0:dy1, dx0:dx1] = self.sparse_fill
                    continue
                sx0 = gx0 - tx * bw
                sx1 = gx1 - tx * bw
                off = int(offsets[ty, tx])
                jobs.append((off, sz, sy0, sy1, sx0, sx1, dy0, dy1, dx0, dx1))

        if not jobs:
            return out

        jobs.sort(key=lambda x: x[0])

        # Coalesce nearby byte ranges into <=8 MiB S3 requests to slash request overhead
        # while maintaining high thread-level decode parallelism.
        max_gap = 256 * 1024
        max_batch_bytes = 8 * 1024 * 1024
        batches = []
        b_start = jobs[0][0]
        b_end = jobs[0][0] + jobs[0][1] - 1
        cur_batch = [jobs[0]]

        for item in jobs[1:]:
            off, sz = item[0], item[1]
            cand_end = max(b_end, off + sz - 1)
            if off <= (b_end + 1 + max_gap) and (cand_end - b_start + 1) <= max_batch_bytes:
                b_end = cand_end
                cur_batch.append(item)
            else:
                batches.append((b_start, b_end, cur_batch))
                b_start = off
                b_end = off + sz - 1
                cur_batch = [item]
        batches.append((b_start, b_end, cur_batch))

        def _process_batch(start_b, end_b, items):
            payload = self._fetch_range_with_retry(start_b, end_b)
            mv = memoryview(payload)
            for off, sz, sy0, sy1, sx0, sx1, dy0, dy1, dx0, dx1 in items:
                rel = off - start_b
                tile = np.frombuffer(zlib.decompress(mv[rel:rel + sz]), dtype=np.uint8).reshape(bh, bw)
                out[dy0:dy1, dx0:dx1] = tile[sy0:sy1, sx0:sx1]

        futures = [executor.submit(_process_batch, sb, eb, items) for sb, eb, items in batches]
        for fut in as_completed(futures):
            fut.result()

        return out


class _SequentialBandReader:
    """
    Hybrid S3/GDAL raster reader preferring parallel coalesced DEFLATE tile reading
    on eligible /vsis3/ GeoTIFFs with automatic verified fallback to GDAL + retry.
    """

    def __init__(self, in_path, ref_path, ibra_path=None, ibra_band_idx=None, io_workers=16):
        self.io_workers = max(1, int(io_workers))
        self.io_pool = None
        self.in_fast = None
        self.ref_fast = None
        self.ibra_fast = None
        self.in_ds = None
        self.in_band = None
        self.ref_ds = None
        self.ref_band = None
        self.ibra_ds = None
        self.ibra_band = None

        # Open GDAL datasets concurrently when reading headers across S3.
        def _open_ro(p):
            return gdal.Open(p, gdal.GA_ReadOnly)

        paths = [in_path, ref_path]
        if ibra_band_idx is not None and ibra_path:
            paths.append(ibra_path)
        with ThreadPoolExecutor(max_workers=len(paths)) as ex:
            opened = list(ex.map(_open_ro, paths))

        self.in_ds = opened[0]
        self.in_band = self.in_ds.GetRasterBand(1)
        self.ref_ds = opened[1]
        self.ref_band = self.ref_ds.GetRasterBand(1)
        if ibra_band_idx is not None:
            if ibra_path:
                self.ibra_ds = opened[2]
                self.ibra_band = self.ibra_ds.GetRasterBand(ibra_band_idx)
            else:
                self.ibra_band = self.ref_ds.GetRasterBand(ibra_band_idx)

        s3_client = self._maybe_build_s3_client(in_path, ref_path, ibra_path)
        has_local = any(
            p and isinstance(p, str) and not p.startswith("/vsis3/") and os.path.isfile(p)
            for p in (in_path, ref_path, ibra_path)
        )
        if s3_client is not None or has_local:
            self.io_pool = ThreadPoolExecutor(max_workers=self.io_workers)

            def _try_fast(path, ds, band_idx, sparse_fill):
                if not path or not isinstance(path, str):
                    return None
                is_s3 = path.startswith("/vsis3/")
                is_local = (not is_s3) and os.path.isfile(path)
                if not is_s3 and not is_local:
                    return None
                if is_s3 and s3_client is None:
                    return None
                band = ds.GetRasterBand(band_idx)
                comp = ds.GetMetadataItem("COMPRESSION", "IMAGE_STRUCTURE")
                bw, bh = band.GetBlockSize()
                ntx = (ds.RasterXSize + bw - 1) // bw
                nty = (ds.RasterYSize + bh - 1) // bh
                if comp != "DEFLATE" or band.DataType != gdal.GDT_Byte or bw < 128 or bh < 128 or (ntx * nty) < 4:
                    return None
                try:
                    if is_s3:
                        bucket, key = _parse_vsis3_uri(path)
                        return _FastS3TiledBandReader(
                            s3_client, bucket, key, band_idx, ds,
                            sparse_fill=sparse_fill, max_workers=self.io_workers
                        )
                    return _FastS3TiledBandReader(
                        None, None, None, band_idx, ds,
                        sparse_fill=sparse_fill, max_workers=self.io_workers, local_path=path
                    )
                except Exception as exc:
                    print(f"Fast reader fallback for {path} band {band_idx}: {exc}", flush=True)
                    return None

            in_nodata = self.in_band.GetNoDataValue()
            in_fill = int(in_nodata) if in_nodata is not None else 255
            ref_fill = int(self.ref_band.GetNoDataValue() or 0)

            def _try_ref_and_optional_ibra():
                ref_r = _try_fast(ref_path, self.ref_ds, 1, ref_fill)
                ib_r = None
                if ibra_band_idx is not None and not ibra_path:
                    ib_r = _try_fast(ref_path, self.ref_ds, ibra_band_idx, 0)
                return ref_r, ib_r

            with ThreadPoolExecutor(max_workers=3) as init_ex:
                f_in = init_ex.submit(_try_fast, in_path, self.in_ds, 1, in_fill)
                f_ref_pair = init_ex.submit(_try_ref_and_optional_ibra)
                f_ib = None
                if ibra_band_idx is not None and ibra_path:
                    f_ib = init_ex.submit(_try_fast, ibra_path, self.ibra_ds, ibra_band_idx, 0)
                self.in_fast = f_in.result()
                self.ref_fast, ib_from_ref = f_ref_pair.result()
                self.ibra_fast = f_ib.result() if f_ib is not None else ib_from_ref

        def _mode_label(r):
            if r is None:
                return "GDAL"
            return "FastLocal" if r.local_path else "FastS3"

        modes = [
            f"in={_mode_label(self.in_fast)}",
            f"ref={_mode_label(self.ref_fast)}",
        ]
        if ibra_band_idx is not None:
            modes.append(f"ibra={_mode_label(self.ibra_fast)}")
        print(f"Reader backend: {', '.join(modes)} (io_workers={self.io_workers})", flush=True)

    @staticmethod
    def _maybe_build_s3_client(*paths):
        if not any(p and isinstance(p, str) and p.startswith("/vsis3/") for p in paths):
            return None
        try:
            import boto3
            from botocore.config import Config
            region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-west-2"
            cfg = Config(
                region_name=region,
                max_pool_connections=64,
                connect_timeout=10,
                read_timeout=30,
                retries={"max_attempts": 10, "mode": "adaptive"},
            )
            return boto3.client("s3", config=cfg)
        except Exception as exc:
            print(f"boto3 unavailable for FastS3 reader, using GDAL: {exc}", flush=True)
            return None

    def read_in_window(self, in_win):
        """Read raw groundcover/bare input window."""
        if self.in_fast is not None:
            return self.in_fast.read_window(in_win, self.io_pool)
        return _read_gdal_band_with_retry(self.in_band, in_win)

    def read_ref_windows(self, ref_win, need_ibra=False):
        """Read reference percentile (and optional IBRA) window."""
        if self.ref_fast is not None:
            arr = self.ref_fast.read_window(ref_win, self.io_pool)
        else:
            arr = _read_gdal_band_with_retry(self.ref_band, ref_win)
        perc = arr.view(np.int8) if arr.dtype == np.uint8 else arr.astype(np.int8)

        ibra = None
        if need_ibra:
            if self.ibra_fast is not None:
                ib_arr = self.ibra_fast.read_window(ref_win, self.io_pool)
            else:
                ib_arr = _read_gdal_band_with_retry(self.ibra_band, ref_win)
            ibra = ib_arr if ib_arr.dtype == np.uint8 else ib_arr.astype(np.uint8)
        return perc, ibra

    def close(self):
        if self.io_pool is not None:
            self.io_pool.shutdown(wait=True)
            self.io_pool = None
        for r in (self.in_fast, self.ref_fast, self.ibra_fast):
            if r is not None:
                r.close()
        self.in_fast = None
        self.ref_fast = None
        self.ibra_fast = None
        self.in_band = None
        self.in_ds = None
        self.ref_band = None
        self.ref_ds = None
        self.ibra_band = None
        self.ibra_ds = None


def _resolve_bbox_bounds(bbox=None, geojson=None, target_proj_wkt=None):
    """Resolve geographic/projected AOI bounds (minx, miny, maxx, maxy) in target raster CRS."""
    if bbox is not None:
        if len(bbox) != 4:
            raise ValueError(f"Expected 4 bbox values [minx, miny, maxx, maxy], got {bbox}")
        minx, miny, maxx, maxy = map(float, bbox)
        if minx >= maxx or miny >= maxy:
            raise ValueError(f"Invalid bbox bounds: {(minx, miny, maxx, maxy)}")
        return minx, miny, maxx, maxy

    ds = ogr.Open(_to_gdal_path(geojson))
    if ds is None:
        raise ValueError(f"Could not open GeoJSON AOI: {geojson}")
    lyr = ds.GetLayer(0)
    src_srs = lyr.GetSpatialRef()

    dst_srs = None
    transform = None
    if target_proj_wkt:
        dst_srs = osr.SpatialReference()
        dst_srs.ImportFromWkt(target_proj_wkt)
        dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        if src_srs is not None:
            src_srs_norm = src_srs.Clone()
            src_srs_norm.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            if not src_srs_norm.IsSame(dst_srs):
                transform = osr.CoordinateTransformation(src_srs_norm, dst_srs)
        else:
            crs84 = osr.SpatialReference()
            crs84.ImportFromEPSG(4326)
            crs84.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            if not crs84.IsSame(dst_srs):
                transform = osr.CoordinateTransformation(crs84, dst_srs)

    if transform is None:
        ext = lyr.GetExtent(force=1)
        ds = None
        return float(ext[0]), float(ext[2]), float(ext[1]), float(ext[3])

    minx = math.inf
    miny = math.inf
    maxx = -math.inf
    maxy = -math.inf
    for feat in lyr:
        geom = feat.GetGeometryRef()
        if geom is None:
            continue
        g = geom.Clone()
        g.Transform(transform)
        gx0, gx1, gy0, gy1 = g.GetEnvelope()
        minx = min(minx, gx0)
        maxx = max(maxx, gx1)
        miny = min(miny, gy0)
        maxy = max(maxy, gy1)
    ds = None
    if not math.isfinite(minx):
        raise ValueError(f"No valid geometries found in GeoJSON AOI: {geojson}")
    return minx, miny, maxx, maxy


def _chunk_bounds(chunk, chunksize, full_height, y_origin_off, in_h, ref_h, in_to_ref_dy, halfwindow, pad_halo):
    """Compute exact source/reference windows and output slices for a chunk."""
    datastartline = chunk * chunksize
    datastopline = min((chunk + 1) * chunksize, full_height)
    if pad_halo:
        in_read_y0 = max(0, y_origin_off + datastartline - halfwindow)
        in_read_y1 = min(in_h, y_origin_off + datastopline + halfwindow)
    else:
        in_read_y0 = y_origin_off + max(0, datastartline - halfwindow)
        in_read_y1 = y_origin_off + min(full_height, datastopline + halfwindow)
    win_ysize = in_read_y1 - in_read_y0
    ref_start_y = in_to_ref_dy + in_read_y0
    ref_read_y = max(0, min(ref_start_y, ref_h - win_ysize))
    y_slice_start = (y_origin_off + datastartline) - in_read_y0
    y_slice_stop = (y_origin_off + datastopline) - in_read_y0
    return datastartline, datastopline, in_read_y0, in_read_y1, ref_read_y, y_slice_start, y_slice_stop


def _upload_to_s3(local_path, bucket, key, eff_cpus):
    """Upload local output GeoTIFF to S3 using parallel multipart transfer."""
    import boto3
    from boto3.s3.transfer import TransferConfig

    out_mb = os.path.getsize(local_path) / 1048576.0
    s3_uri = f"s3://{bucket}/{key}"
    print(f"Uploading {out_mb:.1f} MB output GeoTIFF to {s3_uri}...", flush=True)
    t_up0 = time.perf_counter()
    tcfg = TransferConfig(
        multipart_threshold=32 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        max_concurrency=min(32, max(10, eff_cpus)),
    )
    boto3.client("s3").upload_file(local_path, bucket, key, Config=tcfg)
    t_up = time.perf_counter() - t_up0
    print(f"Upload completed in {t_up:.1f}s", flush=True)
    return t_up, s3_uri


def run_seasonal_drcm_ibra(
    input_groundcover,
    percentile_file,
    ibra_raster=None,
    output_file=None,
    window=1501,
    chunksize=4096,
    lowerquantile=0.90,
    upperquantile=0.95,
    minvalpercent=0.15,
    test_chip=None,
    bbox=None,
    geojson=None,
    threads=None,
    overview_levels=(4, 8, 16, 32),
    no_overviews=False,
    no_ibra=False,
    save=False
):
    eff_cpus = threads if threads is not None else _get_effective_cpus()
    if threads is not None or os.environ.get("NUMBA_NUM_THREADS") is None:
        import numba
        eff_cpus = max(1, int(eff_cpus))
        numba.set_num_threads(eff_cpus)

    if window % 2 == 0:
        window += 1

    n_spatial_modes = sum(x is not None for x in (test_chip, bbox, geojson))
    if n_spatial_modes > 1:
        raise ValueError("Specify at most one of test_chip, bbox, or geojson.")

    if output_file is None:
        output_file = derive_output_name(input_groundcover, no_ibra=no_ibra)
    elif isinstance(output_file, str) and output_file.endswith("/"):
        output_file = f"{output_file}{derive_output_name(input_groundcover, no_ibra=no_ibra)}"

    _configure_vsis3_defaults()

    mode_str = "Single Global Histogram (IBRA disabled)" if no_ibra else "IBRA-Constrained Regional Histograms"
    print(f"Processing Mode: {mode_str} | effective_cpus={eff_cpus}", flush=True)

    in_gdal_path = _to_gdal_path(input_groundcover)
    ref_gdal_path = _to_gdal_path(percentile_file)
    ibra_gdal_path = _to_gdal_path(ibra_raster) if ibra_raster else None

    print(f"Opening input groundcover: {input_groundcover}", flush=True)
    in_ds = gdal.Open(in_gdal_path, gdal.GA_ReadOnly)
    in_gt = in_ds.GetGeoTransform()
    in_proj = in_ds.GetProjection()
    in_band = in_ds.GetRasterBand(1)
    nodata_in = in_band.GetNoDataValue()
    if nodata_in is None:
        nodata_in = 255
    in_w, in_h = in_ds.RasterXSize, in_ds.RasterYSize

    print(f"Opening reference stats file: {percentile_file}", flush=True)
    ref_ds = gdal.Open(ref_gdal_path, gdal.GA_ReadOnly)
    ref_gt = ref_ds.GetGeoTransform()
    ref_w, ref_h = ref_ds.RasterXSize, ref_ds.RasterYSize

    ibra_band_idx = None
    if not no_ibra:
        if ref_ds.RasterCount >= 2:
            ibra_band_idx = 2
            ibra_gdal_path = None
        elif ibra_gdal_path:
            print(f"Opening separate IBRA raster: {ibra_raster}", flush=True)
            ibra_ds_tmp = gdal.Open(ibra_gdal_path, gdal.GA_ReadOnly)
            ibra_band_idx = 1
            ibra_ds_tmp = None
        else:
            raise ValueError(
                f"{percentile_file} has only 1 band. Please specify --ibra-raster, provide a 2-band masked stats+IBRA raster, or use --no-ibra."
            )

    in_band = None
    in_ds = None
    ref_ds = None

    if test_chip:
        x_origin_off, y_origin_off, full_width, full_height = test_chip
        pad_halo = False
        print(f"Running on test chip: xoff={x_origin_off}, yoff={y_origin_off}, w={full_width}, h={full_height}", flush=True)
    elif bbox is not None or geojson is not None:
        minx, miny, maxx, maxy = _resolve_bbox_bounds(bbox=bbox, geojson=geojson, target_proj_wkt=in_proj)
        if in_gt[2] != 0 or in_gt[4] != 0 or in_gt[1] <= 0 or in_gt[5] >= 0:
            raise ValueError(f"Expected north-up axis-aligned GeoTransform for AOI subsetting, got {in_gt}")
        x_origin_off = max(0, min(in_w, math.floor((minx - in_gt[0]) / in_gt[1])))
        x_end = max(0, min(in_w, math.ceil((maxx - in_gt[0]) / in_gt[1])))
        y_origin_off = max(0, min(in_h, math.floor((maxy - in_gt[3]) / in_gt[5])))
        y_end = max(0, min(in_h, math.ceil((miny - in_gt[3]) / in_gt[5])))
        full_width = x_end - x_origin_off
        full_height = y_end - y_origin_off
        if full_width <= 0 or full_height <= 0:
            raise ValueError(f"AOI does not intersect input raster: x=({x_origin_off}, {x_end}), y=({y_origin_off}, {y_end})")
        pad_halo = True
        print(
            f"Subsetting AOI ({bbox if bbox is not None else geojson}) -> "
            f"xoff={x_origin_off}, yoff={y_origin_off}, w={full_width}, h={full_height} (halo-padded)",
            flush=True
        )
    else:
        x_origin_off, y_origin_off = 0, 0
        full_width = in_w
        full_height = in_h
        pad_halo = False

    chip_gt = (
        in_gt[0] + x_origin_off * in_gt[1],
        in_gt[1],
        in_gt[2],
        in_gt[3] + y_origin_off * in_gt[5],
        in_gt[4],
        in_gt[5],
    )

    in_to_ref_dx = int(round((in_gt[0] - ref_gt[0]) / ref_gt[1]))
    in_to_ref_dy = int(round((in_gt[3] - ref_gt[3]) / ref_gt[5]))
    print(
        f"Spatial alignment offset in reference grid: col={in_to_ref_dx + x_origin_off}, "
        f"row={in_to_ref_dy + y_origin_off}",
        flush=True
    )

    is_s3_output = isinstance(output_file, str) and output_file.startswith("s3://")
    tmp_dir_obj = None
    if is_s3_output:
        tmp_dir_obj = tempfile.TemporaryDirectory(prefix="drcm_")
        local_output_file = os.path.join(tmp_dir_obj.name, os.path.basename(output_file))
    else:
        local_output_file = output_file
        out_dir = os.path.dirname(os.path.abspath(local_output_file))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    halfwindow = window // 2
    if pad_halo:
        in_read_x0 = max(0, x_origin_off - halfwindow)
        in_read_x1 = min(in_w, x_origin_off + full_width + halfwindow)
    else:
        in_read_x0 = x_origin_off
        in_read_x1 = x_origin_off + full_width
    win_xsize = in_read_x1 - in_read_x0
    x_slice_start = x_origin_off - in_read_x0
    x_slice_stop = x_slice_start + full_width

    ref_start_x = in_to_ref_dx + in_read_x0
    ref_read_x = max(0, min(ref_start_x, ref_w - win_xsize))

    io_workers = min(32, max(8, eff_cpus // 2 if eff_cpus >= 16 else eff_cpus))
    reader = _SequentialBandReader(
        in_gdal_path,
        ref_gdal_path,
        ibra_path=ibra_gdal_path,
        ibra_band_idx=ibra_band_idx,
        io_workers=io_workers,
    )

    t_start = time.perf_counter()
    total_pixels_processed = 0
    total_io_wait_s = 0.0
    total_io_work_s = 0.0
    total_compute_s = 0.0
    total_write_s = 0.0
    t_ovr = 0.0
    t_up = 0.0

    def _plan_chunk_fetch(chunk_idx, prev_meta):
        ds, de, y0, y1, ry, ys0, ys1 = _chunk_bounds(
            chunk_idx, chunksize, full_height, y_origin_off, in_h, ref_h, in_to_ref_dy, halfwindow, pad_halo
        )
        wys = y1 - y0
        reuse_ok = (
            prev_meta is not None
            and prev_meta[1] > y0
            and prev_meta[0] <= y0
            and (ry - y0) == (prev_meta[2] - prev_meta[0])
        )
        if reuse_ok:
            ov_y1 = min(y1, prev_meta[1])
            ov_rows = ov_y1 - y0
            tail_h = y1 - ov_y1
            fetch_spec = (ov_y1, ry + ov_rows, tail_h)
        else:
            ov_rows = 0
            fetch_spec = (y0, ry, wys)
        return {
            "datastartline": ds,
            "datastopline": de,
            "y0": y0,
            "y1": y1,
            "ref_y": ry,
            "win_ysize": wys,
            "y_slice_start": ys0,
            "y_slice_stop": ys1,
            "reuse_ok": reuse_ok,
            "ov_rows": ov_rows,
            "fetch_spec": fetch_spec,
        }

    def _fetch_tail(fetch_spec):
        read_y_in, read_y_ref, read_h = fetch_spec
        t0 = time.perf_counter()
        if read_h <= 0:
            return None, None, None, 0.0
        bare = reader.read_in_window((in_read_x0, read_y_in, win_xsize, read_h))
        perc_part, ibra_part = reader.read_ref_windows(
            (ref_read_x, read_y_ref, win_xsize, read_h),
            need_ibra=(not no_ibra),
        )
        return bare, perc_part, ibra_part, (time.perf_counter() - t0)

    prefetch_pool = ThreadPoolExecutor(max_workers=1)
    try:
        print(f"Creating output dataset: {output_file} ({full_width} x {full_height})", flush=True)
        driver = gdal.GetDriverByName("GTiff")
        options = [
            "COMPRESS=DEFLATE",
            "ZLEVEL=1",
            "PREDICTOR=2",
            "TILED=YES",
            "INTERLEAVE=BAND",
            "BIGTIFF=YES",
            f"NUM_THREADS={eff_cpus}",
            "BLOCKXSIZE=512",
            "BLOCKYSIZE=512",
        ]
        out_ds = driver.Create(local_output_file, full_width, full_height, 2, gdal.GDT_Byte, options=options)
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

        nchunks = (full_height + chunksize - 1) // chunksize
        print(f"Processing {full_height} lines in {nchunks} chunks (chunksize={chunksize}, window={window})...\n", flush=True)

        prev_state = None
        cur_plan = _plan_chunk_fetch(0, None)
        next_future = prefetch_pool.submit(_fetch_tail, cur_plan["fetch_spec"])

        for chunk in range(nchunks):
            t_chunk_start = time.perf_counter()
            datastartline = cur_plan["datastartline"]
            datastopline = cur_plan["datastopline"]
            chunk_height = datastopline - datastartline
            in_read_y0 = cur_plan["y0"]
            in_read_y1 = cur_plan["y1"]
            ref_read_y = cur_plan["ref_y"]
            win_ysize = cur_plan["win_ysize"]
            y_slice_start = cur_plan["y_slice_start"]
            y_slice_stop = cur_plan["y_slice_stop"]

            pct_start = (datastartline / full_height) * 100
            print(f"[Chunk {chunk + 1}/{nchunks} - {pct_start:5.1f}%] Starting lines {datastartline}:{datastopline}...", flush=True)

            t_io0 = time.perf_counter()
            bare_tail, perc_tail, ibra_tail, io_work_s = next_future.result()
            total_io_work_s += io_work_s
            gc_tail = convert_to_groundcover(bare_tail, nodata_in, inplace=True) if bare_tail is not None else None
            del bare_tail

            if cur_plan["reuse_ok"]:
                ov_rows = cur_plan["ov_rows"]
                prev_s = in_read_y0 - prev_state["y0"]
                prev_e = prev_s + ov_rows

                groundcover_data = np.empty((win_ysize, win_xsize), dtype=np.int8)
                percentile_data = np.empty((win_ysize, win_xsize), dtype=np.int8)
                groundcover_data[:ov_rows] = prev_state["gc"][prev_s:prev_e]
                percentile_data[:ov_rows] = prev_state["perc"][prev_s:prev_e]
                if not no_ibra:
                    ibra_data = np.empty((win_ysize, win_xsize), dtype=np.uint8)
                    ibra_data[:ov_rows] = prev_state["ibra"][prev_s:prev_e]
                else:
                    ibra_data = None
                prev_state = None

                if gc_tail is not None:
                    groundcover_data[ov_rows:] = gc_tail
                    percentile_data[ov_rows:] = perc_tail
                    if not no_ibra:
                        ibra_data[ov_rows:] = ibra_tail
            else:
                prev_state = None
                groundcover_data = gc_tail
                percentile_data = perc_tail
                ibra_data = ibra_tail
            del gc_tail, perc_tail, ibra_tail
            t_io1 = time.perf_counter()
            io_wait_s = t_io1 - t_io0
            total_io_wait_s += io_wait_s

            # Enqueue next chunk I/O before entering Numba compute + disk write.
            if chunk + 1 < nchunks:
                next_plan = _plan_chunk_fetch(chunk + 1, (in_read_y0, in_read_y1, ref_read_y))
                next_future = prefetch_pool.submit(_fetch_tail, next_plan["fetch_spec"])
                cur_plan = next_plan
            else:
                next_future = None

            t_cmp0 = time.perf_counter()
            if not no_ibra:
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
                if pad_halo and (x_slice_start > 0 or x_slice_stop < win_xsize):
                    chunk_ref = chunk_ref[:, x_slice_start:x_slice_stop]
            else:
                chunk_ref = refpixelfilternodata(
                    percentile_data,
                    groundcover_data,
                    window=window,
                    lowerquantile=lowerquantile,
                    upperquantile=upperquantile,
                    minvalpercent=minvalpercent,
                    r_start=y_slice_start,
                    r_stop=y_slice_stop,
                    c_start=x_slice_start,
                    c_stop=x_slice_stop,
                )

            if chunk + 1 >= nchunks:
                del percentile_data, ibra_data

            chunk_gc = groundcover_data[y_slice_start:y_slice_stop, x_slice_start:x_slice_stop]
            dgc_data, ref_data_scaled = _encode_outputs_2d(chunk_gc, chunk_ref)
            del chunk_ref
            t_cmp1 = time.perf_counter()
            total_compute_s += (t_cmp1 - t_cmp0)

            if chunk + 1 < nchunks:
                prev_state = {
                    "y0": in_read_y0,
                    "y1": in_read_y1,
                    "ref_y0": ref_read_y,
                    "gc": groundcover_data,
                    "perc": percentile_data,
                    "ibra": ibra_data,
                }
            else:
                del groundcover_data

            t_wr0 = time.perf_counter()
            dgc_band.WriteArray(dgc_data, 0, datastartline)
            ref_out_band.WriteArray(ref_data_scaled, 0, datastartline)
            del dgc_data, ref_data_scaled
            dgc_band.FlushCache()
            ref_out_band.FlushCache()
            t_wr1 = time.perf_counter()
            total_write_s += (t_wr1 - t_wr0)

            chunk_pixels = chunk_height * full_width
            total_pixels_processed += chunk_pixels
            t_chunk = time.perf_counter() - t_chunk_start
            t_elapsed = time.perf_counter() - t_start
            rate = chunk_pixels / t_chunk if t_chunk > 0 else 0
            overall_rate = total_pixels_processed / t_elapsed if t_elapsed > 0 else 0

            pct_done = (datastopline / full_height) * 100
            remaining_pixels = (full_height - datastopline) * full_width
            eta_seconds = remaining_pixels / overall_rate if overall_rate > 0 else 0

            print(
                f"[Chunk {chunk + 1}/{nchunks} - {pct_done:5.1f}%] Lines {datastartline}:{datastopline} done in {t_chunk:.1f}s "
                f"(io_wait={io_wait_s:.1f}s, io_bg={io_work_s:.1f}s, numba={t_cmp1-t_cmp0:.1f}s, write={t_wr1-t_wr0:.1f}s, {rate/1e6:.2f}M px/s) | "
                f"Elapsed: {format_time(t_elapsed)} | Remaining: ~{format_time(eta_seconds)}\n",
                flush=True
            )

        out_ds.FlushCache()

        if not no_overviews and overview_levels:
            ovr_str = ", ".join(str(lvl) for lvl in overview_levels)
            print(f"Calculating internal overviews (average, levels: {ovr_str})...", flush=True)
            t_ovr_start = time.perf_counter()
            prev_gdal_threads = gdal.GetConfigOption("GDAL_NUM_THREADS")
            gdal.SetConfigOption("COMPRESS_OVERVIEW", "DEFLATE")
            gdal.SetConfigOption("PREDICTOR_OVERVIEW", "1")
            gdal.SetConfigOption("NUM_THREADS", str(eff_cpus))
            gdal.SetConfigOption("GDAL_NUM_THREADS", str(eff_cpus))
            try:
                out_ds.BuildOverviews("AVERAGE", list(overview_levels), callback=gdal.TermProgress_nocb)
                out_ds.FlushCache()
            finally:
                gdal.SetConfigOption("GDAL_NUM_THREADS", prev_gdal_threads)
            t_ovr = time.perf_counter() - t_ovr_start
            print(f"\nOverviews completed in {t_ovr:.1f}s", flush=True)

        dgc_band = None
        ref_out_band = None
        out_ds = None

        if is_s3_output:
            bucket, key = _parse_s3_uri(output_file)
            t_up, _ = _upload_to_s3(local_output_file, bucket, key, eff_cpus)
        elif save:
            save_key = f"{SAVE_DIR}/{os.path.basename(local_output_file)}"
            t_up, output_file = _upload_to_s3(local_output_file, SAVE_BUCKET, save_key, eff_cpus)
            if os.path.exists(local_output_file):
                os.remove(local_output_file)

        t_total = time.perf_counter() - t_start
        overall_rate = total_pixels_processed / t_total if t_total > 0 else 0
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        print(
            f"\n[100.0%] Finished generating {output_file}\n"
            f"Total: {total_pixels_processed:,} pixels in {format_time(t_total)} "
            f"(wall={t_total:.2f}s | avg {overall_rate/1e6:.2f}M px/s | "
            f"io_wait={total_io_wait_s:.1f}s, io_bg={total_io_work_s:.1f}s, "
            f"numba={total_compute_s:.1f}s, write={total_write_s:.1f}s, ovr={t_ovr:.1f}s, up={t_up:.1f}s, "
            f"max_rss={rss_mb:.1f}MB)",
            flush=True
        )
        return output_file
    finally:
        prefetch_pool.shutdown(wait=True)
        reader.close()
        if tmp_dir_obj is not None:
            tmp_dir_obj.cleanup()


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
        bbox=args.bbox,
        geojson=args.geojson,
        threads=args.threads,
        overview_levels=args.overviews,
        no_overviews=args.no_overviews,
        no_ibra=args.no_ibra,
        save=args.save
    )


if __name__ == "__main__":
    main()
