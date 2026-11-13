#!/usr/bin/env python3
"""
run_aus_batch_worker.py

Multi-season AWS Batch array worker ("best-of-both-worlds" pattern):
- Reads a deterministic season shard manifest (e.g. 147 seasons across 15 array tasks).
- Skips seasons whose validated S3 outputs already exist (idempotent across Spot retries).
- Downloads & byte-verifies the reference raster once per container onto /tmp so FastLocal
  (thread-safe POSIX pread + parallel zlib decode) keeps it warm in Linux page cache.
- Overlaps Season N S3 multipart upload with Season N+1 compute to maximize CPU utilization.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import os
from pathlib import Path
import resource
import shutil
import time
import urllib.request

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from osgeo import gdal

import seasonalDRCMIbra


def _imds_meta(path):
    """Best-effort EC2 IMDSv2 metadata lookup."""
    try:
        req = urllib.request.Request(
            "http://169.254.169.254/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        )
        token = urllib.request.urlopen(req, timeout=2).read().decode("utf-8")
        req2 = urllib.request.Request(
            f"http://169.254.169.254/latest/meta-data/{path}",
            headers={"X-aws-ec2-metadata-token": token},
        )
        return urllib.request.urlopen(req2, timeout=2).read().decode("utf-8")
    except Exception:
        return "unknown"


def _build_s3_client():
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-west-2"
    cfg = Config(
        region_name=region,
        max_pool_connections=64,
        connect_timeout=10,
        read_timeout=60,
        retries={"max_attempts": 10, "mode": "adaptive"},
    )
    return boto3.client("s3", config=cfg)


def _load_manifest(s3, manifest_uri):
    if manifest_uri.startswith("s3://"):
        bucket, key = seasonalDRCMIbra._parse_s3_uri(manifest_uri)
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        return json.loads(body.decode("utf-8"))
    return json.loads(Path(manifest_uri).read_text(encoding="utf-8"))


def _head_s3_size(s3, bucket, key):
    try:
        resp = s3.head_object(Bucket=bucket, Key=key)
        return int(resp["ContentLength"])
    except Exception:
        return None


def _ensure_local_ref_cache(s3, bucket, ref_key, cache_dir, min_work_headroom_bytes=16 * 1024 ** 3):
    """
    Download and verify ref_key once onto cache_dir when sufficient disk exists;
    fall back cleanly to s3:// streaming if disk headroom is constrained.
    """
    expected_bytes = _head_s3_size(s3, bucket, ref_key)
    if expected_bytes is None:
        raise RuntimeError(f"Reference object missing on S3: s3://{bucket}/{ref_key}")

    os.makedirs(cache_dir, exist_ok=True)
    local_ref = os.path.join(cache_dir, os.path.basename(ref_key))
    tmp_ref = f"{local_ref}.part"

    if os.path.exists(local_ref) and os.path.getsize(local_ref) == expected_bytes:
        print(
            f"REF_CACHE reuse verified local file: {local_ref} ({expected_bytes / 1048576.0:.1f} MB)",
            flush=True,
        )
        return local_ref, False

    free_bytes = shutil.disk_usage(cache_dir).free
    needed_bytes = expected_bytes + min_work_headroom_bytes
    if free_bytes < needed_bytes:
        print(
            f"REF_CACHE skipped due to limited free disk on {cache_dir}: "
            f"{free_bytes / 1024**3:.2f} GiB < {needed_bytes / 1024**3:.2f} GiB; streaming /vsis3/.",
            flush=True,
        )
        return f"s3://{bucket}/{ref_key}", False

    if os.path.exists(tmp_ref):
        os.remove(tmp_ref)

    t0 = time.perf_counter()
    print(
        f"REF_CACHE downloading s3://{bucket}/{ref_key} ({expected_bytes / 1048576.0:.1f} MB) -> {local_ref}...",
        flush=True,
    )
    tcfg = TransferConfig(
        multipart_threshold=32 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        max_concurrency=32,
    )
    s3.download_file(bucket, ref_key, tmp_ref, Config=tcfg)
    actual_bytes = os.path.getsize(tmp_ref)
    if actual_bytes != expected_bytes:
        os.remove(tmp_ref)
        raise RuntimeError(f"REF_CACHE size mismatch: got {actual_bytes}, expected {expected_bytes}")
    os.replace(tmp_ref, local_ref)
    print(
        f"REF_CACHE ready in {time.perf_counter() - t0:.2f}s ({expected_bytes / 1048576.0:.1f} MB)",
        flush=True,
    )
    return local_ref, True


def _upload_and_cleanup(local_path, bucket, out_key, min_output_bytes):
    t0 = time.perf_counter()
    sz = os.path.getsize(local_path)
    if sz < min_output_bytes:
        raise RuntimeError(f"Output GeoTIFF unexpectedly small ({sz} bytes < {min_output_bytes}): {local_path}")

    s3 = _build_s3_client()
    tcfg = TransferConfig(
        multipart_threshold=32 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        max_concurrency=16,
    )
    s3.upload_file(local_path, bucket, out_key, Config=tcfg)
    remote_sz = _head_s3_size(s3, bucket, out_key)
    if remote_sz != sz:
        raise RuntimeError(f"Uploaded size mismatch for s3://{bucket}/{out_key}: remote={remote_sz} local={sz}")

    try:
        os.remove(local_path)
    except OSError:
        pass
    elapsed = time.perf_counter() - t0
    print(
        f"UPLOAD_OK s3://{bucket}/{out_key} ({sz / 1048576.0:.1f} MB in {elapsed:.1f}s)",
        flush=True,
    )
    return remote_sz


def parse_args():
    p = argparse.ArgumentParser(description="Multi-season AWS Batch worker for Continental DRCM.")
    p.add_argument("--manifest", required=True, help="Local path or s3:// URI to season shard manifest JSON")
    p.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help="Explicit shard index (0-based). Falls back to AWS_BATCH_JOB_ARRAY_INDEX env var, then 0.",
    )
    p.add_argument("--window", type=int, default=1501)
    p.add_argument("--chunksize", type=int, default=16384)
    p.add_argument("--minvalpercent", type=float, default=0.10)
    p.add_argument("--no-ibra", action="store_true", default=True)
    p.add_argument("--ibra", dest="no_ibra", action="store_false")
    p.add_argument("--ibra-raster", default=None, help="Optional separate IBRA raster path or s3:// URI")
    p.add_argument("--bbox", nargs=4, type=float, default=None, metavar=("MINX", "MINY", "MAXX", "MAXY"))
    p.add_argument("--geojson", default=None, help="Optional AOI GeoJSON path or s3:// URI")
    p.add_argument("--threads", type=int, default=None, help="Optional Numba thread count override")
    p.add_argument("--ref-cache-dir", default="/tmp")
    p.add_argument("--work-dir", default="/tmp/drcm_work")
    p.add_argument(
        "--min-output-bytes",
        type=int,
        default=None,
        help="Minimum valid output size in bytes (default: 100000000 full-res, 1024 when --bbox/--geojson is active)",
    )
    p.add_argument("--no-overviews", action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    shard_idx = (
        args.shard_index
        if args.shard_index is not None
        else int(os.environ.get("AWS_BATCH_JOB_ARRAY_INDEX", "0"))
    )

    s3 = _build_s3_client()
    manifest = _load_manifest(s3, args.manifest)
    bucket = manifest["bucket"]
    ref_key = manifest["ref_key"]
    output_prefix = manifest["output_prefix"].rstrip("/") + "/"
    shards = manifest["shards"]

    if shard_idx < 0 or shard_idx >= len(shards):
        raise IndexError(f"Shard index {shard_idx} out of range (manifest has {len(shards)} shards)")

    geojson = args.geojson if args.geojson is not None else manifest.get("geojson")
    bbox = args.bbox if args.bbox is not None else manifest.get("bbox")
    min_output_bytes = (
        args.min_output_bytes
        if args.min_output_bytes is not None
        else (1_024 if (geojson is not None or bbox is not None) else 100_000_000)
    )
    ibra_raster = args.ibra_raster
    if ibra_raster is None and manifest.get("ibra_key"):
        ibra_key = manifest["ibra_key"]
        ibra_raster = ibra_key if ibra_key.startswith("s3://") else f"s3://{bucket}/{ibra_key}"

    shard_keys = shards[shard_idx]
    host_info = (
        f"type={_imds_meta('instance-type')} az={_imds_meta('placement/availability-zone')} "
        f"id={_imds_meta('instance-id')}"
    )
    print(
        f"WORKER_START shard={shard_idx}/{len(shards)} seasons={len(shard_keys)} "
        f"no_ibra={args.no_ibra} chunksize={args.chunksize} minvalpercent={args.minvalpercent} "
        f"min_output_bytes={min_output_bytes} output=s3://{bucket}/{output_prefix} | {host_info}",
        flush=True,
    )

    pending = []
    skipped = 0
    for in_key in shard_keys:
        out_name = seasonalDRCMIbra.derive_output_name(in_key, no_ibra=args.no_ibra)
        out_key = f"{output_prefix}{out_name}"
        existing_sz = _head_s3_size(s3, bucket, out_key)
        if existing_sz is not None and existing_sz >= min_output_bytes:
            print(f"SKIP_EXISTING s3://{bucket}/{out_key} ({existing_sz / 1048576.0:.1f} MB)", flush=True)
            skipped += 1
        else:
            pending.append((in_key, out_name, out_key))

    if not pending:
        print(f"SHARD_DONE shard={shard_idx} processed=0 skipped={skipped} wall_s=0.00s", flush=True)
        return

    t_worker0 = time.perf_counter()
    os.makedirs(args.work_dir, exist_ok=True)
    ref_path, cleanup_ref = _ensure_local_ref_cache(s3, bucket, ref_key, args.ref_cache_dir)

    pending_upload = None
    processed = 0
    up_pool = ThreadPoolExecutor(max_workers=1)

    def _drain_pending_upload():
        nonlocal pending_upload
        if pending_upload is None:
            return
        up_key, fut = pending_upload
        pending_upload = None
        fut.result()

    try:
        for idx, (in_key, out_name, out_key) in enumerate(pending, start=1):
            try:
                gdal.VSICurlClearCache()
            except Exception:
                pass
            gc.collect()

            local_out = os.path.join(args.work_dir, out_name)
            if os.path.exists(local_out):
                os.remove(local_out)

            t_s0 = time.perf_counter()
            print(
                f"\n=== [Shard {shard_idx} | Season {idx}/{len(pending)}] {in_key} -> s3://{bucket}/{out_key} ===",
                flush=True,
            )
            in_uri = in_key if in_key.startswith("s3://") else f"s3://{bucket}/{in_key}"
            seasonalDRCMIbra.run_seasonal_drcm_ibra(
                input_groundcover=in_uri,
                percentile_file=ref_path,
                ibra_raster=ibra_raster,
                output_file=local_out,
                window=args.window,
                chunksize=args.chunksize,
                minvalpercent=args.minvalpercent,
                bbox=bbox,
                geojson=geojson,
                threads=args.threads,
                no_ibra=args.no_ibra,
                overview_levels=(4, 8, 16, 32),
                no_overviews=args.no_overviews,
            )

            # Wait for any prior season's upload before enqueueing the next so /tmp never holds >2 outputs.
            _drain_pending_upload()
            fut = up_pool.submit(_upload_and_cleanup, local_out, bucket, out_key, min_output_bytes)
            pending_upload = (out_key, fut)
            processed += 1
            print(
                f"=== [Shard {shard_idx} | Season {idx}/{len(pending)}] compute+ovr finished in "
                f"{time.perf_counter() - t_s0:.1f}s (upload overlapped) ===",
                flush=True,
            )

        _drain_pending_upload()
        total_s = time.perf_counter() - t_worker0
        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        print(
            f"\nSHARD_DONE shard={shard_idx} processed={processed} skipped={skipped} "
            f"wall_s={total_s:.2f}s avg_per_season_s={total_s / max(1, processed):.2f}s max_rss={rss_mb:.1f}MB",
            flush=True,
        )
    finally:
        up_pool.shutdown(wait=True)
        if cleanup_ref and isinstance(ref_path, str) and os.path.exists(ref_path):
            try:
                os.remove(ref_path)
            except OSError:
                pass
        shutil.rmtree(args.work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
