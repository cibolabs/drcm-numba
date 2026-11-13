#!/usr/bin/env python3
"""
submit_batch_jobs.py

Cibo Labs AWS Batch submission and monitoring orchestrator for the Seasonal DRCM pipeline.

Supports:
- Discovering seasonal groundcover GeoTIFFs in S3 (with optional YYYYMM range and skip-existing filters).
- Building balanced round-robin season shard manifests for multi-season Spot array workers
  (run_aus_batch_worker.py) or 1-season-per-task array runs.
- Staging runtime scripts, optional GeoJSON AOI, and shard manifest to S3 without ECR rebuilds.
- Ensuring Graviton3-safe (NUMBA_CPU_NAME=generic, NUMBA_CPU_FEATURES=+neon) and S3-tuned GDAL
  job definitions & container environment overrides.
- Pre-flight smoke tests (--smoke-test), safe single-shard retries (--retry-shard), dry-run
  inspection (--dry-run), and resilient describe_jobs + CloudWatch log monitoring (--wait / --watch-job-id).
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time

import boto3
from botocore.config import Config

import seasonalDRCMIbra

DEFAULT_BUCKET = "cibo-dat-oregon"
DEFAULT_GC_PREFIX = "GC/lztmre_aus_m"
DEFAULT_SEASON_REGEX = r"^GC/lztmre_aus_m(\d{6})(\d{6})_dp8a2(_vinterim)?\.tif$"
DEFAULT_REF_KEY = "DRCM/lztmre_aus_m198912202012_dixa2_change_slope_ibra.tif"
DEFAULT_OUTPUT_PREFIX = "DRCM/aus/"
DEFAULT_STAGING_PREFIX = "DRCM/_batch_scripts"
DEFAULT_REGION = "us-west-2"
DEFAULT_JOB_QUEUE = "CiboBatchProcessingJobQueueESANonProd"
DEFAULT_JOB_DEFINITION = "drcm-aus-multi-season"
DEFAULT_ECR_IMAGE = "331561773057.dkr.ecr.us-west-2.amazonaws.com/cibobase:baseubuntu_2404_numba"
DEFAULT_LOG_GROUP = "/aws/batch/cibo-processor-batch-test"
DEFAULT_LOG_STREAM_PREFIX = "drcm-test"
DEFAULT_MIN_OUTPUT_BYTES = 100_000_000
DEFAULT_AOI_MIN_OUTPUT_BYTES = 1_024

WORKER_SCRIPT_FILES = (
    "refpix_numba.py",
    "refpix_numba_ibra.py",
    "seasonalDRCMIbra.py",
    "run_aus_batch_worker.py",
    "test_ibra_drcm.py",
)


def load_env_file(env_path):
    """Load key=value pairs from a .env file into os.environ without overwriting existing vars."""
    if not env_path:
        return
    p = Path(env_path)
    if not p.is_file():
        return
    for raw_line in p.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def build_batch_environment(region=DEFAULT_REGION):
    """Return verified AWS Batch container environment variables for Graviton3 + /vsis3/ GDAL."""
    return [
        {"name": "PYTHONUNBUFFERED", "value": "1"},
        {"name": "AWS_REGION", "value": region},
        {"name": "AWS_DEFAULT_REGION", "value": region},
        # Prevent LLVM/llvmlite from emitting ARM SVE instructions when host/container disables SVE
        {"name": "NUMBA_CPU_NAME", "value": "generic"},
        {"name": "NUMBA_CPU_FEATURES", "value": "+neon"},
        # High-throughput /vsis3/ GDAL HTTP & retry configuration
        {"name": "GDAL_CACHEMAX", "value": "1024000000"},
        {"name": "VSI_CACHE", "value": "True"},
        {"name": "VSI_CACHE_SIZE", "value": "1024000000"},
        {"name": "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "value": "YES"},
        {"name": "GDAL_HTTP_MULTIPLEX", "value": "YES"},
        {"name": "GDAL_HTTP_MAX_RETRY", "value": "20"},
        {"name": "GDAL_HTTP_RETRY_DELAY", "value": "1"},
        {"name": "GDAL_HTTP_RETRY_CODES", "value": "429,500,502,503,504,0"},
        {"name": "GDAL_DISABLE_READDIR_ON_OPEN", "value": "EMPTY_DIR"},
        {"name": "CPL_VSIL_CURL_ALLOWED_EXTENSIONS", "value": ".tif,.TIF,.vrt,.geojson"},
        {"name": "CPL_ZIP_ENCODING", "value": "UTF-8"},
        {"name": "CPL_AWS_WEB_IDENTITY_ENABLE", "value": "NO"},
    ]


def filter_season_items(items, pattern=DEFAULT_SEASON_REGEX, start_ym=None, end_ym=None):
    """
    Filter and sort (key, size_bytes) S3 items by regex and optional YYYYMM bounds.

    When the regex captures a 6-digit start group (e.g. m200912201002 -> '200912'),
    start_ym / end_ym filter inclusively on that YYYYMM token.
    """
    rx = re.compile(pattern)
    matched = []
    for key, size_bytes in items:
        m = rx.match(key)
        if not m:
            continue
        if m.lastindex and m.lastindex >= 1:
            ym = m.group(1)
            if start_ym and ym < start_ym:
                continue
            if end_ym and ym > end_ym:
                continue
        matched.append((key, int(size_bytes)))
    matched.sort(key=lambda x: x[0])
    return matched


def build_shard_manifest(
    items,
    bucket=DEFAULT_BUCKET,
    ref_key=DEFAULT_REF_KEY,
    output_prefix=DEFAULT_OUTPUT_PREFIX,
    num_shards=15,
    geojson=None,
    bbox=None,
    ibra_key=None,
):
    """
    Distribute (key, size_bytes) items round-robin into balanced shards and build manifest dict.

    If num_shards <= 0, assigns one shard per season item.
    """
    if not items:
        raise ValueError("Cannot build shard manifest from an empty season list.")

    target_shards = len(items) if num_shards <= 0 else min(int(num_shards), len(items))
    shards = [[] for _ in range(target_shards)]
    shard_bytes = [0 for _ in range(target_shards)]

    for idx, (key, size_bytes) in enumerate(items):
        s_idx = idx % target_shards
        shards[s_idx].append(key)
        shard_bytes[s_idx] += int(size_bytes)

    manifest = {
        "bucket": bucket,
        "ref_key": ref_key,
        "output_prefix": output_prefix.rstrip("/") + "/",
        "total_seasons": len(items),
        "num_shards": target_shards,
        "assignment": "round_robin",
        "shards": shards,
    }
    if geojson is not None:
        manifest["geojson"] = geojson
    if bbox is not None:
        manifest["bbox"] = list(map(float, bbox))
    if ibra_key is not None:
        manifest["ibra_key"] = ibra_key
    return manifest, shard_bytes


def build_worker_command(
    staging_s3_uri,
    manifest_name="aus_seasons_manifest.json",
    work_subdir="drcm_batch",
    window=1501,
    chunksize=16384,
    minvalpercent=0.10,
    no_ibra=True,
    no_overviews=False,
    min_output_bytes=100_000_000,
    explicit_shard_index=None,
):
    """Construct container entrypoint command syncing staged scripts and launching the worker."""
    local_dir = f"/tmp/{work_subdir}"
    worker_args = [
        "python3", "-u", "run_aus_batch_worker.py",
        "--manifest", f"{local_dir}/{manifest_name}",
        "--window", str(int(window)),
        "--chunksize", str(int(chunksize)),
        "--minvalpercent", str(float(minvalpercent)),
        "--min-output-bytes", str(int(min_output_bytes)),
        "--no-ibra" if no_ibra else "--ibra",
    ]
    if no_overviews:
        worker_args.append("--no-overviews")
    if explicit_shard_index is not None:
        worker_args.extend(["--shard-index", str(int(explicit_shard_index))])

    shell_lines = [
        "set -e",
        f"mkdir -p {local_dir} && cd {local_dir}",
        f"aws s3 sync {staging_s3_uri.rstrip('/')}/ . --quiet",
        " ".join(worker_args),
    ]
    return ["bash", "-lc", "\n".join(shell_lines)]


def discover_s3_seasons(s3, bucket, gc_prefix, pattern, start_ym=None, end_ym=None):
    """List and filter matching seasonal GeoTIFF objects from S3."""
    raw_items = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=gc_prefix):
        for obj in page.get("Contents", []):
            raw_items.append((obj["Key"], int(obj["Size"])))
    return filter_season_items(raw_items, pattern=pattern, start_ym=start_ym, end_ym=end_ym)


def filter_existing_outputs(s3, items, bucket, output_prefix, no_ibra, min_output_bytes):
    """Skip seasons whose output GeoTIFF already exists in S3 above min_output_bytes."""
    prefix = output_prefix.rstrip("/") + "/"
    existing = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            existing[obj["Key"]] = int(obj["Size"])

    pending = []
    skipped = 0
    for key, size_bytes in items:
        out_name = seasonalDRCMIbra.derive_output_name(key, no_ibra=no_ibra)
        out_key = f"{prefix}{out_name}"
        if existing.get(out_key, 0) >= min_output_bytes:
            skipped += 1
        else:
            pending.append((key, size_bytes))
    return pending, skipped


def validate_queue_and_compute_env(batch, job_queue, vcpus, memory_mb):
    """Verify target AWS Batch queue and compute environments are ENABLED/VALID."""
    queues = batch.describe_job_queues(jobQueues=[job_queue]).get("jobQueues", [])
    if not queues:
        raise ValueError(f"AWS Batch job queue not found: {job_queue}")
    q = queues[0]
    if q.get("state") != "ENABLED" or q.get("status") != "VALID":
        raise RuntimeError(f"Job queue {job_queue} unhealthy: state={q.get('state')} status={q.get('status')}")

    ce_arns = [c["computeEnvironment"] for c in q.get("computeEnvironmentOrder", [])]
    ces = batch.describe_compute_environments(computeEnvironments=ce_arns).get("computeEnvironments", [])
    healthy_ces = []
    for ce in ces:
        name = ce["computeEnvironmentName"]
        cr = ce.get("computeResources", {})
        healthy_ces.append(
            f"{name}(state={ce.get('state')}, status={ce.get('status')}, "
            f"type={cr.get('type')}, maxvCpus={cr.get('maxvCpus')}, desired={cr.get('desiredvCpus')})"
        )
        if ce.get("state") != "ENABLED" or ce.get("status") != "VALID":
            raise RuntimeError(f"Compute environment {name} unhealthy: {ce.get('state')}/{ce.get('status')}")

    print(f"Queue validated: {job_queue} -> {', '.join(healthy_ces)}", flush=True)
    print(f"Requested per-task container resources: VCPU={vcpus}, MEMORY={memory_mb} MiB", flush=True)


def ensure_job_definition(
    batch,
    jd_name,
    image,
    vcpus,
    memory_mb,
    attempt_duration_s,
    region,
    log_group,
    log_stream_prefix,
    force_register=False,
    job_role_arn=None,
    exec_role_arn=None,
):
    """Resolve active job definition or register a new revision with Graviton/S3 settings."""
    if ":" in jd_name and not force_register:
        res = batch.describe_job_definitions(jobDefinitions=[jd_name]).get("jobDefinitions", [])
        if res:
            jd = res[0]
            return f"{jd['jobDefinitionName']}:{jd['revision']}"
        raise ValueError(f"Specified job definition revision not found: {jd_name}")

    base_name = jd_name.split(":")[0]
    active = batch.describe_job_definitions(jobDefinitionName=base_name, status="ACTIVE").get("jobDefinitions", [])
    if active and not force_register:
        active.sort(key=lambda x: x["revision"], reverse=True)
        latest = active[0]
        return f"{latest['jobDefinitionName']}:{latest['revision']}"

    container_props = {
        "image": image,
        "command": ["python3", "--version"],
        "resourceRequirements": [
            {"type": "VCPU", "value": str(int(vcpus))},
            {"type": "MEMORY", "value": str(int(memory_mb))},
        ],
        "environment": build_batch_environment(region=region),
        "logConfiguration": {
            "logDriver": "awslogs",
            "options": {
                "awslogs-group": log_group,
                "awslogs-region": region,
                "awslogs-stream-prefix": log_stream_prefix,
            },
        },
    }
    if job_role_arn:
        container_props["jobRoleArn"] = job_role_arn
    if exec_role_arn:
        container_props["executionRoleArn"] = exec_role_arn

    reg = batch.register_job_definition(
        jobDefinitionName=base_name,
        type="container",
        containerProperties=container_props,
        retryStrategy={
            "attempts": 4,
            "evaluateOnExit": [
                {"onStatusReason": "Host EC2*", "action": "RETRY"},
                {"onExitCode": "1", "action": "RETRY"},
                {"onReason": "*", "action": "EXIT"},
            ],
        },
        timeout={"attemptDurationSeconds": int(attempt_duration_s)},
    )
    resolved = f"{reg['jobDefinitionName']}:{reg['revision']}"
    print(f"Registered AWS Batch job definition: {resolved}", flush=True)
    return resolved


def stage_scripts_and_manifest(
    s3,
    repo_dir,
    bucket,
    staging_key_prefix,
    manifest,
    local_geojson=None,
    include_test_data=True,
    require_test_data=False,
):
    """Upload pipeline scripts, optional test_data/ chips, manifest JSON, and local GeoJSON to S3 staging prefix."""
    prefix = staging_key_prefix.strip("/")
    for fname in WORKER_SCRIPT_FILES:
        src = Path(repo_dir) / fname
        if not src.is_file():
            raise FileNotFoundError(f"Required worker script missing: {src}")
        s3.upload_file(str(src), bucket, f"{prefix}/{fname}")

    if include_test_data:
        test_data_dir = Path(repo_dir) / "test_data"
        chip_files = sorted(p for p in test_data_dir.iterdir() if p.is_file()) if test_data_dir.is_dir() else []
        if require_test_data and not chip_files:
            raise FileNotFoundError(f"Required test_data/ directory missing or empty for --smoke-test: {test_data_dir}")
        for chip in chip_files:
            s3.upload_file(str(chip), bucket, f"{prefix}/test_data/{chip.name}")

    if local_geojson and not str(local_geojson).startswith(("s3://", "/vsis3/")):
        gj_path = Path(local_geojson)
        if not gj_path.is_file():
            raise FileNotFoundError(f"GeoJSON AOI file not found: {gj_path}")
        s3.upload_file(str(gj_path), bucket, f"{prefix}/{gj_path.name}")

    manifest_key = f"{prefix}/aus_seasons_manifest.json"
    s3.put_object(
        Bucket=bucket,
        Key=manifest_key,
        Body=json.dumps(manifest, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    staging_uri = f"s3://{bucket}/{prefix}"
    print(f"Staged worker scripts and manifest at {staging_uri}/", flush=True)
    return staging_uri


def _tail_cloudwatch(logs_cli, log_group, log_stream, next_token=None, start_from_head=None, limit=None):
    kw = {
        "logGroupName": log_group,
        "logStreamName": log_stream,
        "startFromHead": (next_token is None) if start_from_head is None else bool(start_from_head),
    }
    if next_token is not None:
        kw["nextToken"] = next_token
    if limit is not None:
        kw["limit"] = int(limit)
    try:
        resp = logs_cli.get_log_events(**kw)
    except Exception as exc:
        return next_token, [f"[CW note: {exc}]"]
    events = resp.get("events", [])
    return resp.get("nextForwardToken", next_token), [e["message"] for e in events]


def _extract_log_stream(job_desc):
    ls = job_desc.get("container", {}).get("logStreamName")
    if not ls and job_desc.get("attempts"):
        ls = job_desc["attempts"][-1].get("container", {}).get("logStreamName")
    return ls


def watch_batch_job(
    batch,
    logs_cli,
    job_id,
    log_group=DEFAULT_LOG_GROUP,
    poll_s=30,
    max_runnable_s=720,
    timeout_s=6000,
    array_sample_idx=0,
):
    """
    Monitor an AWS Batch single or array job with CloudWatch tailing, Spot RUNNABLE stall
    detection, and immediate failed-child inspection. Returns 0 on SUCCEEDED, non-zero otherwise.
    """
    t0 = time.time()
    last_status = None
    last_summary = None
    last_ls = None
    cw_token = None
    runnable_since = None
    reported_failed_children = set()

    while True:
        elapsed = time.time() - t0
        if elapsed > timeout_s:
            print(f"WATCHER_TIMEOUT after {elapsed:.1f}s waiting for {job_id}", flush=True)
            return 3

        j = batch.describe_jobs(jobs=[job_id])["jobs"][0]
        status = j["status"]
        age_s = time.time() - j["createdAt"] / 1000.0
        reason = j.get("statusReason")
        is_array = "arrayProperties" in j and j["arrayProperties"].get("size", 0) > 0
        summary = j.get("arrayProperties", {}).get("statusSummary") if is_array else None

        if is_array:
            child_id = f"{job_id}:{array_sample_idx}"
            try:
                cj = batch.describe_jobs(jobs=[child_id])["jobs"][0]
                ls = _extract_log_stream(cj)
            except Exception:
                ls = None
        else:
            ls = _extract_log_stream(j)

        any_runnable = (status == "RUNNABLE") or (
            bool(summary)
            and summary.get("RUNNABLE", 0) > 0
            and summary.get("RUNNING", 0) == 0
            and summary.get("STARTING", 0) == 0
        )
        if any_runnable:
            if runnable_since is None:
                runnable_since = time.time()
        else:
            runnable_since = None

        if status != last_status or summary != last_summary or (ls and ls != last_ls) or int(elapsed) % 60 < poll_s:
            q_name = j["jobQueue"].split("/")[-1]
            ce_str = ""
            try:
                q = batch.describe_job_queues(jobQueues=[q_name])["jobQueues"][0]
                ce_name = q["computeEnvironmentOrder"][0]["computeEnvironment"].split("/")[-1]
                ce = batch.describe_compute_environments(computeEnvironments=[ce_name])["computeEnvironments"][0]
                cr = ce.get("computeResources", {})
                ce_str = f" ce={ce_name}(desired={cr.get('desiredvCpus')},status={ce.get('status')})"
            except Exception:
                pass
            arr_str = f" summary={summary}" if summary is not None else ""
            print(f"[+{age_s:5.1f}s] status={status} reason={reason}{ce_str}{arr_str}", flush=True)
            last_status = status
            last_summary = dict(summary) if summary else None

        if ls and ls != last_ls:
            print(f"  logStream={ls}", flush=True)
            cw_token = None
            last_ls = ls

        if ls:
            cw_token, msgs = _tail_cloudwatch(logs_cli, log_group, ls, cw_token)
            for msg in msgs:
                print(f"  | {msg}", flush=True)

        if is_array and summary and summary.get("FAILED", 0) > 0:
            try:
                failed_list = batch.list_jobs(arrayJobId=job_id, jobStatus="FAILED", maxResults=20)["jobSummaryList"]
                new_failed = [f["jobId"] for f in failed_list if f["jobId"] not in reported_failed_children]
                if new_failed:
                    for fj in batch.describe_jobs(jobs=new_failed)["jobs"]:
                        reported_failed_children.add(fj["jobId"])
                        fls = _extract_log_stream(fj)
                        print(f"CHILD_FAILED {fj['jobId']} reason={fj.get('statusReason')} logStream={fls}", flush=True)
                        if fls:
                            _, fmsgs = _tail_cloudwatch(logs_cli, log_group, fls, None, start_from_head=False, limit=50)
                            for msg in fmsgs[-25:]:
                                print(f"  ! {msg}", flush=True)
            except Exception as exc:
                print(f"  Failed-child inspect note: {exc}", flush=True)

        if runnable_since is not None and (time.time() - runnable_since) > max_runnable_s:
            print(
                f"WATCHER_STALL: Stuck RUNNABLE for {time.time() - runnable_since:.0f}s "
                f"(age {age_s:.0f}s) on job {job_id}.",
                flush=True,
            )
            return 4

        if status in ("SUCCEEDED", "FAILED"):
            if ls:
                time.sleep(2)
                cw_token, msgs = _tail_cloudwatch(logs_cli, log_group, ls, cw_token)
                for msg in msgs:
                    print(f"  | {msg}", flush=True)
            wait_s = (j.get("startedAt", j["createdAt"]) - j["createdAt"]) / 1000.0
            run_s = (
                (j.get("stoppedAt", 0) - j.get("startedAt", 0)) / 1000.0
                if j.get("startedAt") and j.get("stoppedAt")
                else 0.0
            )
            print(
                f"FINAL status={status} wait_s={wait_s:.2f}s container_s={run_s:.2f}s "
                f"attempts={len(j.get('attempts', []))} summary={summary} reason={reason}",
                flush=True,
            )
            return 0 if status == "SUCCEEDED" else 2

        time.sleep(poll_s)


def verify_s3_outputs(s3, bucket, output_prefix, expected_keys, no_ibra, min_output_bytes):
    """Verify expected output GeoTIFF objects exist on S3 with non-trivial byte sizes."""
    prefix = output_prefix.rstrip("/") + "/"
    found = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            found[obj["Key"]] = int(obj["Size"])

    missing = []
    undersized = []
    for in_key in expected_keys:
        out_name = seasonalDRCMIbra.derive_output_name(in_key, no_ibra=no_ibra)
        out_key = f"{prefix}{out_name}"
        sz = found.get(out_key)
        if sz is None:
            missing.append(out_key)
        elif sz < min_output_bytes:
            undersized.append((out_key, sz))

    print(
        f"S3 verification under s3://{bucket}/{prefix}: expected={len(expected_keys)}, "
        f"missing={len(missing)}, undersized={len(undersized)}",
        flush=True,
    )
    if missing or undersized:
        for k in missing[:10]:
            print(f"  MISSING: s3://{bucket}/{k}", flush=True)
        for k, sz in undersized[:10]:
            print(f"  UNDERSIZED: s3://{bucket}/{k} ({sz} bytes < {min_output_bytes})", flush=True)
        return False
    return True


def parse_args():
    p = argparse.ArgumentParser(
        description="Submit and optionally monitor Cibo Labs Seasonal DRCM jobs on AWS Batch."
    )
    p.add_argument("--env-file", default=".env", help="Optional .env file with AWS credentials/region (default: .env)")
    p.add_argument("--region", default=DEFAULT_REGION, help=f"AWS region (default: {DEFAULT_REGION})")
    p.add_argument("--bucket", default=DEFAULT_BUCKET, help=f"S3 bucket (default: {DEFAULT_BUCKET})")
    p.add_argument("--gc-prefix", default=DEFAULT_GC_PREFIX, help=f"Input S3 prefix (default: {DEFAULT_GC_PREFIX})")
    p.add_argument("--pattern", default=DEFAULT_SEASON_REGEX, help="Regex matching input S3 keys")
    p.add_argument("--start-ym", default=None, help="Inclusive start YYYYMM filter (e.g. 200912)")
    p.add_argument("--end-ym", default=None, help="Inclusive end YYYYMM filter (e.g. 202608)")
    p.add_argument("--seasons", nargs="+", default=None, help="Explicit S3 keys or s3:// URIs instead of prefix scan")
    p.add_argument("--skip-existing", action="store_true", help="Pre-skip seasons already present in S3 output prefix")
    p.add_argument("--ref-key", default=DEFAULT_REF_KEY, help=f"Reference raster S3 key (default: {DEFAULT_REF_KEY})")
    p.add_argument("--ibra-key", default=None, help="Optional separate IBRA raster S3 key")
    p.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX, help=f"Output S3 prefix (default: {DEFAULT_OUTPUT_PREFIX})")
    p.add_argument("--staging-prefix", default=DEFAULT_STAGING_PREFIX, help=f"S3 script staging prefix (default: {DEFAULT_STAGING_PREFIX})")
    p.add_argument("--run-tag", default=None, help="Staging subfolder & job tag (default: UTC timestamp)")

    p.add_argument("--shards", type=int, default=15, help="Number of parallel Batch array shards (0 = 1 per season, default: 15)")
    p.add_argument("--window", type=int, default=1501, help="Comparison window in pixels (default: 1501)")
    p.add_argument("--chunksize", type=int, default=16384, help="Rows per processing chunk (default: 16384)")
    p.add_argument("--minvalpercent", type=float, default=0.10, help="Minimum valid pixel fraction (default: 0.10)")
    p.add_argument("--no-ibra", action="store_true", default=True, help="Use unconstrained global histogram (default)")
    p.add_argument("--ibra", dest="no_ibra", action="store_false", help="Enable IBRA regional constraints")
    p.add_argument("--no-overviews", action="store_true", default=False, help="Skip internal GDAL overview pyramids")
    p.add_argument("--geojson", default=None, help="Optional AOI GeoJSON path or s3:// URI")
    p.add_argument("--bbox", nargs=4, type=float, default=None, metavar=("MINX", "MINY", "MAXX", "MAXY"))
    p.add_argument(
        "--min-output-bytes",
        type=int,
        default=None,
        help="Minimum valid output size in bytes (default: 100000000 full-res, 1024 when --bbox/--geojson is active)",
    )

    p.add_argument("--job-name", default=None, help="AWS Batch job name (default: drcm-aus-<run_tag>)")
    p.add_argument("--job-queue", default=DEFAULT_JOB_QUEUE, help=f"AWS Batch job queue (default: {DEFAULT_JOB_QUEUE})")
    p.add_argument("--job-definition", default=DEFAULT_JOB_DEFINITION, help=f"Job definition name[:rev] (default: {DEFAULT_JOB_DEFINITION})")
    p.add_argument("--ensure-job-def", action="store_true", help="Register a new job definition revision with current settings")
    p.add_argument("--image", default=DEFAULT_ECR_IMAGE, help=f"Container ECR image when registering JD (default: {DEFAULT_ECR_IMAGE})")
    p.add_argument("--vcpus", type=int, default=64, help="Container VCPUs (default: 64)")
    p.add_argument("--memory", type=int, default=100000, help="Container memory in MiB (default: 100000)")
    p.add_argument("--attempt-duration", type=int, default=5400, help="Batch attemptDurationSeconds timeout (default: 5400)")
    p.add_argument("--log-group", default=DEFAULT_LOG_GROUP, help=f"CloudWatch log group (default: {DEFAULT_LOG_GROUP})")
    p.add_argument("--log-stream-prefix", default=DEFAULT_LOG_STREAM_PREFIX, help="CloudWatch log stream prefix")
    p.add_argument("--job-role-arn", default=None, help="Optional IAM jobRoleArn when registering JD")
    p.add_argument("--exec-role-arn", default=None, help="Optional IAM executionRoleArn when registering JD")

    p.add_argument("--retry-manifest", default=None, help="Existing manifest s3:// URI or local path for --retry-shard")
    p.add_argument("--retry-shard", type=int, default=None, help="Resubmit a single shard index cleanly using --shard-index")
    p.add_argument("--smoke-test", action="store_true", help="Run test_ibra_drcm.py on AWS Batch and wait before main submission")
    p.add_argument("--wait", action="store_true", help="Monitor submitted job until completion and verify S3 outputs")
    p.add_argument("--watch-job-id", default=None, help="Monitor an existing AWS Batch job ID without submitting")
    p.add_argument("--poll-s", type=int, default=30, help="Batch status poll interval in seconds (default: 30)")
    p.add_argument("--max-runnable-s", type=int, default=720, help="Fail watcher if stuck RUNNABLE beyond N seconds (default: 720)")
    p.add_argument("--timeout-s", type=int, default=6000, help="Watcher wall timeout in seconds (default: 6000)")
    p.add_argument("--dry-run", action="store_true", help="Discover seasons, build manifest, and print submission plan without AWS writes")
    return p.parse_args()


def main():
    args = parse_args()
    load_env_file(args.env_file)
    repo_dir = Path(__file__).resolve().parent

    cfg = Config(
        region_name=args.region,
        connect_timeout=10,
        read_timeout=30,
        retries={"max_attempts": 8, "mode": "adaptive"},
    )
    batch = boto3.client("batch", config=cfg)
    logs_cli = boto3.client("logs", config=cfg)
    s3 = boto3.client("s3", config=cfg)

    if args.watch_job_id:
        rc = watch_batch_job(
            batch,
            logs_cli,
            args.watch_job_id,
            log_group=args.log_group,
            poll_s=args.poll_s,
            max_runnable_s=args.max_runnable_s,
            timeout_s=args.timeout_s,
        )
        sys.exit(rc)

    run_tag = args.run_tag or datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    job_name = args.job_name or f"drcm-aus-{run_tag}".replace("_", "-")
    env_overrides = build_batch_environment(region=args.region)

    if args.retry_shard is not None:
        if not args.retry_manifest:
            raise ValueError("--retry-shard requires --retry-manifest pointing to the staged manifest.")
        if args.retry_manifest.startswith("s3://"):
            mbucket, mkey = seasonalDRCMIbra._parse_s3_uri(args.retry_manifest)
            manifest = json.loads(s3.get_object(Bucket=mbucket, Key=mkey)["Body"].read().decode("utf-8"))
            staging_uri = f"s3://{mbucket}/{os.path.dirname(mkey)}"
        else:
            manifest = json.loads(Path(args.retry_manifest).read_text(encoding="utf-8"))
            staging_uri = f"s3://{args.bucket}/{args.staging_prefix.strip('/')}/{run_tag}"
            if not args.dry_run:
                staging_uri = stage_scripts_and_manifest(
                    s3, repo_dir, args.bucket, f"{args.staging_prefix.strip('/')}/{run_tag}", manifest, args.geojson
                )

        eff_retry_min_bytes = (
            args.min_output_bytes
            if args.min_output_bytes is not None
            else (
                DEFAULT_AOI_MIN_OUTPUT_BYTES
                if (
                    args.bbox is not None
                    or args.geojson is not None
                    or manifest.get("bbox") is not None
                    or manifest.get("geojson") is not None
                )
                else DEFAULT_MIN_OUTPUT_BYTES
            )
        )
        cmd = build_worker_command(
            staging_uri,
            work_subdir=run_tag,
            window=args.window,
            chunksize=args.chunksize,
            minvalpercent=args.minvalpercent,
            no_ibra=args.no_ibra,
            no_overviews=args.no_overviews,
            min_output_bytes=eff_retry_min_bytes,
            explicit_shard_index=args.retry_shard,
        )
        if args.dry_run:
            print(json.dumps({"jobName": f"{job_name}-shard-{args.retry_shard}", "command": cmd}, indent=2))
            return

        validate_queue_and_compute_env(batch, args.job_queue, args.vcpus, args.memory)
        jd_ref = ensure_job_definition(
            batch, args.job_definition, args.image, args.vcpus, args.memory,
            args.attempt_duration, args.region, args.log_group, args.log_stream_prefix,
            force_register=args.ensure_job_def, job_role_arn=args.job_role_arn, exec_role_arn=args.exec_role_arn,
        )
        sub = batch.submit_job(
            jobName=f"{job_name}-shard-{args.retry_shard}",
            jobQueue=args.job_queue,
            jobDefinition=jd_ref,
            containerOverrides={"command": cmd, "environment": env_overrides},
        )
        job_id = sub["jobId"]
        print(f"SUBMITTED_RETRY_JOB_ID: {job_id} (shard={args.retry_shard})", flush=True)
        if args.wait:
            rc = watch_batch_job(
                batch, logs_cli, job_id,
                log_group=args.log_group, poll_s=args.poll_s,
                max_runnable_s=args.max_runnable_s, timeout_s=args.timeout_s,
            )
            sys.exit(rc)
        return

    # 1. Discover seasons
    if args.seasons:
        items = []
        for s in args.seasons:
            if s.startswith("s3://"):
                sb, sk = seasonalDRCMIbra._parse_s3_uri(s)
                if sb != args.bucket:
                    raise ValueError(f"Season URI bucket {sb} differs from --bucket {args.bucket}")
                key = sk
            else:
                key = s
            sz = int(s3.head_object(Bucket=args.bucket, Key=key)["ContentLength"])
            items.append((key, sz))
        items.sort(key=lambda x: x[0])
    else:
        items = discover_s3_seasons(
            s3, args.bucket, args.gc_prefix, args.pattern, start_ym=args.start_ym, end_ym=args.end_ym
        )

    eff_min_output_bytes = (
        args.min_output_bytes
        if args.min_output_bytes is not None
        else (
            DEFAULT_AOI_MIN_OUTPUT_BYTES
            if (args.bbox is not None or args.geojson is not None)
            else DEFAULT_MIN_OUTPUT_BYTES
        )
    )

    if args.skip_existing and items:
        items, skipped = filter_existing_outputs(
            s3, items, args.bucket, args.output_prefix, args.no_ibra, eff_min_output_bytes
        )
        print(f"Pre-filtered existing outputs: {skipped} skipped, {len(items)} remaining", flush=True)

    if not items:
        print("No pending seasons found to submit.", flush=True)
        return

    manifest_geojson = args.geojson
    if args.geojson and not str(args.geojson).startswith(("s3://", "/vsis3/")):
        manifest_geojson = Path(args.geojson).name

    manifest, shard_bytes = build_shard_manifest(
        items,
        bucket=args.bucket,
        ref_key=args.ref_key,
        output_prefix=args.output_prefix,
        num_shards=args.shards,
        geojson=manifest_geojson,
        bbox=args.bbox,
        ibra_key=args.ibra_key,
    )

    num_shards = manifest["num_shards"]
    print(f"Discovered {len(items)} seasons -> {num_shards} shards:", flush=True)
    for idx, shard in enumerate(manifest["shards"]):
        gib = shard_bytes[idx] / (1024 ** 3)
        print(f"  Shard {idx:2d}: n={len(shard):2d} seasons, total_in={gib:.2f} GiB ({shard[0]} .. {shard[-1]})", flush=True)

    staging_key_prefix = f"{args.staging_prefix.strip('/')}/{run_tag}"
    staging_uri = f"s3://{args.bucket}/{staging_key_prefix}"
    explicit_idx = 0 if num_shards == 1 else None
    cmd = build_worker_command(
        staging_uri,
        work_subdir=run_tag,
        window=args.window,
        chunksize=args.chunksize,
        minvalpercent=args.minvalpercent,
        no_ibra=args.no_ibra,
        no_overviews=args.no_overviews,
        min_output_bytes=eff_min_output_bytes,
        explicit_shard_index=explicit_idx,
    )

    if args.dry_run:
        plan = {
            "jobName": job_name,
            "jobQueue": args.job_queue,
            "jobDefinition": args.job_definition,
            "arraySize": num_shards if num_shards > 1 else 1,
            "stagingUri": staging_uri,
            "manifestSummary": {k: v for k, v in manifest.items() if k != "shards"},
            "command": cmd,
            "environmentCount": len(env_overrides),
        }
        print("\nDRY_RUN_PLAN:")
        print(json.dumps(plan, indent=2))
        return

    # 2. Validate queue/CE, ensure job definition, and stage scripts + manifest to S3
    validate_queue_and_compute_env(batch, args.job_queue, args.vcpus, args.memory)
    jd_ref = ensure_job_definition(
        batch,
        args.job_definition,
        args.image,
        args.vcpus,
        args.memory,
        args.attempt_duration,
        args.region,
        args.log_group,
        args.log_stream_prefix,
        force_register=args.ensure_job_def,
        job_role_arn=args.job_role_arn,
        exec_role_arn=args.exec_role_arn,
    )
    stage_scripts_and_manifest(
        s3,
        repo_dir,
        args.bucket,
        staging_key_prefix,
        manifest,
        args.geojson,
        include_test_data=True,
        require_test_data=args.smoke_test,
    )

    # 3. Optional pre-flight smoke test before submitting main array
    if args.smoke_test:
        smoke_cmd = [
            "bash", "-lc",
            "\n".join([
                "set -e",
                f"mkdir -p /tmp/{run_tag}_smoke && cd /tmp/{run_tag}_smoke",
                f"aws s3 sync {staging_uri}/ . --quiet",
                "python3 -u run_aus_batch_worker.py --help",
                "python3 -u test_ibra_drcm.py",
            ]),
        ]
        smoke_sub = batch.submit_job(
            jobName=f"{job_name}-smoke",
            jobQueue=args.job_queue,
            jobDefinition=jd_ref,
            containerOverrides={
                "command": smoke_cmd,
                "environment": env_overrides,
                "resourceRequirements": [
                    {"type": "VCPU", "value": "8"},
                    {"type": "MEMORY", "value": "14000"},
                ],
            },
        )
        smoke_id = smoke_sub["jobId"]
        print(f"SUBMITTED_SMOKE_JOB_ID: {smoke_id}", flush=True)
        smoke_rc = watch_batch_job(
            batch, logs_cli, smoke_id,
            log_group=args.log_group, poll_s=20, max_runnable_s=480, timeout_s=900,
        )
        if smoke_rc != 0:
            raise RuntimeError(f"Pre-flight smoke test failed with exit code {smoke_rc}")

    # 4. Submit main job (array if >1 shard, standalone with explicit --shard-index 0 if 1 shard)
    submit_kw = {
        "jobName": job_name,
        "jobQueue": args.job_queue,
        "jobDefinition": jd_ref,
        "containerOverrides": {"command": cmd, "environment": env_overrides},
    }
    if num_shards > 1:
        submit_kw["arrayProperties"] = {"size": num_shards}

    sub = batch.submit_job(**submit_kw)
    job_id = sub["jobId"]
    desc = batch.describe_jobs(jobs=[job_id])["jobs"][0]
    arr_sz = desc.get("arrayProperties", {}).get("size", 1)
    print(f"SUBMITTED_JOB_ID: {job_id} (name={job_name}, queue={args.job_queue}, jd={jd_ref}, arraySize={arr_sz}, status={desc['status']})", flush=True)

    if args.wait:
        rc = watch_batch_job(
            batch,
            logs_cli,
            job_id,
            log_group=args.log_group,
            poll_s=args.poll_s,
            max_runnable_s=args.max_runnable_s,
            timeout_s=args.timeout_s,
        )
        if rc != 0:
            sys.exit(rc)
        ok = verify_s3_outputs(
            s3,
            args.bucket,
            args.output_prefix,
            [k for k, _ in items],
            args.no_ibra,
            eff_min_output_bytes,
        )
        sys.exit(0 if ok else 5)


if __name__ == "__main__":
    main()
