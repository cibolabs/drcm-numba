# Dynamic Reference Cover Method (DRCM) Pipeline — Cibo Labs Production Edition

This repository contains Cibo Labs' high-performance Python, Numba, and AWS Batch (Graviton `arm64`) implementation of the Australian seasonal Groundcover Reference Pixel Filter and Dynamic Reference-Cover Method (DRCM) pipeline, built on the foundational methodology of **Bastin et al. (2012)** ([DOI: 10.1016/j.rse.2012.02.021](https://doi.org/10.1016/j.rse.2012.02.021)).

The software modernizes the original Joint Remote Sensing Research Program (JRSRP) codebase ([`jrsrp/themes/cover/groundcover/drcm`](https://gitlab.com/jrsrp/themes/cover/groundcover/drcm)), replacing the legacy Fortran 90 module (`refpixelfilter.f90`) and single-node drivers with pure-Python Numba JIT kernels, direct parallel S3/local TIFF tile readers, halo-padded AOI subsetting, and a resilient multi-season AWS Batch Spot processing workflow capable of computing continental Australia (`141,481 × 135,159` pixels, `30 m` Australian Albers) in **~2–3 minutes per season**.

---

## 1. Scientific Foundation: The Dynamic Reference-Cover Method

The Dynamic Reference-Cover Method (DRCM) was established by **Bastin et al. (2012)** to resolve the fundamental challenge of separating grazing management effects from climate variability across rangelands at regional to continental scales.

### A. The Rangeland Monitoring Challenge
- **Climate vs. Management Confounding**: In arid and semi-arid rangelands, high inter-annual rainfall variability ("boom and bust" cycles) frequently masks or confounds management-driven changes in ground cover (grazing pressure, stocking rates, spelling).
- **Scarcity of Physical Benchmarks**: Establishing static, ungrazed reference exclosures across millions of square kilometers is logistically and economically unfeasible. Furthermore, fixed benchmarks cannot capture local environmental gradients or long-term climate dynamics.
- **Absence of Management Spatial Data**: Regional monitoring cannot rely on ground-based reference sites or property-level management GIS records, as these are rarely available consistently at landscape scale.

### B. Conceptual Framework and Methodology
DRCM establishes a dynamic, image-derived reference framework using multitemporal satellite ground cover sequences (derived from Landsat TM/ETM+/OLI and operational DEA/JRSRP fractional cover):
1. **Temporal Minimum Ground Cover as a Persistence Indicator**:
   - The method computes a temporal minimum ground cover image (or bare ground maximum) across multi-decadal imagery to identify locations that retain ground cover during years of lowest rainfall (climatic stress).
   - Areas maintaining high cover under severe rainfall deficits serve as persistent, resilient landscape benchmarks under sustainable grazing or conservative utilization.
2. **Moving Window Local Neighbourhood**:
   - Operates a large moving spatial window (operational default: `1501 × 1501` pixels / ~45 km × 45 km, ~2,000 km²) to define a local reference population subject to equivalent macro-climatic conditions and broad geomorphology.
3. **Cover-Index Percentile Benchmarks**:
   - Within each moving window, reference pixels are selected from the upper percentiles (typically the `90th` to `95th` percentiles) of the temporal minimum ground cover distribution.
4. **Difference Ground Cover (DGC)**:
   - Evaluates the difference between the central pixel's ground cover and the mean of surrounding reference pixels:

     ```math
     \text{DGC} = \text{Groundcover}_{\text{pixel}} - \text{Groundcover}_{\text{reference}}
     ```

   - This difference standardizes ground cover against the local landscape's climatic potential, isolating management-induced cover deficit or surplus from seasonal rainfall fluctuations.

### C. Multi-Scale Empirical Validation (Bastin et al., 2012)
The original paper validated the method across three operational scales in Queensland tropical savanna woodlands:
1. **Grazing Trial Scale (Wambiana)**: In small paddocks under controlled stocking, heavy stocking resulted in substantial ground cover decline that DRCM successfully detected and distinguished from rainfall effects.
2. **Commercial Paddock Scale**: In commercial paddocks, wet-season spelling (destocking during critical growth periods) produced measurable, positive ground cover recovery detected by DRCM.
3. **Regional / Property Scale (`1.3 × 10⁶ km²`)**: Across commercial beef properties with different initial land condition ratings across Queensland's rangelands, DRCM showed broad agreement with ground-based Grazing Land Condition (GLC) assessments. Disagreements occurred primarily where changing land condition reflected shifts in pasture botanical composition (e.g. loss of perennial 3P tussock grasses) rather than total fractional cover alone.

### D. Primary Reference and Citation

> **Bastin, G., Scarth, P., Chewings, V., Sparrow, A., Denham, R., Schmidt, M., O'Reagain, P., Shepherd, R., & Abbott, B. (2012).**  
> *Separating grazing and rainfall effects at regional scale using remote sensing imagery: A dynamic reference-cover method.*  
> **Remote Sensing of Environment**, 121, 443–457.  
> DOI: [10.1016/j.rse.2012.02.021](https://doi.org/10.1016/j.rse.2012.02.021)

```bibtex
@article{bastin2012separating,
  title={Separating grazing and rainfall effects at regional scale using remote sensing imagery: A dynamic reference-cover method},
  author={Bastin, G. and Scarth, P. and Chewings, V. and Sparrow, A. and Denham, R. and Schmidt, M. and O'Reagain, P. and Shepherd, R. and Abbott, B.},
  journal={Remote Sensing of Environment},
  volume={121},
  pages={443--457},
  year={2012},
  publisher={Elsevier},
  doi={10.1016/j.rse.2012.02.021}
}
```

- **Original JRSRP GitLab Repository**: [https://gitlab.com/jrsrp/themes/cover/groundcover/drcm](https://gitlab.com/jrsrp/themes/cover/groundcover/drcm)

---

## 2. Cibo Labs Data Lake & System Architecture

### A. S3 Data Lake Conventions (`us-west-2`, Bucket `cibo-dat-oregon`)
Cibo Labs operates seasonal DRCM across the `141,481 × 135,159` continental Australian grid (`EPSG:3577`, 30 m resolution, `512 × 512` DEFLATE Cloud-Optimized/Tiled GeoTIFFs) via an S3 Gateway VPC Endpoint (`com.amazonaws.us-west-2.s3`) with zero S3 data transfer fees:

| Artifact | S3 URI / Pattern | Notes |
| :--- | :--- | :--- |
| **Seasonal Fractional Cover** | `s3://cibo-dat-oregon/GC/lztmre_aus_mYYYYMMYYYYMM_dp8a2[_vinterim].tif` | `147` seasons (`198912`–`202608`), avg `~13.13 GiB` / season |
| **Masked Reference + IBRA** | `s3://cibo-dat-oregon/DRCM/lztmre_aus_m198912202012_dixa2_change_slope_ibra.tif` | `3,801.0 MB` (`3.71 GiB`), Band 1: Reference stats, Band 2: IBRA 7.0 code (`1..89`) |
| **Continental DRCM Outputs** | `s3://cibo-dat-oregon/DRCM/aus/lztmre_aus_mYYYYMMYYYYMM_doda2[_ibra][_vinterim].tif` | Default `--no-ibra` output (`doda2`), `~2.0–4.8 GB` per season with internal overviews |
| **Regional AOI / Legacy Saves** | `s3://cibo-dat-oregon/DRCM/nsw_field/` or `s3://cibo-dat-oregon/DRCM/` | `--save` uploads local outputs to `s3://cibo-dat-oregon/DRCM/` and cleans up local files |
| **Staged Batch Runners** | `s3://cibo-dat-oregon/DRCM/_batch_scripts/<run_tag>/` | Versioned scripts + shard manifest (`aus_seasons_manifest.json`) synced at container start |

### B. End-to-End Operational Workflow

```text
[NCAS Woody Extent]      [Terrain Slope]       [1989-2020 Bare Stats]      [IBRA 7.0 Vector]
woodyMaxMinCleared.tif  slope_relief_3s.tif   dixa2_bareStats.tif (Band 7)      ibra.fgb
         │                      │                        │                         │
         └──────────────────────┼────────────────────────┴─────────────────────────┘
                                │
                                ▼
         ┌──────────────────────────────────────────────────────────┐
         │ Stage 1: maskChangeSteepIbra.py (RIOS)                   │
         │ - Excludes NCAS Forest (Woody Max == 2)                  │
         │ - Excludes Historical Clearing (Cleared == 1)            │
         │ - Excludes Steep Terrain (Slope relief >= 39 / >= 10%)   │
         │ - Burns IBRA Bioregions (OBJECTID 1..89 into Band 2)     │
         └──────────────────────────┬───────────────────────────────┘
                                    │
                                    ▼
  Reference GeoTIFF: s3://cibo-dat-oregon/DRCM/lztmre_aus_m198912202012_dixa2_change_slope_ibra.tif
  Band 1: Masked Baseline Groundcover (1..101, 0=NoData)
  Band 2: IBRA Biogeographic Region ID (1..89, 0=NoData)
                                    │
    ┌───────────────────────────────┴───────────────────────────────┐
    │                                                               │
    ▼ (AWS Batch Array / submit_batch_jobs.py)                      ▼ (FastS3 / FastLocal Tiled I/O)
[run_aus_batch_worker.py (Shards 0..N-1)]            [Stage 2 Compute Kernels (v5 Serpentine)]
 - Caches 3.8 GB reference COG once onto /tmp NVMe   - Default (--no-ibra): refpix_numba.py
 - Skips existing valid S3 outputs                     Serpentine 2D tiled (16384x512) uint16 histograms
 - Streams seasonal inputs via FastS3                  + 256x256 ocean block & row-range trimming
 - Overlaps S3 upload of season i with season i+1    - Stratified (--ibra): refpix_numba_ibra.py
                                                       Serpentine 2D tiled (1..255 regions), k=1 fast path
                                                       + k>=2 tight-bbox passes + 2-bit 256x256 skipping
    └───────────────────────────────┬───────────────────────────────┘
                                    │
                                    ▼
         ┌──────────────────────────────────────────────────────────┐
         │ Stage 3: seasonalDRCMIbra.py                             │
         │ - Auto-detects dp8a2 (0..100%) vs dixa2 (100..200)       │
         │ - Concurrent multi-band read_all_windows (coord + band)  │
         │ - Overlap halo reuse + next-chunk background prefetch    │
         │ - Halo-padded --bbox / --geojson + native c_start/c_stop │
         │ - Internal DEFLATE overviews (4, 8, 16, 32) + S3 upload  │
         └──────────────────────────┬───────────────────────────────┘
                                    │
                                    ▼
  Output GeoTIFF: lztmre_aus_mYYYYMMYYYYMM_doda2[_ibra][_vinterim].tif
  Band 1: Difference Ground Cover (DGC = 100 + Groundcover - Refpixel, 0=NoData)
  Band 2: Reference Ground Cover (100 + Refpixel, 0=NoData)
```

### C. Output Product Specification (`GTiff`)
- **Compression & Tiling**: `COMPRESS=DEFLATE`, `ZLEVEL=1`, `PREDICTOR=2`, `TILED=YES`, `INTERLEAVE=BAND`, `BLOCKXSIZE=512`, `BLOCKYSIZE=512`, `BIGTIFF=YES`, `NUM_THREADS={effective_cpus}`.
- **Internal Overviews**: `AVERAGE` resampling at levels `4 8 16 32` built with `COMPRESS_OVERVIEW=DEFLATE`, `PREDICTOR_OVERVIEW=1`, `GDAL_NUM_THREADS={effective_cpus}`.
- **Band 1 — Difference Ground Cover (`DGC`)**:
  - Scaled as `100 + (Groundcover - Refpixel)` (`uint8`, `NoData = 0`)
  - `100` = Zero Difference (matches local reference); `> 100` = Surplus Cover; `< 100` = Cover Deficit.
- **Band 2 — Reference Ground Cover**:
  - Scaled as `100 + Refpixel` (`100 = 0%`, `200 = 100%`, `NoData = 0`).

### D. Technical Rationale: Masking & IBRA Stratification vs. Global Unconstrained Mode
- **NCAS Woody Extent (`inputs.changeData[0] != 2`)**: Excludes forest canopy (`Class 2`) where canopy shadow and scattering confound understorey groundcover unmixing.
- **Historical Clearing (`inputs.changeData[2] != 1`)**: Excludes cleared tracts so disturbed sites do not depress reference benchmarks.
- **Terrain Slope Masking (`inputs.terrainData[0] < 39`)**: Retains plains, rises, and low hills (`< 10%` slope per Speight 2009 Table 5) while masking rolling/steep terrain (`>= 39`).
- **IBRA Stratification vs. `--no-ibra` Operational Mode**:
  - Stratified mode (`--ibra`, default in `seasonalDRCMIbra.py`) constrains candidate reference pixels to the matching `uint8` IBRA 7.0 region (`1..255`, lifted from the legacy `< 90` cap) to avoid cross-bioregion contamination across sharp landscape boundaries (e.g., floodplains next to dunefields). With the `v5` Serpentine 2D tiled upgrade (`refpix_numba_ibra.py`), `--ibra` achieves **38×–77× faster** execution than the legacy per-row kernel and reaches virtual compute parity with `--no-ibra` (`10.76M px/s` vs `10.82M px/s` on a `16.0M px` continental-width chunk at 4 vCPUs).
  - Unconstrained global mode (`--no-ibra`, operational default in `submit_batch_jobs.py` / `run_aus_batch_worker.py`) treats all unmasked pixels inside the moving window as a single pool (`refpix_numba.py`), upgraded to the `strip_h=16384` Serpentine kernel (**1.38×–1.45× faster** than `v4`).

---

## 3. High-Performance Kernel & I/O Optimizations (`v1` → `v5`)

Cibo Labs upgraded the pipeline across five optimization generations while maintaining bit-for-bit agreement with the legacy baseline (`0` pixel mismatches verified across NSW AOI, continental Australia, and the independent histogram-free test oracle):

1. **2D Tiled Column Histograms & `uint16` Quantization (`v3`–`v4`, `refpix_numba.py`)**:
   - Decomposes the sliding window into `col_tile_w=512` 2D tiles with column-major pre-aggregated histograms (`col_perc`, `col_dryhist`, `col_drysum`), slashing inner-loop memory traffic.
   - Automatically selects `uint16` histogram accumulators (`_KERNEL_U16` / `_IBRA_KERNEL_U16`) whenever `window <= 65535` (halving L2 cache footprint relative to `int32`), with automatic `int32` fallback (`_KERNEL_I32` / `_IBRA_KERNEL_I32`) for extreme windows (`window > 65535`).
   - Pads histogram width to `104` bins for SIMD/cache-line alignment and uses branchless `_bounds_fast` with tracked non-zero totals (`perc_tot`).
2. **The Serpentine Kernel & Adaptive `strip_h=16384` Tiling (`v5`, `refpix_numba.py` & `refpix_numba_ibra.py`)**:
   - **Why standard row-by-row tiling stalled**: In earlier 2D tiled kernels (`v4`), every row inside a tile swept left-to-right (`c0 → c1 - 1`). Stepping down from `(r, c1 - 1)` back to `(r + 1, c0)` jumped `~512` pixels back to the left edge, forcing the kernel to **zero out the active 104-bin histograms and re-sum all `window` (`1,501`) vertical column histograms from scratch on every single row**.
   - **How the Serpentine (boustrophedon) sweep works**:
     ```
     Row r   (even):  c0 ────────────────────────► c1-1   (slide window right)
                                                     │     turnaround: only drop rem_r & add add_r
     Row r+1 (odd):   c0 ◄──────────────────────── c1-1   (slide window left, zero rebuild!)
     ```
     Alternating left-to-right on even rows and right-to-left on odd rows keeps the `1,501`-column active histogram parked at the exact starting column for the next row. Vertical transitions only update one exiting row (`r - 1 - halfwindow`) and one entering row (`r + halfwindow`), completely eliminating per-row `1,501`-column histogram resets and re-summation.
   - **Adaptive Strip Height (`strip_h=16384`)**: Raises the default strip height from `256` to `16384` so an entire operational chunk (`16,384` rows) runs as a single continuous Serpentine sweep per `512`-column tile—amortizing top-of-tile `window`-row setup by `64×`. On smaller grids or thin AOI slices, `eff_strip_h` automatically halves down toward `128` while `n_row_strips × n_col_tiles < 256` so CPU threads remain saturated.
   - **Unified Multi-IBRA Serpentine Engine (`1..255` Regions)**: Replaces the legacy per-row `90 × 103` `int32` `--ibra` kernel (previously limited to region IDs `< 90`) with the full Serpentine 2D tiled `uint16`/`int32` engine:
     * **2-bit `256 × 256` block mask (`_build_ibra_nonzero_blocks`)**: Bit `0` tracks `ibra > 0`; bit `1` tracks valid paired `(cover, ground)` pixels (`0 < pv <= 101` & `ground > 0`). `_tile_active_core_only` and `_tile_has_valid_halo` skip pure-ocean and halo-empty tiles without touching histograms.
     * **Two-tier region specialization (`_scan_core_regions_2d`)**: Supports all `1..255` `uint8` IBRA bioregions. Interior single-region tiles (`k == 1`) execute a fast zero-branch Serpentine sweep (`only_write_matching=False`), while bioregion boundary tiles (`k >= 2`) run Serpentine passes restricted strictly to each active region's tight 2D bounding box `(reg_rmin..reg_rmax, reg_cmin..reg_cmax)`.
     * **Native Halo Slicing**: Accepts `r_start`, `r_stop`, `c_start`, `c_stop` directly in both `--no-ibra` and `--ibra` kernels, eliminating post-compute Python array slicing on padded AOIs.
     * **Leading & Trailing Ocean Row Trimming**: In `--no-ibra` mode, `_tile_valid_row_range` trims both top and bottom all-ocean segments per column tile via `_fill_empty_tile` before entering the Serpentine loop.
3. **Coarse `256 × 256` Empty-Block Ocean Skipping (`_build_valid_blocks` / `_build_ibra_nonzero_blocks`)**:
   - Over half of the continental `141,481 × 135,159` bounding box is ocean or masked NoData. Coarse `256 × 256` masks skip tiles with zero valid pixels inside their halo neighborhood while preserving exact edge/interior semantics.
4. **Direct Parallel Tile Readers (`FastS3` & `FastLocal`) + Concurrent `read_all_windows` (`v5`)**:
   - Inspects GeoTIFF TIFF IFD `TileOffsets` and `TileByteCounts` directly (`TIFF`/`BigTIFF` Classic & 64-bit) and verifies `COMPRESS=DEFLATE` + `PREDICTOR=1` on tile `(0, 0)` against GDAL before activating (automatically falling back to GDAL for `PREDICTOR=2` or unsupported layouts, as verified in Test 9).
   - **`FastS3`** (`boto3` pooled byte-range `GetObject`s, `max_pool_connections=64`) and **`FastLocal`** (`os.pread` + `zlib.decompress`) eliminate GDAL `/vsis3/` lock contention.
   - **Concurrent Multi-Band Tail Fetches (`read_all_windows`, `v5`)**: Replaces sequential `read_in_window()` → `read_ref_windows()` calls with a `2–3` thread `coord_pool` feeding dedicated per-band worker pools (`band_pools["in"]`, `"ref"`, `"ibra"`), fetching all input/reference/IBRA bands in parallel while keeping shared-dataset GDAL fallback reads serialized safely (`can_split_ref_ibra`).
5. **Chunk Overlap Reuse, Prefetching, Numba Encoding, & Container Safeguards**:
   - Operational `--chunksize 16384` reuses the overlapping `2 × halfwindow` border rows from chunk `k` inside chunk `k+1` and overlaps background prefetch (`ThreadPoolExecutor(max_workers=1)`) with Numba compute.
   - Uses a Numba LUT (`convert_to_groundcover(..., inplace=True)`), parallel 2D encoder (`_encode_outputs_2d`), Linux cgroup v1/v2 quota detection (`_get_effective_cpus()`), halo-padded `--bbox`/`--geojson` alignment, and direct S3 `-o` / `--save` streaming.

---

## 4. Cibo Labs AWS Batch Graviton Infrastructure & Engineering Safeguards

### A. Production AWS Batch Topology (`us-west-2`)
- **Job Queues & Spot Compute Environments**:
  - `CiboBatchProcessingJobQueueESANonProd` → `ausmedian_batch_nonprod_compute_environment` (default in `submit_batch_jobs.py`)
  - `CiboBatchProcessingJobQueueESA` → `ausmedian_batch_compute_environment`
  - `cibo-processor-batch-test-job-queue` → `cibo-processor-batch-test-compute-environment` (`maxvCpus=64`, smoke tests / AOI tests)
  - Graviton Spot families (`SPOT_CAPACITY_OPTIMIZED`, `maxvCpus=2048` production): `m7g`, `m6g`, `r7g`, `r6g`, `c7g`, `c6g` and `*d` local-NVMe variants.
- **Job Definition (`drcm-aus-multi-season`)**:
  - Image: `331561773057.dkr.ecr.us-west-2.amazonaws.com/cibobase:baseubuntu_2404_numba`
  - Container resources: `64 VCPU`, `100,000 MiB` memory (`c7g.16xlarge` = 128 GiB host RAM), `attemptDurationSeconds=5400`, CloudWatch Log Group `/aws/batch/cibo-processor-batch-test`.
  - Retry strategy (`attempts=4`): retries both `Host EC2*` Spot interruptions and transient `onExitCode="1"` network/S3 errors.

### B. Critical Production Pitfalls Resolved
1. **Graviton3 SVE `SIGILL` Safeguard (`NUMBA_CPU_NAME=generic`, `NUMBA_CPU_FEATURES=+neon`)**:
   - Containerized `c7g`/`r7g`/`m7g` (`Neoverse-V1`) runtimes can expose SVE CPU flags while disabling SVE execution in the container cgroup/kernel context, causing LLVM 15 inside Numba to emit SVE instructions that crash with `SIGILL`.
   - `submit_batch_jobs.py` injects `NUMBA_CPU_NAME=generic` and `NUMBA_CPU_FEATURES=+neon` on every Batch submission.
2. **Scoped `GDAL_NUM_THREADS` (Avoiding 3.2× `/vsis3/` Slowdown)**:
   - Setting `GDAL_NUM_THREADS=ALL_CPUS` globally caused a measured **3.2× `/vsis3/` read slowdown** (`20.93–21.11s` vs `6.60s`) due to GDAL lock/block-cache contention across reader threads.
   - `seasonalDRCMIbra.py` leaves global GDAL reads single-threaded (`FastS3`/`FastLocal` drive parallel tile reads cleanly) and scopes `NUM_THREADS={eff_cpus}` strictly to output `GTiff` creation and `BuildOverviews` (`GDAL_NUM_THREADS`).
3. **Standalone Single-Shard Retry Workaround (`--shard-index`)**:
   - Submitting a standalone non-array AWS Batch job with `AWS_BATCH_JOB_ARRAY_INDEX` inside `containerOverrides.environment` causes AWS Batch to strip/ignore the reserved variable.
   - `run_aus_batch_worker.py` accepts explicit `--shard-index N`, and `submit_batch_jobs.py --retry-shard N` passes it explicitly on the CLI so single-shard recoveries execute accurately.
4. **Multi-Season Worker Resiliency (`run_aus_batch_worker.py`)**:
   - Downloads and verifies `3,801.0 MB` `ref_key` once onto `/tmp` (`FastLocal` `os.pread`), skips seasons already meeting `--min-output-bytes` in S3, overlaps the S3 upload of season `i` in `ThreadPoolExecutor(max_workers=1)` while season `i+1` computes, and removes local output GeoTIFFs immediately after `upload_file` completes to keep ephemeral storage minimal.
5. **Active Batch Monitoring & Fast Failure Triage (`submit_batch_jobs.py`)**:
   - `--wait` and `--watch-job-id` poll `describe_jobs` + CloudWatch Logs (`logs.get_log_events`) every `--poll-s` (default `30s`), detect `RUNNABLE` Spot capacity stalls beyond `--max-runnable-s`, stream CloudWatch output live, fail fast on deterministic errors, and verify S3 object counts/byte sizes on completion.

---

## 5. Observed Benchmarks

### A. Cibo Labs AWS Batch Graviton (`arm64`) Benchmarks (`window=1501`, `--no-ibra`)

| Benchmark | Hardware / Mode | Grid Dimensions | Wall Time | Throughput / Notes |
| :--- | :--- | :--- | :--- | :--- |
| **NSW `field.geojson` (`v1` baseline)** | `r7g.2xlarge` (8 vCPU Spot) | `29,294 × 27,403` (`802.7M px`) | `1,358.57 s` | Baseline single-histogram path |
| **NSW `field.geojson` (`v3`, 66-season array)** | `m7g.4xlarge` / `c7gd.2xlarge` / `m7g.2xlarge` | `29,294 × 27,403` (`802.7M px`) | `105–109 s` | `2.83–3.01 GB` max RSS |
| **NSW `field.geojson` (`v4` 2D Tiled + `FastS3`)** | 64 vCPU Graviton Spot | `29,294 × 27,403` (`802.7M px`) | **`14.38 s`** | **`0` mismatches vs `v3`** (~94× faster than `v1`) |
| **Continental Australia (`m202603202605`, single)** | `c7g.16xlarge` (64 vCPU Spot, `FastS3`) | `141,481 × 135,159` (`19,122.4M px`) | **`164.72 s`** (`2m 44s`) | Avg `116.09M px/s`; `io_wait=12.1s`, `numba=93.8s`, `write=11.6s`, `ovr=38.9s`, `up=8.2s`, `max_rss=16,622.1 MB`, out `4,873.6 MB`; `0` mismatches vs `nsw_field` `v3`; est. Spot `$0.03664` (`$0.8008/hr`) |
| **Multi-Season Worker (`m198912199002`, `FastLocal`)** | `r7g.16xlarge` (64 vCPU Spot, NVMe ref cache) | `141,481 × 135,159` (`17,372.4M px`) | **`110.23 s`** | Ref cache staged in `6.05s` (`3,801.0 MB`); avg `157.60M px/s` (`io_wait=11.6s`, `numba=51.6s`, `write=9.6s`, `ovr=37.3s`, `max_rss=16,374.8 MB`); overlapped S3 upload `1,975.3 MB` in `4.3s` |

### B. Serpentine Kernel & Concurrent Multi-Band I/O Benchmarks (`v5` vs `v4`, Local 4-vCPU `AMD EPYC 9R14`)

Directly measured apples-to-apples benchmarks (`NUMBA_NUM_THREADS=4`, `window=1501`, warm JIT, `0` pixel differences verified between `v4` (`354491b`) and `v5` Serpentine outputs):

| Kernel / Reader Benchmark | Test Workload | `v4` (`354491b`) | `v5` Serpentine (`strip_h=16384`) | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **`--ibra` (`k=1` single-region)** | `3,000 × 2,000` input, `1,500 × 2,000` (`3.0M px`) core chunk | `9.811 s` (`0.31M px/s`) | **`0.259 s` (`11.60M px/s`)** | **`37.94×`** |
| **`--ibra` (`4`-region boundary grid)** | `3,000 × 2,000` input, `3.0M px` core chunk (`k=1..4`) | `10.247 s` (`0.29M px/s`) | **`0.171 s` (`17.57M px/s`)** | **`60.01×`** |
| **`--ibra` (`70%` ocean coastal)** | `3,000 × 2,000` input, `3.0M px` core chunk | `3.719 s` (`0.81M px/s`) | **`0.048 s` (`62.46M px/s`)** | **`77.42×`** |
| **`--no-ibra` (dense land)** | `3,000 × 2,000` input, `3.0M px` core chunk | `0.339 s` (`8.84M px/s`) | **`0.234 s` (`12.82M px/s`)** | **`1.45×`** |
| **`--no-ibra` (`70%` ocean coastal)** | `3,000 × 2,000` input, `3.0M px` core chunk | `0.108 s` (`27.70M px/s`) | **`0.079 s` (`38.19M px/s`)** | **`1.38×`** |
| **Continental-width chunk (`--no-ibra` vs `--ibra`)** | `3,500 × 8,000` input, `2,000 × 8,000` (`16.0M px`, IBRA `17..255`) | — | `--no-ibra`: `1.479 s` (`10.82M px/s`)<br>`--ibra`: `1.487 s` (`10.76M px/s`) | **Virtual parity (`--ibra ≈ --no-ibra`)** |
| **Multi-band S3 `FastS3` read (`read_all_windows`)** | Live `/vsis3/cibo-dat-oregon/` 3-band `4,096 × 2,048` (`io_workers=24`) | `0.164 s` (sequential) | **`0.100 s`** (concurrent `coord_pool` + `band_pools`) | **`1.65×`** |

### C. Legacy Micro-Benchmarks (`maskChangeSteepIbra.py` & Pre-Serpentine `refpix_numba_ibra.py`)
- **Original per-row `refpix_numba_ibra.py`** (`2000 × 2000`, `4,000,000 px`, `window=1501`, 8 cores): `5.80 s` (`689,599 px/s`, `2.13×` faster than `gfortran`/`f2py` at `12.38 s`; now superseded by Serpentine `v5` at `11.6–17.6M px/s` dense on just 4 cores).
- **`maskChangeSteepIbra.py`**: Ocean fast-path skip + reusable in-memory `GDAL` dataset + single-pass OGR spatial filter reduced block time from `64.5 ms` to `30.2 ms` (`2.13×` faster).

---

## 6. Verification and Test Suite (`test_ibra_drcm.py`) & Portable Test Data (`test_data/`)

The repository includes ~`495 KiB` of real Australian Albers (`EPSG:3577`) Landsat test chips in `test_data/` so contributors outside Cibo Labs can run full end-to-end satellite pipeline tests without AWS credentials:
- `test_data/lztmre_aus_m198912202012_dixa2_change_slope_ibra.tif` (`520 × 520`, `156 KiB`) — 2-band reference stats + IBRA 7.0 regions (`17` Darling Riverine Plains & `58` Cobar Peneplain)
- `test_data/lztmre_aus_m202606202608_dp8a2_vinterim.tif` (`440 × 440`, `338 KiB`) — seasonal bare cover chip nested at offset `(col=40, row=40)` inside the reference grid
- `test_data/aoi_nsw_ibra_boundary.geojson` (`793 B`) — `EPSG:4326` polygon spanning the IBRA 17 / 58 boundary for offline halo-padded AOI verification

The test suite provides **14 cheat-proof verification tests** supporting both full online `/vsis3/` testing (`python test_ibra_drcm.py`) and credential-free local/CI runs (`python test_ibra_drcm.py --offline`):

| Test | Scope | Verified Properties |
| :--- | :--- | :--- |
| **Test 1** | Utility & URI Helpers | `format_time`, `derive_output_name` (`dp8a2`/`dixa2`), `_parse_s3_uri`, `_parse_vsis3_uri`, `_to_gdal_path` |
| **Test 2** | Groundcover Encoding Conversion | `convert_to_groundcover` (`dp8a2`, `dixa2`, filename stage inference, range fallback), `inplace=True`, legacy fallback, empty arrays |
| **Test 3** | Kernel Micro-Functions | `_bounds`, `_bounds_fast` (Numba), and multi-bin `_calcmean` `int64` floor rounding + empty histograms |
| **Test 4** | Independent Brute-Force Oracle & Tiled Kernels | Histogram-free sorted-array oracle (`_oracle_refpixel`) across asymmetric `(47 × 61)` grids, `190 × 215` parity, sub-window slicing, custom tile sizes, and `_KERNEL_U16` vs `_KERNEL_I32` |
| **Test 5** | Row & Column Boundary Isolation | Zero cross-IBRA contamination across both vertical and horizontal bioregion boundaries |
| **Test 6** | Ocean Masking, Empty-Block Skip & `minvalpercent` | `_build_valid_blocks` empty-block skip parity against unskipped kernel + oracle, ocean `0 → -1`, validity fraction thresholds |
| **Test 7** | Portable Real Satellite Multi-IBRA *(offline + online)* | `test_data/` `440 × 440` chip offset `(40, 40)` inside `520 × 520` ref, `FastLocal` on all bands, `4` overviews, IBRA 17 vs 58 divergence, and `0`-mismatch parity vs `_oracle_refpixel` |
| **Test 8** | Portable Real Satellite `--no-ibra` & `--geojson` *(offline + online)* | `test_data/` `--no-ibra` vs `_oracle_refpixel` + `EPSG:4326` `aoi_nsw_ibra_boundary.geojson` subsetting (`282 × 282`), `0` mismatches in `--no-ibra` and `--ibra` modes |
| **Test 9** | `FastLocal` Multi-Band, Concurrent `read_all_windows` & Fallbacks | `INTERLEAVE=BAND` 2-band `FastLocal` activation, `coord_pool` + `band_pools` `read_all_windows` parity (with & without IBRA), and clean fallback to `GDAL` on `PREDICTOR=2` and `INTERLEAVE=PIXEL` inputs |
| **Test 10** | Multi-Chunk Overlap, Spatial Offset & Strict AOI Bounds | Multi-chunk (`chunksize=96`) vs single-chunk identity; offset `(40, 35)` alignment; unpadded edge mismatch check proving halo necessity; strict `--bbox`/`--geojson` bounds & tight subset dimensions |
| **Test 11** | Batch Submission, `--save` S3 Upload & Worker Content | `load_env_file`, round-robin shard contents, Graviton SVE/`GDAL` env, `--save` upload + local delete, AOI `1 KiB` `--min-output-bytes` auto-default, `--smoke-test` `test_data/` staging, and pixel-level verification of the GeoTIFF uploaded by `run_aus_batch_worker.py` |
| **Test 12** | Serpentine Kernels (`strip_h=16384`), `1..255` IBRA & Edge Cases *(offline + online)* | Serpentine IBRA (`reg=90..255`, `k=1` vs `k>=2` bboxes, `c_start`/`c_stop` halo slicing, `window=4621` `int64` accumulators, `_IBRA_KERNEL_I32`), `--no-ibra` (`strip_h=16384` vs `24`, ocean trimming), `cover=105`/`ground=120 > 101` bounds guards, `window=1`, `nrow=1`, `ncol<halfwindow`, `minvalpercent=0.0`, and empty slices vs `_oracle_refpixel` |
| **Test 13** | Stage 1 NumPy-2 Guard, `dixa2`/`255` & Off-Grid Alignment | Unambiguous `dixa2` (`100..200`) vs `dp8a2` stage handling even with `255` chunks, off-grid input/ref/separate `--ibra-raster` zero-padding & CRS/pixel-size rejection, and `maskChangeSteepIbra` Stage-1 underflow guard, land-hole Band 2 persistence, and `OBJECTID > 255` `ValueError` |
| **Test 14** | Remote `/vsis3/` & `test_data/` Authenticity *(online)* | Streams `FastS3` `/vsis3/cibo-dat-oregon/` COGs (`--ibra` and `--no-ibra`), proves `test_data/` chips match live S3 pixels 100% bit-for-bit, and verifies `0` output mismatches |

Latest verified results in the `rios` environment:
- `python test_ibra_drcm.py --offline` (with AWS credential env vars unset): **13/13 offline tests passed**.
- `python test_ibra_drcm.py` (with AWS credentials in `us-west-2`): **All 14/14 tests passed** (see PR/CI run output).

---

## 7. Repository Scripts & Command-Line Usage

### Script Summary

| Script | Role |
| :--- | :--- |
| `seasonalDRCMIbra.py` | Main pipeline driver: `FastS3`/`FastLocal` tiled readers + concurrent `read_all_windows`, `--bare-encoding` (`auto`/`dp8a2`/`dixa2`), aligned-grid & off-grid zero-padded readers, overlap reuse + prefetch, `--no-ibra` & `--ibra`, `--bbox`/`--geojson`, `-o s3://...` & `--save`, overviews, cgroup CPUs, `max_rss` telemetry |
| `refpix_numba.py` | Serpentine 2D tiled `uint16`/`int64`/`int32` (`strip_h=16384`) single-histogram Numba kernel with `256 × 256` empty-block ocean trimming, `0 < value <= 101` validation, and native `r_start/r_stop/c_start/c_stop` slicing (`--no-ibra` engine) |
| `refpix_numba_ibra.py` | Serpentine 2D tiled `uint16`/`int64`/`int32` (`strip_h=16384`) multi-IBRA Numba kernel supporting all `1..255` regions, `k=1`/`k>=2` tight-bbox passes, `256 × 256` 2-bit block skipping, `0 < value <= 101` validation, and native `c_start`/`c_stop` slicing (`--ibra` engine) |
| `run_aus_batch_worker.py` | AWS Batch array shard worker: local NVMe reference caching, `--skip-existing` S3 check (auto `1 KiB` AOI `--min-output-bytes` default), overlapped S3 uploads, explicit `--shard-index`/`--ibra-raster`/`--bbox`/`--geojson`/`--threads` |
| `submit_batch_jobs.py` | AWS Batch orchestrator: S3 season discovery & `YYYYMM` filtering, round-robin sharding, S3 script + `test_data/` staging, Graviton SVE + GDAL env injection, `--smoke-test`, `--retry-shard`, `--wait`, `--watch-job-id`, `--dry-run` |
| `maskChangeSteepIbra.py` | RIOS Stage 1 preprocessor generating 2-band NCAS/slope masked reference stats + IBRA raster (`--bare-encoding`, `--ibra-attr` validated in `1..255`) |
| `test_ibra_drcm.py` | 14-test cheat-proof verification suite (`--offline` using `test_data/` + full `/vsis3/` mode) |

### Common Operational Commands

#### 1. Dry-Run Discovery & Production AWS Batch Array Submission (`submit_batch_jobs.py`)

```bash
# Inspect live S3 season discovery (147 seasons), round-robin shards, and submission plan (no AWS writes)
conda run -n rios python submit_batch_jobs.py --env-file .env --dry-run

# Submit a filtered multi-season Batch run, skipping seasons already in S3, and monitor CloudWatch + S3
conda run -n rios python submit_batch_jobs.py \
    --env-file .env \
    --start-ym 200912 \
    --end-ym 202608 \
    --shards 15 \
    --skip-existing \
    --no-ibra \
    --smoke-test \
    --wait
```

#### 2. Retry a Failed Array Shard or Resume Monitoring an Existing Batch Job

```bash
# Retry only shard 7 cleanly via explicit --shard-index against the existing staged manifest
conda run -n rios python submit_batch_jobs.py \
    --env-file .env \
    --retry-manifest s3://cibo-dat-oregon/DRCM/_batch_scripts/<run_tag>/aus_seasons_manifest.json \
    --retry-shard 7 \
    --wait

# Attach the CloudWatch + Batch watcher to a running job ID
conda run -n rios python submit_batch_jobs.py \
    --env-file .env \
    --watch-job-id <aws-batch-job-id>
```

#### 3. Run `seasonalDRCMIbra.py` Directly (Continental, AOI GeoJSON/BBox, or Legacy `--save`)

```bash
# Continental run streaming from/to S3 in high-speed --no-ibra mode
conda run -n rios python seasonalDRCMIbra.py \
    /vsis3/cibo-dat-oregon/GC/lztmre_aus_m202603202605_dp8a2_vinterim.tif \
    --percentile-file /vsis3/cibo-dat-oregon/DRCM/lztmre_aus_m198912202012_dixa2_change_slope_ibra.tif \
    --window 1501 \
    --chunksize 16384 \
    --minvalpercent 0.10 \
    --no-ibra \
    -o s3://cibo-dat-oregon/DRCM/aus/

# Regional AOI subset with halo padding from an EPSG:4326 GeoJSON
conda run -n rios python seasonalDRCMIbra.py \
    /vsis3/cibo-dat-oregon/GC/lztmre_aus_m202606202608_dp8a2_vinterim.tif \
    --percentile-file /vsis3/cibo-dat-oregon/DRCM/lztmre_aus_m198912202012_dixa2_change_slope_ibra.tif \
    --geojson field.geojson \
    --window 1501 \
    --no-ibra \
    -o s3://cibo-dat-oregon/DRCM/nsw_field/

# Local run saving backward-compatible output to s3://cibo-dat-oregon/DRCM/ and deleting local file
conda run -n rios python seasonalDRCMIbra.py \
    lztmre_nsw_m202606202608_dp8a2_vinterim.tif \
    --percentile-file lztmre_aus_m198912202012_dixa2_change_slope_ibra.tif \
    --save
```

#### 4. Regenerate the 2-Band Reference + IBRA Raster (`maskChangeSteepIbra.py`)

```bash
conda run -n rios python maskChangeSteepIbra.py \
    --change woodyMaxMinCleared.tif \
    --terrain slope_relief_class_3s.tif \
    --bare lztmre_aus_m198912202012_dixa2_bareStats.tif \
    --ibra ibra.fgb \
    -o lztmre_aus_m198912202012_dixa2_change_slope_ibra.tif
```

#### 5. Run the Verification Suite (`test_ibra_drcm.py`)

```bash
# Offline / CI mode (runs Tests 1-13 including portable test_data/ real satellite chips; zero AWS creds needed)
conda run -n rios python test_ibra_drcm.py --offline

# Full 14-test suite including live /vsis3/cibo-dat-oregon streaming & chip authenticity parity
conda run -n rios python test_ibra_drcm.py
```
