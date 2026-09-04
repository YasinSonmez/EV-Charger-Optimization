# Remote scaling experiment guide

This branch provides one coordinate-driven workflow for any region. The three
files in `configs/rebuttal/` are replaceable examples. They request secondary+
roads, topology simplification, 5 m intersection consolidation, deterministic
candidate/OD generation, simple fixed-reference BPR fitting, and the original
better-response queue method.

## Calibrated numerical settings

- BPR: 17 capacity-relative flow levels from 0 to 2C, one simulation per
  level, with no replication or bootstrap layer.
- Nash better response: 20 independent simulations of the current assignment
  per response step (`NUM_ITERS=20`).
- Nash stopping tolerance: 1% relative gap (`ALPHA=0.01`).
- Nash response-step cap: 300 (`MAX_NE_ITERATIONS=300`). Reaching the cap or
  detecting a repeated assignment is a nonconverged result, and the pipeline
  stops before the final placement comparison.
- Final greedy/exhaustive comparison: 20 replications (`N=20`).

These are bounded runtime/accuracy compromises, not universal constants. The
calibration evidence and limitations are in `docs/sampling_count_calibration.md`.
In particular, a 100-step legacy-Nash test did not reach the 1% criterion.
Linear trend fits put a possible crossing around 168--253 steps, so 300 is a
budget-aware diagnostic ceiling, not a guarantee of convergence.

## Scientific eligibility

A full result is eligible only when:

- the generated graph is within the configured node tolerance and all
  generated points are feasible in the final SCC;
- every active BPR fit uses finite, positive observations and fixed capacity
  and free-flow references, with no proxy or arbitrary fallback;
- every placement passes solver-status, nonnegative-flow, conservation, and
  objective-recomputation checks;
- every queue assignment reaches the 1% gap before the iteration cap without
  cycling; and
- the final comparison completes all configured replications.

## Validate and inspect before computing

```bash
python run_suite.py --manifest configs/rebuttal/suite.json --validate-only
python pipeline.py --config configs/rebuttal/scale_0100.json --network-only
```

`expected_nodes` validates fixed geography; it never resizes the bounding box.
Because OSM changes, retain the raw cache and use `"cache_policy": "require"`
for exact reruns after the first download. Run all three configs network-only
before BPR fitting and visually inspect their pruning/scenario maps.

## Local end-to-end sanity check

`configs/rebuttal/sanity_small.json` uses the same secondary+ network method,
17-level strict BPR fitting, 20 queue replications, 1% Nash criterion, and
300-iteration safety cap as the scaling configs. It deliberately uses a
49-node window, one candidate/charger, and only one F1 plus one F2 vehicle so
that it tests every pipeline stage quickly; it is an execution check, not a
scientific placement experiment.

```bash
MPLCONFIGDIR=/tmp/evopt-mpl XDG_CACHE_HOME=/tmp/evopt-cache \
conda run -n evopt python pipeline.py \
  --config configs/rebuttal/sanity_small.json \
  --results-root results/sanity-runs
```

The verified local run completed in 171.7 seconds on eight CPUs, validated all
100 BPR links, converged its single queue configuration in one iteration, and
passed output sanity checks. Larger-demand diagnostic attempts on the same
network entered exact cycles at iterations 5--6; this is why production runs
retain cycle detection and must not infer a production iteration count from
the two-vehicle sanity result.

## Container runtime model

The image contains the pinned Conda environment and Linux C++ shortest-path
library. `scripts/run_container.sh` supports two deliberately distinct modes:

- `workspace` (default) mounts a host Git checkout read-only at `/workspace`.
  Python and configuration changes therefore do not require an image rebuild.
  The checkout commit and dirty/clean state are recorded in every run.
- `image` runs the code baked into the image. Use this with an immutable image
  digest for final paper results.

In both modes, results and the OSM cache are host directories. The native
library is loaded from `/opt/evopt/lib/liblsp.so`, outside either code
directory, so mounting a checkout cannot hide it.

Build the runtime once. Rebuild only after dependencies, the native library,
or the desired immutable image-mode source changes:

```bash
docker build \
  --build-arg VCS_REF="$(git rev-parse HEAD)" \
  -t evopt:rebuttal-runtime .
```

Run any checked-out configuration and inspect outputs on the host immediately:

```bash
scripts/run_container.sh \
  --engine docker \
  --image evopt:rebuttal-runtime \
  --mode workspace \
  --config configs/rebuttal/sanity_small.json \
  --results results/docker-sanity \
  --cache data/graphs \
  --cpus 8
```

Add `--resume`, `--network-only`, or `--validate-config` as needed. The script
prints the resolved image, execution mode, commit, result path, cache path, and
CPU count before starting.

## Apptainer and Slurm

Build and publish the cluster image for `linux/amd64` from an amd64 machine or
with Docker Buildx:

```bash
docker buildx build --platform linux/amd64 \
  --build-arg VCS_REF="$(git rev-parse HEAD)" \
  -t REGISTRY/evopt:rebuttal-runtime --push .
```

On the cluster login node, clone the same branch and convert the OCI image
once. Keeping both the checkout and SIF makes the executed source inspectable:

```bash
git clone --branch rebuttal-scalable-pipeline \
  https://github.com/YasinSonmez/EV-Charger-Optimization.git evopt
cd evopt
apptainer pull evopt-rebuttal.sif docker://REGISTRY/evopt:rebuttal-runtime
mkdir -p results osm-cache slurm_logs
```

After that preparation, a complete configurable run is one command:

```bash
scripts/run_container.sh \
  --engine apptainer \
  --image "$PWD/evopt-rebuttal.sif" \
  --mode workspace \
  --config configs/rebuttal/sanity_small.json \
  --results "$PWD/results" \
  --cache "$PWD/osm-cache" \
  --cpus 16
```

For immutable final execution, replace `--mode workspace` with `--mode image`.
The selected host configuration is still mounted at `/inputs/config.json`.

Submit the three independent scale configurations as a Slurm array:

```bash
export EVOPT_IMAGE="$PWD/evopt-rebuttal.sif"
export EVOPT_PROJECT_DIR="$PWD"
export EVOPT_RESULTS_DIR="$PWD/results"
export EVOPT_CACHE_DIR="$PWD/osm-cache"
export EVOPT_EXECUTION_MODE=workspace
sbatch scripts/run_slurm_suite.sh
```

The array script delegates to the same container launcher, mounts source
read-only, and propagates the Slurm CPU allocation. Create `slurm_logs/` before
`sbatch`, because Slurm opens log files before the job body starts. The array
requests 16 CPUs, 64 GiB, 72 hours, and a five-minute preemption signal.
BLAS/OpenMP are limited to one thread per worker, and each scale is independent.

Resume or summarize through the same interface:

```bash
scripts/run_container.sh --engine apptainer \
  --image "$PWD/evopt-rebuttal.sif" --mode workspace \
  --manifest configs/rebuttal/suite.json --index 0 \
  --results "$PWD/results" --cache "$PWD/osm-cache" --cpus 16 --resume

scripts/run_container.sh --engine apptainer \
  --image "$PWD/evopt-rebuttal.sif" --mode workspace \
  --manifest configs/rebuttal/suite.json \
  --results "$PWD/results" --cache "$PWD/osm-cache" --summarize
```

Docker was exercised end-to-end with the sanity configuration: every stage
passed in 265.1 seconds on Docker Desktop with eight CPUs. It reproduced the
same topology size, BPR pass count, charger choice, Nash status, queue travel
time, and greedy/exhaustive conclusion as the native run. Tiny projection
floating-point differences across macOS and Linux intentionally produce a
different byte-level network hash. Final experiments should all use the same
`linux/amd64` image digest and compare native/container runs by recorded
scientific invariants rather than claiming byte-identical artifacts. The
revised amd64 image also passed immutable-image validation, mounted-workspace
suite validation, native-library loading, and a complete checkpoint-resume
run under amd64 emulation.

## BPR worker benchmark

```bash
python benchmarks/benchmark_bpr_workers.py \
  --config configs/rebuttal/scale_0100.json \
  --output-dir /path/to/results/bpr-worker-benchmark \
  --workers 1,2,4,8,16 --max-links 8
```

Use `worker_scaling.csv` to select the worker count where added cores stop
improving throughput. This bounded benchmark is a machine-sizing check, not a
paper result.
