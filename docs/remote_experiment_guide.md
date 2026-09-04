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

## Container and cluster launch

```bash
docker build --platform linux/amd64 -t USER/evopt:rebuttal-scalable .
docker push USER/evopt:rebuttal-scalable

git clone --branch rebuttal-scalable-pipeline REPOSITORY_URL evopt
cd evopt
mkdir -p slurm_logs /path/to/results /path/to/osm-cache
export EVOPT_IMAGE=docker://USER/evopt:rebuttal-scalable
export EVOPT_PROJECT_DIR="$PWD"
export EVOPT_RESULTS_DIR=/path/to/results
export EVOPT_CACHE_DIR=/path/to/osm-cache
sbatch scripts/run_slurm_suite.sh
```

The Slurm array requests 16 CPUs, 64 GiB, 72 hours, and a five-minute
preemption signal. BLAS/OpenMP are limited to one thread per worker. Each scale
is independent.

Resume or summarize with:

```bash
python run_suite.py --manifest configs/rebuttal/suite.json \
  --results-root /path/to/results --index 0 --resume
python run_suite.py --manifest configs/rebuttal/suite.json \
  --results-root /path/to/results --summarize
```

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
