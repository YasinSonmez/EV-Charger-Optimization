# Final Rebuttal Experiment Suites

The final design deliberately separates computational scaling from scientific sensitivity.

## Suite definitions

- `configs/rebuttal/final/scaling_suite.json` runs the complete secondary+ pipeline on the approximately 100-, 500-, and 1,000-node DC windows. It uses 5 candidates, 2 chargers, K=16, independent routes, balanced charger quotas, uniform initialization, 20 queue replications per better-response iteration, and a 300-iteration cap with cycle detection.
- `configs/rebuttal/final/sensitivity_suite.json` expands to exactly 20 jobs on one 100-node network. The baseline runs first. Every subsequent job changes one factor and reuses the baseline network and BPR calibration after validating their exact hashes and calibration identity.

The sensitivity factors are demand, F2 share, OD count, candidate/charger budget, K, route source, initialization, NE replication count, and queue seed. There is no alpha sweep; alpha remains the 1% equilibrium criterion. The BPR seed remains 42 when the queue seed changes.

Independent F2 route sets contain up to K routes in total. Slots are assigned one at a time to the least-loaded charger with a remaining candidate; ties use the lowest-ranked next route, then charger id. With sufficient supply each charger receives `floor/ceil(K/C)` routes, so counts differ by at most one; chargers with fewer feasible routes are capped and other chargers fill the remaining slots. The CG-recovered sensitivity job applies the same rule, ranking routes within each charger by recovered flow.

## Local validation and execution

```bash
python run_suite.py --manifest configs/rebuttal/final/scaling_suite.json \
  --results-root results/rebuttal/scaling --validate-only

python run_suite.py --manifest configs/rebuttal/final/sensitivity_suite.json \
  --results-root results/rebuttal/sensitivity --validate-only

python run_suite.py --manifest configs/rebuttal/final/sensitivity_suite.json \
  --results-root results/rebuttal/sensitivity --resume
```

Suite execution is fail-fast. A rerun with `--resume` skips digest-matched complete runs, resumes the first failed/partial run, and proceeds only after it succeeds.

The smoke suite validated this path locally (`configs/rebuttal/smoke/smoke_suite.json`, two 49-node jobs reusing cached artifacts): both jobs completed with all reviewer baselines, balanced 4/4 route quotas, the dependency-artifact mechanism, and a valid export bundle.

## Slurm/Apptainer execution (final, paste-ready)

On the login node, check out the executed commit and build the image once:

```bash
git clone --branch rebuttal-scalable-pipeline \
  https://github.com/YasinSonmez/EV-Charger-Optimization.git evopt
cd evopt
git log -1 --oneline            # expect: cc050d21 rebuttal: final scaling/...
apptainer pull evopt-rebuttal.sif docker://REGISTRY/evopt:rebuttal-runtime
```

Publish that image first from an amd64 machine (see `docs/remote_experiment_guide.md`);
rebuild is needed only when dependencies or the native library change, because
workspace mode mounts the code from the checkout.

Submit both suites as two independent `afterok` chains. Use separate results
roots so the scaling and sensitivity runs never share a summary:

```bash
cd evopt
export EVOPT_IMAGE="$PWD/evopt-rebuttal.sif"
export EVOPT_PROJECT_DIR="$PWD"
export EVOPT_CACHE_DIR="$PWD/osm-cache"
export EVOPT_EXECUTION_MODE=workspace

export EVOPT_RESULTS_DIR="$PWD/results/rebuttal/scaling"
scripts/submit_suite.sh configs/rebuttal/final/scaling_suite.json

export EVOPT_RESULTS_DIR="$PWD/results/rebuttal/sensitivity"
scripts/submit_suite.sh configs/rebuttal/final/sensitivity_suite.json
```

Each experiment receives 55 CPUs, 64 GiB, and a 12-hour budget (`scripts/run_slurm_suite.sh`). Jobs are linked with Slurm `afterok`, so a failed job prevents downstream experiments from starting. Re-running `submit_suite.sh` skips completed jobs and rebuilds the chain beginning at the first non-complete digest.

Monitor with `squeue` and inspect per-job logs under `slurm_logs/`. When the
chains finish, verify both roots report all-complete:

```bash
python run_suite.py --manifest configs/rebuttal/final/scaling_suite.json \
  --results-root results/rebuttal/scaling --validate-only --summarize
python run_suite.py --manifest configs/rebuttal/final/sensitivity_suite.json \
  --results-root results/rebuttal/sensitivity --validate-only --summarize
```

## Compact result export

```bash
python run_suite.py \
  --manifest configs/rebuttal/final/scaling_suite.json \
  --results-root results/rebuttal/scaling \
  --summarize \
  --export-bundle rebuttal_scaling_results.zip

python run_suite.py \
  --manifest configs/rebuttal/final/sensitivity_suite.json \
  --results-root results/rebuttal/sensitivity \
  --summarize \
  --export-bundle rebuttal_sensitivity_results.zip
```

The ZIP contains compact run, placement, queue, BPR, and paper-table CSVs; paper PNGs; resolved configurations; status records; provenance; and checksums. It excludes caches, simulation traces, BPR observations, pickles, and checkpoints and is rejected if it exceeds 25 MiB.

## Interpretation limits

- Nested DC windows support computational scaling claims, not geographic generalization.
- F2 is the share required to charge once, not general EV penetration.
- Demand is controlled synthetic demand, not observed traffic demand.
- Low-R² BPR fits are disclosed and retained only when all links still have finite nonnegative model parameters, positive capacity/FFT, full coverage, and provenance.
- A cycle-state queue result is a labeled heuristic approximation, never a verified Nash equilibrium.
- Balanced top-K libraries reduce charger-specific route-count bias but remain restricted strategy sets.
