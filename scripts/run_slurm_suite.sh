#!/usr/bin/env bash
#SBATCH --job-name=evopt-scale
#SBATCH --cpus-per-task=55
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

: "${EVOPT_IMAGE:?Set EVOPT_IMAGE to an Apptainer .sif or docker:// image URI}"
: "${EVOPT_PROJECT_DIR:=${SLURM_SUBMIT_DIR:-$PWD}}"
: "${EVOPT_RESULTS_DIR:?Set EVOPT_RESULTS_DIR to persistent result storage}"
: "${EVOPT_CACHE_DIR:?Set EVOPT_CACHE_DIR to persistent OSM cache storage}"

SUITE_MANIFEST="${SUITE_MANIFEST:-configs/rebuttal/suite.json}"
SUITE_INDEX="${SUITE_INDEX:?Set SUITE_INDEX for the chained suite job}"
EVOPT_EXECUTION_MODE="${EVOPT_EXECUTION_MODE:-workspace}"
mkdir -p "$EVOPT_RESULTS_DIR" "$EVOPT_CACHE_DIR"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

child=""
checkpoint_and_exit() {
    if [[ -n "$child" ]]; then
        kill -TERM "$child" 2>/dev/null || true
        wait "$child" 2>/dev/null || true
    fi
    exit 99
}
trap checkpoint_and_exit USR1 TERM

"$EVOPT_PROJECT_DIR/scripts/run_container.sh" \
    --engine apptainer \
    --image "$EVOPT_IMAGE" \
    --mode "$EVOPT_EXECUTION_MODE" \
    --workspace "$EVOPT_PROJECT_DIR" \
    --manifest "$EVOPT_PROJECT_DIR/$SUITE_MANIFEST" \
    --results "$EVOPT_RESULTS_DIR" \
    --cache "$EVOPT_CACHE_DIR" \
    --cpus "$SLURM_CPUS_PER_TASK" \
    --index "$SUITE_INDEX" \
    --resume &
child=$!
set +e
wait "$child"
child_status=$?
set -e
child=""
echo "EVOPT container process exited with status $child_status"
exit "$child_status"
