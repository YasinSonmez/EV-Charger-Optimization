#!/usr/bin/env bash
#SBATCH --job-name=evopt-scale
#SBATCH --array=0-2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -euo pipefail

: "${EVOPT_IMAGE:?Set EVOPT_IMAGE to a Docker/Apptainer image URI or local .sif}"
: "${EVOPT_PROJECT_DIR:?Set EVOPT_PROJECT_DIR to the pulled Git repository}"
: "${EVOPT_RESULTS_DIR:?Set EVOPT_RESULTS_DIR to persistent result storage}"
: "${EVOPT_CACHE_DIR:?Set EVOPT_CACHE_DIR to persistent OSM cache storage}"

SUITE_MANIFEST="${SUITE_MANIFEST:-configs/rebuttal/suite.json}"
mkdir -p "$EVOPT_RESULTS_DIR" "$EVOPT_CACHE_DIR" "$EVOPT_PROJECT_DIR/slurm_logs"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

if [[ "$EVOPT_IMAGE" == docker://* ]]; then
    SIF_PATH="$EVOPT_CACHE_DIR/evopt-${SLURM_JOB_ID}.sif"
    apptainer pull --force "$SIF_PATH" "$EVOPT_IMAGE"
else
    SIF_PATH="$EVOPT_IMAGE"
fi

child=""
checkpoint_and_exit() {
    if [[ -n "$child" ]]; then
        kill -TERM "$child" 2>/dev/null || true
        wait "$child" 2>/dev/null || true
    fi
    exit 99
}
trap checkpoint_and_exit USR1 TERM

apptainer exec --cleanenv \
    --env "EVOPT_GRAPH_CACHE_DIR=/cache" \
    --env "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}" \
    --bind "$EVOPT_PROJECT_DIR:/workspace:ro" \
    --bind "$EVOPT_RESULTS_DIR:/results" \
    --bind "$EVOPT_CACHE_DIR:/cache" \
    "$SIF_PATH" \
    bash -lc "cd /workspace && python run_suite.py --manifest '$SUITE_MANIFEST' --results-root /results --index '${SLURM_ARRAY_TASK_ID}' --resume" &
child=$!
wait "$child"
