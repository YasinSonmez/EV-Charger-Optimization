#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: scripts/submit_suite.sh MANIFEST" >&2
    exit 2
fi
: "${EVOPT_IMAGE:?Set EVOPT_IMAGE to an Apptainer .sif or docker:// URI}"
: "${EVOPT_RESULTS_DIR:?Set EVOPT_RESULTS_DIR to persistent result storage}"
: "${EVOPT_CACHE_DIR:?Set EVOPT_CACHE_DIR to persistent OSM cache storage}"

project_dir="${EVOPT_PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
manifest="$1"
mkdir -p "$project_dir/slurm_logs" "$EVOPT_RESULTS_DIR" "$EVOPT_CACHE_DIR"
count="$(python "$project_dir/run_suite.py" --manifest "$manifest" \
    --results-root "$EVOPT_RESULTS_DIR" --print-count)"
first="$(python "$project_dir/run_suite.py" --manifest "$manifest" \
    --results-root "$EVOPT_RESULTS_DIR" --first-pending)"
if (( first >= count )); then
    echo "All $count experiments are already complete."
    exit 0
fi

dependency=""
for ((index=first; index<count; index++)); do
    command=(sbatch --parsable)
    if [[ -n "$dependency" ]]; then
        command+=(--dependency="afterok:$dependency")
    fi
    command+=(--export="ALL,SUITE_MANIFEST=$manifest,SUITE_INDEX=$index,EVOPT_PROJECT_DIR=$project_dir")
    dependency="$("${command[@]}" "$project_dir/scripts/run_slurm_suite.sh")"
    echo "Submitted suite index $index as job $dependency"
done
