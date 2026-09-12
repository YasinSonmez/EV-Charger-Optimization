#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: scripts/submit_suite.sh [--start-index N] [--continue-on-failure] MANIFEST" >&2
}
start_index=""
continue_on_failure=false
manifest=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --start-index)
            [[ $# -ge 2 && "$2" =~ ^[0-9]+$ ]] || { usage; exit 2; }
            start_index="$2"
            shift 2
            ;;
        --continue-on-failure)
            continue_on_failure=true
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --*)
            usage
            exit 2
            ;;
        *)
            [[ -z "$manifest" ]] || { usage; exit 2; }
            manifest="$1"
            shift
            ;;
    esac
done
[[ -n "$manifest" ]] || { usage; exit 2; }
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
if [[ -n "$start_index" ]]; then
    if (( start_index > count )); then
        echo "Start index $start_index is outside suite range 0..$count" >&2
        exit 2
    fi
    if (( start_index > first )); then
        first="$start_index"
    fi
fi
if (( first >= count )); then
    echo "All $count experiments are already complete."
    exit 0
fi

dependency=""
dependency_policy="afterok"
if [[ "$continue_on_failure" == true ]]; then
    dependency_policy="afterany"
fi
for ((index=first; index<count; index++)); do
    command=(sbatch --parsable)
    if [[ -n "$dependency" ]]; then
        command+=(--dependency="$dependency_policy:$dependency")
    fi
    command+=(--export="ALL,SUITE_MANIFEST=$manifest,SUITE_INDEX=$index,EVOPT_PROJECT_DIR=$project_dir")
    dependency="$("${command[@]}" "$project_dir/scripts/run_slurm_suite.sh")"
    echo "Submitted suite index $index as job $dependency (dependency: $dependency_policy)"
done
