#!/usr/bin/env bash
# Run one configuration or one suite entry with Docker or Apptainer.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/run_container.sh --image IMAGE --config CONFIG [options]
  scripts/run_container.sh --image IMAGE --manifest MANIFEST [--index N] [options]

Options:
  --engine auto|docker|apptainer  Runtime to use (default: auto)
  --mode workspace|image         Mounted Git code or code baked into image
                                 (default: workspace)
  --workspace DIR                Git checkout mounted read-only (default: repo root)
  --results DIR                  Host results directory
  --cache DIR                    Host OSM cache directory
  --cpus N                       CPU limit and worker count advertised to pipeline
  --resume                       Resume the deterministic run directory
  --network-only                 Run only network generation (single config only)
  --validate-config              Validate configuration and exit (single config only)
  --pruning-sweep                Run pruning sweep (single config only)
  --summarize                    Summarize suite results (manifest only)
  --validate-only                Validate every suite config (manifest only)

Environment equivalents: EVOPT_IMAGE, EVOPT_ENGINE, EVOPT_EXECUTION_MODE,
EVOPT_PROJECT_DIR, EVOPT_RESULTS_DIR, EVOPT_CACHE_DIR, and EVOPT_CPUS.
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
default_workspace="$(cd "$script_dir/.." && pwd)"
engine="${EVOPT_ENGINE:-auto}"
image="${EVOPT_IMAGE:-}"
mode="${EVOPT_EXECUTION_MODE:-workspace}"
workspace="${EVOPT_PROJECT_DIR:-$default_workspace}"
results="${EVOPT_RESULTS_DIR:-$default_workspace/results/container-runs}"
cache="${EVOPT_CACHE_DIR:-$default_workspace/data/graphs}"
cpus="${EVOPT_CPUS:-${SLURM_CPUS_PER_TASK:-}}"
config=""
manifest=""
index=""
pipeline_flags=()
suite_flags=()
single_only=false
suite_only=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --engine) engine="$2"; shift 2 ;;
        --image) image="$2"; shift 2 ;;
        --mode) mode="$2"; shift 2 ;;
        --workspace) workspace="$2"; shift 2 ;;
        --results|--results-root) results="$2"; shift 2 ;;
        --cache) cache="$2"; shift 2 ;;
        --cpus) cpus="$2"; shift 2 ;;
        --config) config="$2"; shift 2 ;;
        --manifest) manifest="$2"; shift 2 ;;
        --index) index="$2"; shift 2 ;;
        --resume) pipeline_flags+=(--resume); suite_flags+=(--resume); shift ;;
        --network-only|--validate-config|--pruning-sweep)
            pipeline_flags+=("$1"); single_only=true; shift ;;
        --summarize) suite_flags+=(--summarize); suite_only=true; shift ;;
        --validate-only) suite_flags+=(--validate-only); suite_only=true; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$image" ]]; then
    echo "Set --image or EVOPT_IMAGE." >&2
    exit 2
fi
if [[ -n "$config" && -n "$manifest" ]] || [[ -z "$config" && -z "$manifest" ]]; then
    echo "Select exactly one of --config or --manifest." >&2
    exit 2
fi
if [[ -n "$config" && "$suite_only" == true ]]; then
    echo "--summarize/--validate-only require --manifest." >&2
    exit 2
fi
if [[ -n "$manifest" && "$single_only" == true ]]; then
    echo "--network-only/--validate-config/--pruning-sweep require --config." >&2
    exit 2
fi
if [[ "$mode" != "workspace" && "$mode" != "image" ]]; then
    echo "--mode must be workspace or image." >&2
    exit 2
fi
if [[ "$engine" == "auto" ]]; then
    if command -v apptainer >/dev/null 2>&1; then
        engine="apptainer"
    elif command -v docker >/dev/null 2>&1; then
        engine="docker"
    else
        echo "Neither apptainer nor docker is installed." >&2
        exit 127
    fi
fi
if [[ "$engine" != "docker" && "$engine" != "apptainer" ]]; then
    echo "--engine must be auto, docker, or apptainer." >&2
    exit 2
fi
if [[ -n "$cpus" ]] && ! [[ "$cpus" =~ ^[1-9][0-9]*$ ]]; then
    echo "--cpus must be a positive integer." >&2
    exit 2
fi

workspace="$(cd "$workspace" && pwd)"
mkdir -p "$results" "$cache"
results="$(cd "$results" && pwd)"
cache="$(cd "$cache" && pwd)"

code_commit="baked into image (recorded at runtime)"
container_workdir="/app"
pipeline_program="/app/pipeline.py"
suite_program="/app/run_suite.py"
mounts=()
if [[ "$mode" == "workspace" ]]; then
    if [[ ! -f "$workspace/pipeline.py" ]]; then
        echo "Workspace does not contain pipeline.py: $workspace" >&2
        exit 2
    fi
    container_workdir="/workspace"
    pipeline_program="/workspace/pipeline.py"
    suite_program="/workspace/run_suite.py"
    mounts+=("$workspace:/workspace:ro")
    if git -C "$workspace" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        code_commit="$(git -C "$workspace" rev-parse HEAD)"
        if [[ -n "$(git -C "$workspace" status --porcelain)" ]]; then
            code_commit="${code_commit}-dirty"
        fi
    else
        code_commit="workspace-without-git-metadata"
    fi
fi

if [[ -n "$config" ]]; then
    config_dir="$(cd "$(dirname "$config")" && pwd)"
    config_abs="$config_dir/$(basename "$config")"
    [[ -f "$config_abs" ]] || { echo "Config not found: $config_abs" >&2; exit 2; }
    mounts+=("$config_abs:/inputs/config.json:ro")
    command_args=("$pipeline_program" --config /inputs/config.json --results-root /results "${pipeline_flags[@]}")
else
    manifest_dir="$(cd "$(dirname "$manifest")" && pwd)"
    manifest_abs="$manifest_dir/$(basename "$manifest")"
    [[ -f "$manifest_abs" ]] || { echo "Manifest not found: $manifest_abs" >&2; exit 2; }
    mounts+=("$manifest_dir:/suite:ro")
    command_args=("$suite_program" --manifest "/suite/$(basename "$manifest_abs")" --results-root /results "${suite_flags[@]}")
    if [[ -n "$index" ]]; then
        command_args+=(--index "$index")
    fi
fi

liblsp_path="${EVOPT_LIBLSP_PATH:-/opt/evopt/lib/liblsp.so}"
image_digest="${EVOPT_IMAGE_DIGEST:-}"
docker_image=""
if [[ "$engine" == "docker" ]]; then
    docker_image="${image#docker://}"
    if [[ -z "$image_digest" ]]; then
        image_digest="$(docker image inspect --format '{{.Id}}' "$docker_image" 2>/dev/null || true)"
    fi
fi
echo "EVOPT container launch"
echo "  engine:     $engine"
echo "  image:      $image"
echo "  digest:     ${image_digest:-not resolved by launcher}"
echo "  mode:       $mode"
echo "  code:       $code_commit"
echo "  results:    $results"
echo "  OSM cache:  $cache"
echo "  CPUs:       ${cpus:-container default}"

common_env=(
    "EVOPT_CONTAINERIZED=1"
    "EVOPT_EXECUTION_MODE=$mode"
    "EVOPT_IMAGE_REF=$image"
    "EVOPT_LIBLSP_PATH=$liblsp_path"
    "EVOPT_GRAPH_CACHE_DIR=/cache"
    "OMP_NUM_THREADS=1"
    "OPENBLAS_NUM_THREADS=1"
    "MKL_NUM_THREADS=1"
    "NUMEXPR_NUM_THREADS=1"
    "MPLCONFIGDIR=/tmp/evopt-matplotlib"
    "HOME=/tmp/evopt-home"
)
if [[ "$mode" == "workspace" ]]; then
    common_env+=("EVOPT_CODE_COMMIT=$code_commit")
fi
if [[ -n "$image_digest" ]]; then
    common_env+=("EVOPT_IMAGE_DIGEST=$image_digest")
fi
if [[ -n "$cpus" ]]; then
    common_env+=("SLURM_CPUS_PER_TASK=$cpus")
fi

if [[ "$engine" == "docker" ]]; then
    runtime=(docker run --rm --init --user "$(id -u):$(id -g)" --workdir "$container_workdir" --entrypoint conda)
    [[ -n "$cpus" ]] && runtime+=(--cpus "$cpus")
    for value in "${common_env[@]}"; do runtime+=(-e "$value"); done
    runtime+=(-v "$results:/results" -v "$cache:/cache")
    for value in "${mounts[@]}"; do runtime+=(-v "$value"); done
    runtime+=("$docker_image" run --no-capture-output -n evopt python "${command_args[@]}")
else
    runtime=(apptainer exec --cleanenv --pwd "$container_workdir")
    for value in "${common_env[@]}"; do runtime+=(--env "$value"); done
    runtime+=(--bind "$results:/results" --bind "$cache:/cache")
    for value in "${mounts[@]}"; do runtime+=(--bind "$value"); done
    runtime+=("$image" conda run --no-capture-output -n evopt python "${command_args[@]}")
fi

echo "Results are available on the host at: $results"
exec "${runtime[@]}"
