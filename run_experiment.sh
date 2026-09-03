#!/usr/bin/env bash
# Run a single experiment from config file.
# Usage:
#   ./run_experiment.sh config.json              # local
#   ./run_experiment.sh --docker config.json      # via Docker
#   ./run_experiment.sh --docker --build config.json  # build Docker first
set -euo pipefail

DOCKER_IMAGE="ev-charger-opt"
DOCKER_BUILD=false
USE_DOCKER=false
CONFIG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --docker) USE_DOCKER=true; shift ;;
        --build) DOCKER_BUILD=true; shift ;;
        --image) DOCKER_IMAGE="$2"; shift 2 ;;
        --help)
            echo "Usage: $0 [--docker] [--build] [--image NAME] CONFIG"
            echo "  --docker   Run via Docker container"
            echo "  --build    Build Docker image before running"
            echo "  --image    Docker image name (default: ev-charger-opt)"
            exit 0
            ;;
        *) CONFIG="$1"; shift ;;
    esac
done

if [ -z "$CONFIG" ]; then
    echo "Error: config file required"
    echo "Usage: $0 [--docker] [--build] config.json"
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo "Error: config file not found: $CONFIG"
    exit 1
fi

CONFIG_ABS=$(realpath "$CONFIG")
CONFIG_NAME=$(basename "$CONFIG_ABS")

if [ "$USE_DOCKER" = true ]; then
    if [ "$DOCKER_BUILD" = true ]; then
        echo "Building Docker image..."
        docker build -t "$DOCKER_IMAGE" .
    fi

    RESULTS_DIR=$(pwd)/results
    mkdir -p "$RESULTS_DIR"

    echo "Running via Docker: $CONFIG_NAME"
    docker run --rm \
        -v "$CONFIG_ABS:/app/$CONFIG_NAME:ro" \
        -v "$RESULTS_DIR:/app/results" \
        "$DOCKER_IMAGE" \
        --config "/app/$CONFIG_NAME"

    echo "Results saved to: $RESULTS_DIR"
else
    echo "Running locally: $CONFIG_NAME"
    python3 pipeline.py --config "$CONFIG_ABS"
fi
