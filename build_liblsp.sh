#!/usr/bin/env bash
# Build liblsp.so (the C++ shortest-path library) and install it into dlls/.
# Run this ON THE LINUX MACHINE, from the project root. See README_LINUX.md.
set -euo pipefail

BRANCH=dataframe_2026     # NOT the default branch: only this one has creategraph()
REPO=https://github.com/cb-cities/sp.git
DEST="$(cd "$(dirname "$0")" && pwd)/dlls"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

echo ">> cloning $REPO ($BRANCH)"
git clone -q --depth 1 --branch "$BRANCH" "$REPO" "$WORK/sp"

echo ">> compiling"
g++ -std=c++14 -O3 -shared -fPIC \
    -I"$WORK/sp/include" -I"$WORK/sp/external" \
    "$WORK/sp/src/graph.cc" "$WORK/sp/src/py.cc" \
    -o "$WORK/liblsp.so"

echo ">> verifying exported symbols"
missing=()
for sym in simplegraph readgraph creategraph dijkstra update_edge writegraph origin distance parent clear; do
    nm -D --defined-only "$WORK/liblsp.so" | grep -qE " T $sym\$" || missing+=("$sym")
done
if [ ${#missing[@]} -ne 0 ]; then
    echo "ERROR: library is missing symbols: ${missing[*]}" >&2
    exit 1
fi

mkdir -p "$DEST"
install -m 0755 "$WORK/liblsp.so" "$DEST/liblsp.so"
echo ">> installed $DEST/liblsp.so"
file "$DEST/liblsp.so"
echo
echo "Now run:  python3 -c 'import sim_package; print(\"OK\")'"
