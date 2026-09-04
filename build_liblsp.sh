#!/usr/bin/env bash
# Build liblsp.so (the C++ shortest-path library) and install it into dlls/.
# Run this ON THE LINUX MACHINE, from the project root. See README_LINUX.md.
set -euo pipefail

REPO=https://github.com/cb-cities/sp.git
COMMIT=475298f4570109378a57b4e592f01b8a26fe0c90
DEST="$(cd "$(dirname "$0")" && pwd)/dlls"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

echo ">> fetching $REPO at $COMMIT"
git init -q "$WORK/sp"
git -C "$WORK/sp" remote add origin "$REPO"
git -C "$WORK/sp" fetch -q --depth 1 origin "$COMMIT"
git -C "$WORK/sp" checkout -q --detach FETCH_HEAD

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
