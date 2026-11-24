# EV Charger Placement with Traffic Equilibrium Models

Code for *Congestion Reduction in EV Charger Placement Using Traffic Equilibrium
Models* (IEEE T-ITS). For a given set of candidate charger sites, the pipeline
compares charger placements under two equilibrium models:

1. **Congestion game (CG):** link-space Wardrop equilibrium solved as a convex
   program with per-link BPR delays.
2. **Queue model:** an atomic, mesoscopic spatial-queue simulator with charging
   stations, solved by better-response over a restricted route set.

The steps are:
1. Build the network from OpenStreetMap.
2. Calibrate the BPR delays from link-level queue simulations.
3. Solve the CG for every candidate placement.
4. Recover route flows.
5. Run the queue better-response search.
6. Compare greedy, single-swap and exhaustive placement under both models.

## Layout

| Path | Contents |
|---|---|
| `pipeline.py` | Runs one configuration end to end |
| `run_suite.py` | Runs a manifest of configurations; summarizes results and exports ZIP bundles |
| `paper_tables.py`, `scripts/plot_*.py` | Paper tables and figures from finished runs |
| `src/` | Network construction and pruning, CG optimization, configuration, reporting |
| `queue_sim/` | Queue simulator, BPR data generation, Nash search, placement comparison |
| `configs/paper_r2/` | Final experiment suite (see its README) |
| `configs/revision1/` | Exact configurations of the Revision-1 results, kept for provenance |
| `scripts/run_container.sh` | Docker/Apptainer launcher; `run_slurm_suite.sh`, `submit_suite.sh` for Slurm |
| `tests/` | Unit and regression tests (`pytest -q tests`) |

## Running

The supported way to run is the published container image with the repository
mounted as the workspace. Python changes don't require rebuilding the image.

```bash
git clone --branch rebuttal-scalable-pipeline \
  https://github.com/YasinSonmez/EV-Charger-Optimization.git evopt && cd evopt
apptainer pull ev-charger-placement-6de1600.sif docker://ausar/ev-charger-placement:6de1600
SIF="$PWD/ev-charger-placement-6de1600.sif"

# 1. Five-minute end-to-end check
scripts/run_container.sh --engine apptainer --image "$SIF" --mode workspace \
  --config configs/paper_r2/sanity.json \
  --results "$PWD/results/sanity" --cache "$PWD/osm-cache" --cpus 16

# 2. Full paper suite, then the results ZIP (rerun the same command to resume)
scripts/run_container.sh --engine apptainer --image "$SIF" --mode workspace \
  --manifest configs/paper_r2/suite.json \
  --results "$PWD/results/paper_r2" --cache "$PWD/osm-cache" --cpus 55 \
  --resume --continue-on-failure \
  --bundle-profile analysis --export-bundle paper_r2_analysis.zip
```

The ZIP is written to `results/paper_r2/`. To rebuild it later without rerunning
anything, use the same command with `--summarize` added.

For a native run (not used for the paper), install `environment.yml` and build
the shortest-path library with `./build_liblsp.sh`. Native library versions can
select different candidate sites than the pinned image, so reproduce paper
numbers in the container.
