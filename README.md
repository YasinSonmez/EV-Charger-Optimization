# EV Charger Optimization — Implementation Report

## Executive summary

This repository evaluates EV-charger locations on an OpenStreetMap road
network. Its intended experiment is:

1. simplify a directed road graph;
2. obtain link delay data from a microscopic spatial-queue simulator and fit
   one BPR delay curve per link;
3. solve a continuous congestion-equilibrium problem for each charger
   placement;
4. reconstruct a finite set of routes from the optimized link flows;
5. use those routes as fixed strategies in a queue-simulation Nash/better-
   response loop; and
6. compare greedy placement, greedy plus swaps, and exhaustive placement.

The main entry point is [`pipeline.py`](pipeline.py). The pipeline now uses a
canonical network artifact, typed F1/F2 demand contracts, explicit link IDs,
joint multi-OD queue assignments, deterministic named seeds, and structured
run manifests. The real queue/BPR stages still require the simulator library;
offline tests use deterministic fixtures.

The original implementation had these confirmed problems, which are now
addressed by the refactor:

- the queue Nash stage passed an incompatible pruning keyword;
- BPR generation derived filenames from the hard-coded `college_park` name;
- parallel road edges were collapsed by later `nx.DiGraph` and pair lookups;
- the CVXPY model is a user-equilibrium/Beckmann-potential model, while the
  placement ranking uses a different total-travel-time expression;
- configuration flags such as `random_seed`, `queue_simulation.enabled`, and
  `road_filter.enabled` were not consistently honored; and
- the historical end-to-end test depends on a deleted configuration and an
  environment-specific simulator binary.

The measured paper-scale run produced all structural artifacts, but its queue
better-response searches hit the 200-iteration bound for all six charger
configurations without reaching the requested 1% relative gap. Its placement
numbers are therefore diagnostic/provisional, not validated Nash-equilibrium
results. See the detailed [paper-scale report](results/2026-07-25_19-44-24_n=3_chargers=2/report.md).

## Repository map

| Area | Implementation | Responsibility |
|---|---|---|
| Orchestration | [`pipeline.py`](pipeline.py) | Five-step run, caching, plots, summaries |
| Configuration | [`src/config.py`](src/config.py) | Defaults, JSON conversion, limited validation |
| Shared contracts | [`src/contracts.py`](src/contracts.py) | Typed demand, deterministic seeds, timing, stable serialization |
| Network artifact | [`src/network_artifact.py`](src/network_artifact.py) | `network_manifest.json`, `nodes.csv`, `edges.csv`, hash validation |
| Road network | [`src/road_network.py`](src/road_network.py) | OSM download, filtering, contraction, DataFrame export |
| Graph cache | [`src/graph_cache.py`](src/graph_cache.py) | Pickled OSM graph cache |
| BPR fitting | [`queue_sim/bpr_data_generator.py`](queue_sim/bpr_data_generator.py), [`src/model_fitter.py`](src/model_fitter.py) | Simulation samples and curve fitting |
| Congestion model | [`src/traffic_optimizer.py`](src/traffic_optimizer.py) | Route generation, CVXPY equilibrium, optional SciPy solver |
| Placement search and analysis | [`src/utils.py`](src/utils.py) | Greedy, swap, exhaustive search, route reconstruction, plots |
| Microscopic simulation | [`queue_sim/queue_model_EV.py`](queue_sim/queue_model_EV.py), [`queue_sim/runner_EV.py`](queue_sim/runner_EV.py) | Spatial queue, intersections, EV stations |
| Queue experiments | [`queue_sim/find_nash.py`](queue_sim/find_nash.py), [`queue_sim/comparison.py`](queue_sim/comparison.py) | Better-response assignment and placement comparison |
| Benchmarks | [`benchmarks/run_benchmarks.py`](benchmarks/run_benchmarks.py) | Bounded fixture scaling and real-network artifact timing |
| Tests | [`tests/`](tests/) | Configuration, BPR formula, helpers, limited integration coverage |

## 1. How the methodology works

### 1.1 Network and demand representation

The configuration uses a bounding box `[north, south, east, west]`, integer
node IDs after network cleaning, candidate charger node IDs, and OD demand of
the form:

```json
"od_demand": {"origin,destination": [non_charging_demand, charging_demand]}
```

For example, `[60, 120]` means 60 non-charging vehicles and 120 vehicles that
must charge. `Config.get_od_demand_tuples()` converts string keys such as
`"0,1"` to `(0, 1)`.

`RoadNet` first holds a `networkx.MultiDiGraph`, then converts the cleaned
graph to node and link DataFrames with sequential `node_id` and `link_id`
values. Link records contain endpoints, length, lanes, maximum speed, capacity,
highway type, and WKT geometry.

Every pipeline run writes `network/network_manifest.json`, `network/nodes.csv`,
`network/edges.csv`, and (when stage maps are available) `network/stage_maps.json`.
The manifest contains the stable `network_hash`, source/filter settings,
source-to-canonical ID mappings, every cleaning-stage count, and the final
graph size.
BPR, CG, route recovery, and queue stages use this artifact rather than
rebuilding OSM data. Road-edge identity is the persisted `link_id` plus the
original MultiDiGraph edge key. A `(u,v)` lookup is only accepted when it is
unique.

Legacy demand shorthand remains supported, but it is normalized internally to
typed F1/F2 records. F1 vehicles never charge; F2 vehicles charge exactly once.
The queue stage receives one demand table containing all OD/type classes and
therefore evaluates a shared multi-OD network state.

### 1.2 Link delay model

Each link is represented by a fitted BPR function:

\[
t_l(x_l) = FFT_l\left(1+a_l\left(\frac{x_l}{C_l}\right)^{b_l}\right),
\]

where `x_l` is link flow, `FFT_l` is free-flow travel time, `C_l` is fitted
capacity, and `a_l`, `b_l` control congestion growth. The fitting process is
described in [How simulation is used](#3-how-simulation-is-used).

### 1.3 Flow formulation used by CVXPY

For each candidate placement, `Network.optimize_with_cvxpy()` creates
continuous link-flow variables:

- `x_nc[i]`: non-charging flow for OD pair `i`;
- `x_plus[i,c]`: charging flow from OD origin `i` to charger `c`;
- `x_minus[i,c]`: charging flow from charger `c` to the OD destination; and
- `q_c[i]`: amount of charging demand for OD pair `i` assigned to charger `c`.

Flow conservation is imposed with a node-link incidence matrix. Non-charging
flow must move from origin to destination. Charging demand is split across
chargers, and each charger receives the same amount that it sends onward.
All vehicle classes share the same aggregate link flow in the BPR delay.

The optimization objective is the Beckmann potential:

\[
\min_x \sum_l FFT_l\left[x_l +
\frac{a_l C_l}{b_l+1}\left(\frac{x_l}{C_l}\right)^{b_l+1}\right]
 + L_c\sum_c \hat{x}_c,
\]

where `hat_x_c` is charger throughput and `L_c` is the configurable
`charger_self_link_length` parameter. This is the standard separable
potential whose optimum represents a Wardrop/user equilibrium under the
specified link costs; it is not directly the total system-travel-time
objective.

After solving, the code separately evaluates:

\[
TT = \sum_l x_l t_l(x_l) + L_c\sum_c \hat{x}_c,
\]

and uses this `travel_time_obj` to rank charger placements. Thus the inner
flow is obtained by minimizing the integral objective, but the outer placement
ranking is based on total delay. That distinction should be stated explicitly
in any scientific interpretation.

The code also appends one synthetic self-link per charger to the DataFrame.
Those self-links are not added to the NetworkX graph used for path discovery;
their cost is handled separately in the CVXPY formulation. The optional SciPy
path-flow formulation instead includes them in generated charging routes.

### 1.4 Placement search

`outer_optimization()` evaluates placements in three phases:

1. **Greedy construction:** choose the best one-charger placement, then add the
   best remaining candidate, repeating until `num_chargers` is reached.
2. **Single-swap refinement:** replace one selected charger with one unselected
   candidate and evaluate the resulting placements. The CG implementation
   evaluates the one-swap list once; the queue comparison repeats improving
   swaps until no improvement is found.
3. **Exhaustive search:** if
   `calculate_on_all_possible_positions` is true, evaluate every
   `C(num_candidates, num_chargers)` combination not already seen.

Every placement is solved independently and stored in `grids`. The final
result is selected by the minimum `travel_time_obj` across all stored grids.
With exhaustive search enabled this normally includes all target-size
placements; with it disabled, the code can compare partial greedy placements
against the requested final-size placement and return an unintended size.

### 1.5 Route reconstruction

The CVXPY solution is a link-flow solution, not a route-flow solution. For
analysis and queue simulation, `reconstruct_route_flows()` generates up to
`paths_per_od` shortest simple paths for each OD pair and up to
`paths_per_oc_cd` paths for each origin-to-charger and charger-to-destination
leg. It then solves a nonnegative least-squares-style CVXPY problem that:

- preserves non-charging and charging OD totals;
- attempts to match the optimized link-flow vector; and
- optionally attempts to match charger self-link flows.

Routes are flattened and sorted by reconstructed flow. The configured `k`
largest routes are used for coverage, MAE, RMSE, maximum error, and
correlation plots, and the top `K` are passed to the queue experiment.

This is an approximation: a finite shortest-path candidate set is being used
to explain a continuous link flow. A high correlation or coverage does not
prove that the reconstructed route assignment is the unique or exact
equilibrium assignment.

## 2. How pruning works

The cleaning implementation is in [`src/road_network.py`](src/road_network.py)
and is documented in more detail in [`docs/network_pruning.md`](docs/network_pruning.md).

### Stage 01 — raw graph

OSMnx downloads or loads a cached drivable graph. The graph is a directed
`MultiDiGraph`; every directed OSM edge and its geometry are retained.

### Stage 02 — highway filtering

If `highway_types` is non-empty, each edge is inspected. If its `highway`
attribute is not in the allow-list, the edge is removed, followed by removal
of isolated nodes. The canonical config keeps motorway, trunk, primary, and
secondary roads. This is a hard filter, so discarded residential/service
roads cannot be used by later routing.

### Stage 03 — close-node contraction

An undirected auxiliary graph is built from edges shorter than the hard-coded
default threshold of 30 m. Each connected component with at least two nodes is
replaced by a synthetic centroid node. External incoming and outgoing edges
are redirected to the centroid, and their geometry endpoints are snapped to
the centroid. Self-loops created by redirection are removed.

This treats tight ramp/interchange structures as one macroscopic routing
decision. It is a modeling approximation: internal interchange choices and
their delays are removed.

### Stage 04 — degree-2 chain contraction

Up to three iterations remove nodes whose union of predecessors and successors
contains exactly two unique neighbors. For every compatible incoming/outgoing
edge pair, the implementation creates a merged edge:

- length is summed;
- geometry is concatenated through the removed node;
- maximum speed is combined using a length-weighted harmonic mean;
- lane count is the minimum of the two lane counts; and
- `name` and `ref` are discarded.

Parallel edge combinations are retained at this stage. Later routing carries
the canonical `link_id` sequence; a node-pair adapter is used only when the
pair is unique, and ambiguous legacy routes fail instead of selecting the
first edge.

### Final SCC operation

After the cleaning stages, `_keep_largest_scc()` retains only the largest
strongly connected component. The resulting `05_final_scc` stage is recorded
after extraction and is the exact graph serialized into the canonical network
artifact and consumed by optimization.

### What the pruning configuration actually controls

`merge_chains`, `contract_threshold`, `prune_dead_ends`, and
`road_filter.enabled` are wired through the cleaning pipeline. Unsupported
`suppress_t_junctions=true` configurations fail validation rather than being
silently ignored. Every stage has counts and map data, including the final
SCC stage.

## 3. How simulation is used

There are two distinct uses of simulation.

### 3.1 Microscopic simulation for BPR calibration

The default `historical_artifact_compatible` mode restores the preferred
historical experiment while using the current canonical network. For every
canonical road link it:

1. identify a geometrically “straight-ahead” incoming link and outgoing link;
2. set the temporary OD to the start of the incoming link and end of the
   outgoing link;
3. run exactly 25 historical absolute flow levels from 1 through 250;
4. create isolated per-link/per-flow simulation directories, force all
   vehicles through `sa_in -> target_link -> sa_out`, and run the
   spatial-queue model;
5. record the simulator-measured target-link flow and travel time; and
6. fit the resulting `(flow, travel_time)` pairs with the historical
   four-free-parameter BPR model.

If a link has no straight-ahead predecessor or successor, historical mode
uses a worker-local synthetic source/sink context and still measures the
canonical target link in the queue simulator. Synthetic links are calibration
scaffolding only: they are not added to the canonical artifact or downstream
route graph. If that calibration fails, the configured legacy fallback creates
physical-property proxy observations with the same 25 x-values and marks the
link as `proxy`; it never silently claims those observations were simulated.
With the relaxed pipeline defaults, weak-trend and low-quality
observations still receive a nonlinear attempt and finite solutions are
labeled `full_relaxed`; their `honest_R2` is preserved. Set
`fit_screening=legacy` and `accept_low_r2=false` to reproduce the historical
constant-sentinel behavior for those cases. Numerical failures and unavailable
contextual routes remain explicitly labeled rather than being presented as
successful nonlinear fits.

The queue model is time-stepped. It uses virtual source links, link travel
times and storage, lane-based sending/receiving capacities, intersection
conflict rules, and random movement when capacity is tight. A simulation stops
when all agents arrive or the 10,801-second horizon is exhausted.

Historical mode allows the explicitly configured legacy proxy fallback. The
strict alternative is `mode=capacity_fraction_strict`; it uses capacity
fractions, fixed canonical FFT/capacity references, and rejects any non-full
fit. Both modes write `bpr_manifest.json` with the canonical `network_hash`,
mode, seed, sample count, fallback counts, and per-link timings.

`TrafficModelFitter` fits each link in a `ProcessPoolExecutor` using bounded
`scipy.optimize.curve_fit`. Historical mode uses initialization `[1,1,1,1]`
and bounds `[0,0.8,1,0]` to `[∞,5,1000,∞]`; strict mode uses the modern
fixed-reference fitter. The historical committed artifact at `37eab33` is a
regression fixture only: its 122-link IDs and hash cannot enter optimization
on a different current canonical network.

### 3.2 Queue simulation for route assignment and placement comparison

After CG optimization, the queue stage reads the reconstructed routes. For
each placement it:

1. keeps the top `K` routes;
2. converts continuous route flows to integer vehicle counts with largest
   remainder rounding;
3. runs `NUM_ITERS` independent simulations for the current assignment;
4. measures route “cost”; and
5. moves one vehicle from a currently used high-cost route to a low-cost route
   while the cost spread exceeds `THRESH`, for at most 200 iterations.

The queue implementation measures route travel time from simulated agents,
uses free-flow costs for unused routes, and applies the relative paper
criterion:

```text
max_used_route_cost - min_available_route_cost
    <= alpha * min_available_route_cost
```

It is still a stochastic, bounded better-response search. A result with
`converged=false` is not a formally verified Nash equilibrium and must be
reported as nonconverged rather than treated as an optimization result.

The final comparison uses the same route assignments and runs:

- a greedy placement experiment over candidate nodes;
- optional repeated single-swap improvement; and
- exhaustive simulation for every target-size combination.

Each placement/replicate receives a deterministic seed derived from the
replicate and placement, which reduces Monte Carlo asymmetry inside this
comparison. The resulting “suboptimality” is:

```text
(best_greedy_average - best_exhaustive_average)
/ best_exhaustive_average * 100
```

It is a comparison of the implemented stochastic simulator and fixed route
assignments, not a proof that greedy is suboptimal for the original continuous
equilibrium problem.

The simulator library is loaded through [`queue_sim/interface.py`](queue_sim/interface.py).
The repository includes a macOS `liblsp.dylib`; Linux can use the
[`build_liblsp.sh`](build_liblsp.sh) script to build `liblsp.so`, and the
Dockerfile does so. Therefore “macOS-only” is true for the bundled binary, not
for the Python interface as currently written.

## 4. How a configuration run proceeds

The actual `run_pipeline(config_path)` sequence is:

### Step 0 — load and clean the network

1. Parse JSON with `Config.from_json()` and perform basic checks.
2. Create the experiment directory and save the normalized configuration.
3. Build one shared `RoadNet('pipeline')` from OSMnx/cache.
4. Apply filtering, contraction, chain merging, and final SCC extraction.
5. Assign deterministic canonical node/link IDs only after the final SCC.
6. Write `network/nodes.csv`, `network/edges.csv`, and
   `network/network_manifest.json`; the manifest contains the stable hash.
7. Save stage counts and maps, including the final SCC, to
   `plots/pruning_phases.png`.

### Step 1 — load or fit BPR curves

The code checks, in order:

1. a BPR cache/CSV whose manifest matches the current canonical hash, mode,
   flow sweep, seed, and fitter version;
2. no unmanifested legacy cache is accepted for a canonical-artifact run;
3. queue-simulator generation using the exact canonical node/edge files; or
4. fail explicitly if no valid source is available.

Generated BPR samples are partitioned by `link_<link_id>`, and each worker
processes a complete flow sweep with deterministic sub-seeds. Fail-fast is the
default; proxy rows require explicit degraded mode and are marked in
`bpr_manifest.json`, which also records per-link sweep timings.

### Step 2 — solve charger placements

1. Convert OD keys to integer tuples.
2. Run `outer_optimization()` with the shared network and BPR parameters.
3. Solve the selected CVXPY or SciPy inner model for every placement visited by
   greedy, swap, and exhaustive phases.
4. Save per-placement heatmaps and the aggregate
   `all_optimization_results.pkl`.
5. Reconstruct routes and run top-k plus parameter-sweep analysis for each
   placement.
6. Attach the canonical `network_hash`, seed, and solver metadata to the
   optimization artifact.

Route-library K is applied per `(origin, destination, vehicle class)` group.
This prevents a global top-K truncation from dropping every F2 route or every
route for a second OD pair.

### Step 3 — find queue assignments

If queue simulation is enabled and the library loads, copy the canonical
artifact into queue inputs, build one shared OD table containing every OD/type
class, and run the paper's relative-gap better-response loop in a process pool.
Each Nash iteration evaluates all OD/type route groups jointly. Save
`queue/NE_path_assignments.pkl`, `queue/queue_manifest.json`, convergence CSV,
and plot. `queue_manifest.json` records per-configuration, per-iteration
timings and failure reasons.

### Step 4 — compare greedy and exhaustive placements

Run `N` Monte Carlo replicates for greedy and exhaustive placement. Replicates
are parallelized, while the greedy search itself remains sequential. Each
replicate and charger configuration receives a private output directory. Save
`queue/comparison_results.json` and, when both CG and queue results exist,
`plots/objective_comparison.png`.

### Step 5 — aggregate

Write `run_summary.txt`, the generated experiment `report.md`,
`experiment_summary.json`, `plots/timing_breakdown.png`, and
`sanity_check.json`. The final sanity check validates the artifact hash,
optimization link/route IDs, finite objectives, queue demand conservation,
queue/comparison hashes, and required output files before the run is declared
successful.

The `--network-only` mode runs the network stage, writes the canonical artifact,
validates its hash, and generates the final pruning plot. It intentionally does
not run BPR, CG, or queue simulation.

## 5. Configuration reference

The checked-in [`config.json`](config.json) contains:

| Field | Meaning in the current code |
|---|---|
| `coordinates` | `[north, south, east, west]` OSM bounding box |
| `num_chargers` | Target number of charger nodes |
| `possible_charger_positions` | Candidate node IDs after cleaning |
| `od_demand` | `{"o,d": [non_charging, charging]}` demand |
| `use_cvxpy` | Select CVXPY instead of the legacy SciPy path-flow solver |
| `use_derivatives` | Used by SciPy only; no effect for CVXPY |
| `max_iter` | Solver iteration limit where supported; the actual solver, options, and status are recorded in the run manifest |
| `single_swap` | CG one-swap phase |
| `calculate_on_all_possible_positions` | Add all target-size combinations |
| `route_analysis.k_values` | Route counts used for reconstruction diagnostics |
| `queue_simulation.K` | Route count passed to queue assignment |
| `queue_simulation.ALPHA` | Relative paper Nash-gap stopping threshold |
| `queue_simulation.NUM_ITERS` | Replicates per better-response iteration |
| `queue_simulation.N` | Replicates per placement in final comparison |
| `queue_simulation.SIMULATION_HORIZON` | Maximum simulator time; timeout is recorded as a failed result |
| `queue_simulation.MAX_NE_ITERATIONS` | Maximum relative-gap better-response iterations |
| `queue_simulation.*_CAPACITY` | Queue simulator station capacities |
| `pipeline.bpr_generation.timeout` | Per-link BPR worker timeout in seconds |
| `pipeline.bpr_generation.mode` | `historical_artifact_compatible` (default) or `capacity_fraction_strict` |
| `pipeline.bpr_generation.num_samples` / `max_flow` | Historical mode defaults to 25 measured samples over flows 1–250 |
| `pipeline.bpr_generation.fit_validation` | `parameter_complete` for labeled historical fallbacks or `full` for strict fits |
| `pipeline.bpr_generation.fallback_policy` | `legacy_proxy_and_constant` or `none` |
| `pipeline.bpr_generation.flow_fractions` | Capacity fractions used for BPR calibration |
| `pipeline.bpr_generation.capacity_source` | `simulator` or `artifact` capacity source for `C` |
| `pipeline.bpr_generation.capacity_per_lane` | Simulator capacity in vehicles/hour/lane |
| `pipeline.bpr_generation.calibration_window_hours` | Converts hourly-equivalent BPR flow to simulator agent count |
| `pipeline.bpr_generation.require_full_fit` | Rejects the BPR stage if any canonical road link is not a full fit |
| `pipeline.bpr_generation.missing_context_policy` | `synthetic_boundary` adds worker-local source/sink context; `proxy` or `fail_fast` are explicit alternatives |
| `pipeline.bpr_generation.synthetic_context_capacity_multiplier` / `synthetic_context_length_m` | Capacity and length controls for calibration-only synthetic boundary links |
| `pipeline.bpr_generation.min_r2` | Minimum full-fit R² in strict mode |
| `pipeline.bpr_generation.fit_screening` | `legacy` applies historical correlation/variation gates; `none` attempts the nonlinear fit for every finite observation vector |
| `pipeline.bpr_generation.accept_low_r2` | Retains finite nonlinear fits below `min_r2` as `full_relaxed`; the honest R² remains visible and is not improved artificially |
| `pipeline.bpr_generation.correlation_threshold` / `variation_ratio_threshold` | Thresholds used only when `fit_screening=legacy` |
| `pipeline.cg_fit_policy` | `allow_degraded` records active relaxed/proxy/constant links; `reject_proxy_or_constant` or `validated_only` enforce stricter CG inputs |
| `pipeline.skip_bpr_fitting` | Cache-only switch; fails if no valid BPR cache/CSV exists |
| `pipeline.skip_cg_optimization` | Requires a pre-existing pickle in the newly created experiment directory |
| `pipeline.skip_queue_simulation` | Skips queue stages when true |
| `pipeline.random_seed` | Global seed with deterministic named worker streams |
| `pipeline.parallel_workers` | Global worker default; `null` uses `os.cpu_count()`, while stage-specific worker settings override it |
| `pipeline.bpr_generation.workers` | BPR link-worker count; one worker owns a complete flow sweep |
| `pipeline.bpr_generation.fit_workers` | BPR curve-fitting worker count; `1` enables bounded serial CI mode |
| `pipeline.bpr_generation.save_fit_plots` | Whether per-link BPR diagnostic plots are written |
| `road_filter.highway_types` | OSM highway allow-list |
| `road_filter.merge_chains` | Enables stage 04 |
| `road_filter.enabled` | Enables/disables the cleaning transformations while retaining final SCC extraction |

For queue runs, `queue_simulation.K` must also appear in the route-analysis
`k_values`; configuration validation rejects inconsistent values.

## 6. Problems and bugs

### Confirmed execution and correctness problems

| Status | Historical issue | Current behavior |
|---|---|---|
| Resolved | Queue NE passed an incompatible pruning keyword | Queue code consumes the canonical artifact and no longer rebuilds the OSM network |
| Resolved | BPR filenames depended on `college_park` | BPR receives exact canonical node/edge files and writes isolated link-sweep directories |
| Resolved | Cleaning flags were ignored | `enabled`, chain merging, contraction threshold, and dead-end pruning are explicit; unsupported T-junction suppression fails clearly |
| Resolved | Parallel edges were collapsed or selected by first match | `link_id` and edge keys are persisted; ambiguous pair lookup raises an error |
| Resolved | Queue used only the first OD pair | F1/F2 demand and route assignments are processed jointly for every OD pair |
| Resolved | Queue enablement and seed handling were inconsistent | `queue_simulation.enabled` is honored and named deterministic seed streams are recorded |
| Resolved | Shared simulator output paths collided between workers | BPR links, Nash iterations, and placement replications receive isolated output directories |
| Resolved | Historical BPR required a straight-ahead predecessor/successor for every link | Historical mode uses the old contextual route where available and labeled physical-property proxy data otherwise; strict probe mode remains separate |
| Resolved | Global top-K route truncation could remove an entire OD/type class | K is selected independently for each OD and vehicle class |
| Resolved | Active-link flow vectors produced invalid/missing heatmaps | Plot writers expand active-link vectors to canonical full-link vectors |
| Resolved | Process-based BPR fitting failed in restricted environments | Fitting has configurable workers and a recorded serial fallback; numerical fit failures are handled as constant models |
| Remaining | The simulator can still be unavailable or terminate before all arrivals | Real queue results require a simulator preflight; integration output must record failures and environment provenance |
| Remaining | CG is still a continuous equilibrium approximation with a finite reconstructed route library | Route coverage, solver status, and reconstruction residuals must be reported with every scientific result |
| Remaining | The bounded queue better-response loop can fail to reach `alpha` before `MAX_NE_ITERATIONS` | Results are retained with `status=nonconverged`; strict production comparisons should refuse to claim equilibrium results |

### Modeling and numerical limitations

- The CVXPY model has no explicit road-capacity constraint; capacity only
  affects the BPR curve. Charger capacity and entrance/exit capacity exist in
  the queue simulator but not in the CG model.
- The queue station charging time and CG charger penalty are separate model
  parameters; they are configurable but are not automatically calibrated from
  one empirical station-delay curve.
- The route reconstruction is limited to generated shortest simple paths and
  solves an approximate flow-matching problem. It can omit routes that carry
  valid flow in the continuous model.
- The BPR fitter can still produce a constant model for weak data, but it now
  reports the constant-model score and `fit_status`; fail-fast generation is
  the default.
- Maximum-speed parsing in `rearrange_data()` assumes a digit-only token. OSM
  values such as `signals`, decimals, units other than the expected form, or
  unusual lists can raise errors or be misread. Lane parsing has similar
  assumptions.
- `Network` now copies the graph as well as node and edge DataFrames.
- Stage counts and pruning maps include the final SCC and are tied to the
  canonical artifact hash.
- Randomness is seeded globally and per worker; solver numerical differences
  across environments can still remain within tolerance.
- Relative paths and the import side effect in `queue_sim/__init__.py`
  (changing the process working directory) make library use from another
  application fragile.

## 7. Is the code optimized? How to make it much faster

The code has some useful optimizations—OSM graph caching, process-based BPR
fitting, shortest-path warm starts, deterministic placement seeds, and process
pools for queue experiments—but the dominant loops are still expensive and
rebuild too much state.

### Highest-value changes

1. **Make one canonical network artifact.** Build the cleaned network once,
   serialize nodes/edges and a stable network hash, and pass those exact files
   to BPR, CG, and queue stages. This removes repeated OSM work, naming drift,
   and mismatched node IDs.
2. **Preserve edge identity.** Represent a route edge as `(u, v, key)` or a
   sequential link ID everywhere. Keep a precomputed mapping from node pairs to
   link IDs only when the pair is unique. This is both a correctness fix and a
   speedup over repeated DataFrame scans.
3. **Precompute route incidence.** Generate candidate paths once per OD and
   charger leg, build sparse route-link matrices, and reuse them across
   placements and reconstruction. Current nested loops repeatedly scan every
   link for every route segment.
4. **Avoid rebuilding CVXPY models for each placement.** Use sparse matrices,
   CVXPY `Parameter`s, warm starts between nearby placements, and a shared
   model where possible. If placement is the only discrete decision, consider
   a separate outer discrete optimizer using cached route costs rather than
   recompiling the full continuous model.
5. **Parallelize BPR experiments at the right level.** The generator batches
   one complete flow sweep per link worker, reuses immutable canonical inputs
   and one resettable Runner per link, and writes unique link directories.
6. **Replace repeated plotting and pickling in the hot path.** Generate
   diagnostics only for selected links/configurations, and write one compact
   structured result at the end or at explicit checkpoints.
7. **Make queue experiments in-memory and collision-free.** Avoid CSV reloads
   for every replicate, give each worker a private output path, and reuse a
   prebuilt topology. Simulate only finalists when the goal is ranking, then
   increase replicates for close candidates.
8. **Use robust early screening.** Evaluate cheap free-flow/lower-bound scores
   first, prune placements that cannot beat the incumbent, then run full CG or
   queue simulation only for survivors. This is safer than the current
   “pruning” terminology, which only prunes the road graph and does not prune
   the placement search.

### Complexity hotspots

- `Network.get_od_pairs_and_demands()` repeatedly filters a pandas DataFrame to
  find a link for every route segment.
- Route reconstruction and parameter sweeps scan all links for every route
  segment and repeat path generation for every parameter pair and placement.
- The SciPy Hessian materializes `diag(...)` and dense matrix products, which
  becomes expensive as route count grows.
- Each CVXPY placement creates variables for every OD, charger, and link,
  including many zero/unusable link variables.
- Queue comparison performs `N × (greedy candidates + swaps + exhaustive
  combinations)` full microscopic simulations, each with up to 10,801 time
  steps.

## 8. Remaining limitations and validation status

The high-priority correctness blockers are addressed, but these limitations
remain material for scientific-scale runs:

- The bundled simulator and OSM/Overpass access are environment-dependent.
  The deterministic contract suite can run offline; a complete paper-scale
  queue result requires a working simulator binary and input access.
- Configuration validates node IDs after network construction, but a future
  preflight should also check OD reachability, route-library coverage, and
  charger feasibility before expensive optimization.
- The CVXPY model does not enforce physical road capacity or station queues;
  those effects are represented in BPR/queue stages and are not automatically
  calibrated to one shared station-delay model.
- Dependencies are not pinned by a lockfile, and the Docker base image is
  still mutable. Solver results therefore require numerical-tolerance rather
  than bit-for-bit comparison across environments.
- The finite shortest-route library remains an approximation of the
  continuous link equilibrium. Reconstruction residuals and route coverage
  should be reported before treating queue comparisons as definitive.
- Exhaustive placement remains combinatorial. The implementation parallelizes
  independent Monte Carlo replications and BPR link workers, but deliberately
  keeps greedy placement sequential and does not parallelize charger decisions.

The historical end-to-end test is skipped because `config_test.json` is absent.
The current offline suite is `25 passed, 1 skipped` after adding artifact, queue,
route-library, BPR-boundary, and orchestrator checks. A real two-OD smoke run
using the supplied paper bounding box completed all stages in about 24 seconds
with 2 BPR samples, one charger candidate, and one queue replication. It
produced a validated 33-node/95-link result at
`results/2026-07-25_19-09-28_n=1_chargers=1/`; the paper's approximate
48-node/123-link counts and M=100 are validation targets, not forced
dimensions.

## 9. Measured paper-scale run

The run used the paper bounding box, 180 vehicles (60 F1 + 120 F2), `K=16`,
`alpha=0.01`, 100 queue replications per Nash iteration, six charger
configurations, and `MAX_NE_ITERATIONS=200`. The machine had 8 available CPUs;
all unset worker parameters resolved to 8, and BPR, queue replications, and
placement replications used that pool. Greedy construction and Nash iteration
updates remain sequential by algorithmic dependency.

Measured canonical graph: **33 nodes / 95 directed road links**, hash
`67e386ab0b44897bfaad4cf4d598f1fa8acb6957a18275522bb5303035097e9b`.

| Measurement | Result |
|---|---:|
| BPR coverage | 95/95 links, 25 levels/link, 0 failures |
| CG configurations | 6, all finite objectives |
| Queue replications | 6 × 200 × 100 = 120,000 completed simulator replications |
| Queue iteration wall time | 12.73–13.78 s mean per iteration with 8 workers |
| Separate placement comparison | 53.19 s internal / 55.21 s external wall, N=100 |
| Observed two-launch artifact assembly | 5,437.68 s wall (1.51 h) |
| Queue convergence | 0/6 reached 1% gap within 200 iterations |

The two-launch total includes the first launch that exposed and partially
completed the destination-charger case, followed by a repair/resume launch;
it is not a clean single-launch benchmark. The queue stage is the dominant
bottleneck. A crude edge-only extrapolation from 95 to the paper target of
123 links is 1.295×, suggesting roughly 16–18 seconds per queue iteration for
the same demand, route library, and worker count. This is only a planning
estimate: event density, OD count, `K`, and convergence behavior can dominate
network size. Peak memory for the long run was not available because macOS
`/usr/bin/time -l` could not query `sysctl` in this environment.

The complete machine-readable timing and status record is
[`paper_scale_resource_timing.json`](results/2026-07-25_19-44-24_n=3_chargers=2/paper_scale_resource_timing.json);
the detailed audit is
[`report.md`](results/2026-07-25_19-44-24_n=3_chargers=2/report.md). The CG
optimization result is in
[`all_optimization_results.pkl`](results/2026-07-25_19-44-24_n=3_chargers=2/all_optimization_results.pkl),
and queue comparison outputs are under
[`queue/`](results/2026-07-25_19-44-24_n=3_chargers=2/queue/).

## Running the available pieces

Install the declared dependencies with either:

```bash
pip install -r requirements.txt
# or
conda env create -f environment.yml
conda activate evopt
```

Network-only inspection:

```bash
python pipeline.py --config config.json --network-only
```

Deterministic bounded fixture benchmark (1x/2x/4x topology, multiple OD and
`K` values):

```bash
python benchmarks/run_benchmarks.py --mode fixture \
  --sizes 8,16,32 --od-counts 1,2,4,8 --k-values 8,16,32
```

Real-network cleaning/artifact benchmark:

```bash
python benchmarks/run_benchmarks.py --mode network --config config.json
```

The fixture benchmark writes `benchmark_results/benchmark_results.json` with
wall-clock timings, graph sizes, route counts, seed, worker/replication
parameters, stable hashes, and peak memory. Full BPR/queue bottleneck timings
are emitted in a pipeline run's `run_manifest.json` when the simulator is
available.

Full pipeline attempt:

```bash
python pipeline.py --config config.json
```

To run the full pipeline from a previously validated canonical network without
redownloading or repruning it, set `pipeline.artifact_dir` to the directory that
contains `network_manifest.json`, `nodes.csv`, and `edges.csv`. Candidate and OD
node IDs must refer to that artifact's contiguous node IDs. The pipeline verifies
the artifact hash, records the parent hash, and writes a self-contained network
copy into the new experiment directory.

Paper-scale run with the default full-CPU policy:

```bash
MPLCONFIGDIR=/private/tmp/evopt-mpl \
python pipeline.py --config config_paper_scale.json
```

Set `pipeline.parallel_workers` to an integer to cap the global pool. Leave it
`null` (the default) to use `os.cpu_count()`; `queue_simulation.WORKERS` and
`pipeline.bpr_generation.workers`/`fit_workers` can override individual stages.

Bounded real-network smoke run, including two OD pairs and queue comparison:

```bash
MPLCONFIGDIR=/private/tmp/evopt-mpl \
python pipeline.py --config config_smoke.json
python validate_outputs.py results/<experiment> --queue
```

The optimization result is `all_optimization_results.pkl` in the experiment
directory. The primary human-readable outputs are `report.md` and
`run_summary.txt`; queue placement results are under `queue/`, and the
machine-readable pass/fail contract is `sanity_check.json`.

Docker builds the Linux shortest-path library before running:

```bash
./run_experiment.sh --docker --build config.json
```

The smoke run and deterministic fixture validate the output chain. Paper-scale
claims still require running the full configured BPR sweep and `M=100` queue
replications in the intended simulator environment, then inspecting the
timings and fit-status records in `run_manifest.json` and `bpr_manifest.json`.
