# Rebuttal-Scale Experimental Rebuild

## 1. Current situation

The previous experiments will not be reused as empirical evidence. The rebuttal
will be based on newly generated, pruned networks:

| Scale | Region | Target |
|---|---|---:|
| Test fixture | Synthetic/deterministic | approximately 20 nodes |
| Small | Washington, DC | 100 nodes |
| Medium | SF Bay Area | 500 nodes |
| Large | New York City | 1,000 nodes |

Current code already provides a unified pipeline, canonical network artifacts
and hashes, multi-OD demand handling, deterministic seed management, parallel
BPR/queue stages, strict BPR calibration, queue-gap histories, greedy,
exhaustive, and single-swap placement paths, and structural output validation.

The principal unresolved limitations are:

- The current custom pruning can remove ramps, over-contract junctions,
  fabricate parallel-edge combinations, and discard large regions through SCC
  selection.
- Existing city figures are exploratory and must be regenerated after pruning
  is corrected.
- Queue comparison has placement-seed ordering and conditional-averaging risks.
- Queue routes are derived from CG outputs, creating a circular dependency.
- Queue convergence lacks robust initialization, cycle, confidence, and
  route-augmentation checks.
- Existing output generation can create excessive numbers of files.
- No trustworthy large-network runtime measurements exist yet.

## 2. Planned implementation changes

### Study interface

Add a versioned study manifest and runner exposing city construction, demand,
candidate selection, charger budget, CG/queue controls, BPR settings, seeds,
workers, output retention, and Slurm resources. Save resolved per-job
configurations with network hashes, source revision, environment information,
and named seeds.

### Defensible network pruning

1. Archive an unsimplified OSM drive graph.
2. Normalize mixed highway tags and retain major roads plus matching link/ramp
   classes.
3. Preserve necessary connector paths.
4. Use topology-aware simplification for true pass-through nodes.
5. Consolidate intersection complexes conservatively in projected coordinates.
6. Validate directed connectivity and shortest-path distortion.
7. Search geographic window size only after the pruning method is fixed.
8. Preserve source-edge provenance, geometry, length, and free-flow time.

### Demand and candidate construction

Use deterministic boundary-sector OD pairs, capacity-calibrated transparent
synthetic demand, and five reproducible, geographically separated candidate
sites. Use 30% EV demand as the baseline, with low/base/high demand and EV-share
sensitivities.

### Modeling and statistical corrections

- Canonicalize charger sets before seed derivation.
- Separate search replications from independent evaluation replications.
- Use common random numbers across placements.
- Generate the primary queue route library independently from CG.
- Add CG, uniform, and random initializations and cycle detection.
- Evaluate final unilateral deviations explicitly.
- Use strict BPR fitting without proxy or constant fallback.

## 3. Planned experiment protocol

For DC-100, SF-500, and NYC-1000, use five candidate sites and charger budget
`C=2`. Evaluate all ten charger pairs as the exact reference, the singleton
configurations needed by greedy search, greedy with and without single swap,
and centrality, separation, and random baselines in both CG and queue models.

Use a tiered sensitivity design:

- Full demand, EV-share, and `C=1,2,3` comparisons on DC-100.
- Base-case full comparisons on SF-500 and NYC-1000.
- Targeted larger-network reevaluation of the leading placements under the
  non-baseline demand and EV-share settings.
- Queue route-count, initialization, stochastic-seed, update-rule, and explicit
  deviation checks on the leading configurations.

Report exact-match rate, median and maximum greedy gap, placement rankings and
uncertainty, total travel time, link-flow errors and correlations, station
utilization, and cross-model placement agreement.

## 4. Execution and resource policy

Assume four Slurm hosts with 16 CPUs each. Use independent, resumable jobs by
network, scenario, placement, and seed. Cap each job below 12 hours and
checkpoint queue state after every equilibrium iteration.

Measure all new stage runtimes rather than using old runs. Fit stage-specific
runtime models against nodes, edges, routes, vehicles, iterations,
replications, and effective workers. Report wall time, CPU-hours, parallel
efficiency, storage, and conservative upper prediction bounds.

Standard runs retain compressed summaries, assignments, convergence histories,
timings, and only a small number of debug traces. They do not retain every
per-timestep or per-agent simulation file.

## 5. Reviewer-facing deliverables

The final evidence package will contain:

- Validated DC, SF Bay, and NYC network artifacts and pruning figures.
- Multi-network, demand, EV-share, and charger-budget results.
- Queue convergence and route-independence diagnostics.
- Exact greedy-versus-exhaustive gaps and established baselines.
- Strict BPR fit diagnostics and downstream sensitivity.
- CG-versus-queue link-flow, station-flow, travel-time, and placement metrics.
- Hardware, environment, runtime, CPU-scaling, and storage tables.
- A reviewer-comment-to-experiment matrix with evidence links.

The empirical conclusions will be determined from the new artifacts. Old runs
are excluded from the resulting evidence tables.
