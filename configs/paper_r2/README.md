# Final rerun for the second-round revision

One manifest, `suite.json`, reruns every experiment reported in the paper with
the corrected queue code:

| Jobs | Paper result |
|---|---|
| `sensitivity_base` … `queue_seed_44` (21 jobs, base `base_500.json`) | Tables II–IV, Figs. 10–11 (501-node DC network) |
| `sensitivity_base` | Table V, "DC (medium)" |
| `scale_dc_0100` (`dc_0100.json`) | Table V, "DC (small)" |
| `scale_sf_1000` (`sf_1000.json`) | Table V, "SF (large)" |

Changes from `configs/revision1/`: `queue_simulation.N` 1 → 20 (paired final
replications) in all three bases, and `ALPHA` 0.01 → 0.025 in `dc_0100.json`
so every job uses the same convergence tolerance. Everything else is unchanged.

`sanity.json` is a five-minute end-to-end check (106 nodes, 3 candidates,
2 chargers, N = 2). Run it first after pulling new code.
