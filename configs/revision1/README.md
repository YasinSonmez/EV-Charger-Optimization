# Revision 1 configurations (as run on the cluster)

Exact configuration files that produced the results in Revision 1 of the
T-ITS manuscript (T-ITS-25-12-6135.R1). Kept unchanged for provenance; the
corrected rerun lives in `configs/paper_r2/`.

| File | Paper result |
|---|---|
| `sensitivity_base_500.json` + `sensitivity_suite.json` | Tables II–IV and Figs. 10–11 (501-node DC network) |
| `sensitivity_base.json` | Table V, "DC (small)" row (106 nodes) |
| `sf_1000.json` | Table V, "SF (large)" row (961 nodes) |

Table V's "DC (medium)" row is the `sensitivity_base` job of the 500-node suite.
The `demand_120` suite entry was added after the original run and never completed.

Known issues in the code these runs used (fixed in the next commit):
charging vehicles were not routed into the station model, unused charging
routes omitted charging time, the final comparison used one replication
(`N = 1`) with a different random seed per placement, and cycling searches
kept the assignment at cycle detection.
