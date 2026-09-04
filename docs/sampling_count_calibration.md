# BPR and queue sample-count calibration

## Decision

The production scaling configs now use:

| Setting | Value | Meaning |
|---|---:|---|
| BPR flow levels | 17 | One queue simulation at each capacity-relative level |
| Nash replications | 20 | Simulations of the current assignment per response step |
| Final comparison replications | 20 | Independent greedy/exhaustive repetitions |
| Nash tolerance | 1% | Relative used-route versus best-route gap |
| Nash response-step cap | 300 | Budget-aware failure boundary, not guaranteed convergence |

The adaptive BPR replication and explicit one-agent counterfactual queue paths
have been removed from the production workflow. The retained method is the
original current-assignment better response, corrected so an unused route uses
the currently simulated travel times of its links rather than an automatic
free-flow estimate.

## BPR experiment

The calibration downsampled a saved dense dataset containing 95 links and 104
single-run flow levels per link. For each candidate count, evenly spaced levels
were fit with fixed capacity and free-flow time, then evaluated over all 104
dense levels. The reference is the fit to all 104 levels.

| Levels | p95 curve deviation | Worst curve deviation | p95 high-flow error |
|---:|---:|---:|---:|
| 5 | 8.55% | 16.32% | 17.69% |
| 9 | 6.40% | 8.35% | 14.57% |
| 13 | 4.97% | 7.74% | 12.24% |
| 17 | 3.76% | 4.68% | 10.45% |
| 25 | 3.18% | 3.80% | 7.47% |

Seventeen is the first tested count whose worst fitted-curve deviation from
the dense reference is below 5%. It reduces BPR simulations by 32% relative to
25 levels and by 83.7% relative to the 104-level reference. The observed-data
p95 normalized error is still 11.06% at 17 levels and 10.66% at 25, indicating
model/simulator mismatch that additional sampling alone does not remove. All
production fits therefore remain subject to the strict fit gate.

## Queue replication experiment

For three saved charger configurations on the prior 33-node/95-link network,
100 corrected current-assignment simulations were generated per configuration.
Bootstrap prefixes compared 1, 2, 5, 10, 20, 50, and 100 replications against
the 100-run reference.

| Replications | Worst p95 route-time MAPE | Worst p95 gap error | Worst exact-action agreement |
|---:|---:|---:|---:|
| 5 | 2.89% | 0.288 | 18.2% |
| 10 | 1.99% | 0.190 | 28.4% |
| 20 | 1.38% | 0.140 | 36.4% |
| 50 | 0.88% | 0.079 | 48.4% |
| 100 | 0.62% | 0.055 | 62.4% |

Exact move identity remains unstable because several routes are nearly tied;
even 100 replications do not produce 95% agreement. Twenty is therefore a
pragmatic minimum, not a statistically certified move count. Fifty would be
the conservative setting when sub-1% route-time estimation is more important
than runtime.

Actual 20-step trajectories with 5, 10, and 20 replications ended with gaps
0.387, 0.375, and 0.365 respectively. A longer 20-replication trajectory
reached gap 0.206 after 100 steps (minimum 0.203) and did not converge. The gap
decreased in 66 of 99 transitions and increased in 33. Linear fits over the
full and trailing trajectory segments predict a 1% crossing between about 168
and 253 steps. This is extrapolation, not observed convergence; 300 was chosen
as a budget-aware cap with margin.

## Scope and limitations

- Calibration reused existing simulations as requested; no new 100/500/1,000
  node end-to-end experiment was run.
- Both studies use the saved small network, so the selected values are launch
  defaults that must be reassessed from pilot outputs on the new maps.
- The BPR comparison measures stability relative to a dense BPR fit, not proof
  that BPR is the correct functional form.
- The queue experiment shows that 20 replications estimates route means fairly
  well but does not stabilize near-tied move identities.
- Any run that reaches the Nash cap or a cycle is diagnostic only and must not
  be reported as an equilibrium result.

## Runtime implication

Relative to the previous 25-level BPR and 100-replication queue settings, the
new settings reduce BPR simulator calls by 32% and queue simulator calls per
response step by 80%. On the saved small network and eight workers, one
20-replication configuration took 76.24 seconds for 20 response steps and
378.79 seconds for 100 steps; a linear 300-step ceiling is about 18.9 minutes
for that one configuration on the old 95-edge graph. The previous complete six-configuration queue run
used 100 replications and 200 steps and spent about 45.9 minutes in each of two
active configuration waves (about 1.47 hours total).

These values cannot be multiplied by node count alone: simulator cost depends
on directed edges, active route lengths, vehicle interactions, and the number
of charger configurations. With five candidate sites and two chargers there
are 15 placement configurations. Scaling the observed one-configuration time
by 15 and by the 100-node map's 216/95 edge ratio gives roughly 10.8 hours at
the 300-step cap, before non-queue overhead. This is a coarse upper-budget
projection, not a runtime guarantee. The safe launch sequence is therefore
the 100-node job first, followed by 500 only after measured timing and
convergence are available. Do not launch the full 1,000-node queue stage under
a 12-hour budget based only on the small-network calibration.

## Artifacts

- Main calibration: `results/sampling_calibration/2026-09-04_simple-method-v2/`
- 5/10/20 trajectory study: `results/sampling_calibration/2026-09-04_nash-trajectories/`
- 100-step trajectory: `results/sampling_calibration/2026-09-04_nash-20rep-long/`
