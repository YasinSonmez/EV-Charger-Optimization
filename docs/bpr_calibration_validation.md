# BPR Calibration Diagnosis and Validation

## Decision

The former capacity-fraction link probe is not valid for production results.
It paired offered demand with target-link traversal time while omitting the
queue upstream of the target, and its nominally nonbinding continuation could
have the same capacity as the target. This produced noisy, nonmonotone curves.

The corrected production-preserving method is versioned as
`offered_cohort_entry_wait_v2`. It:

1. injects one deterministic cohort at each configured demand level;
2. makes every probe continuation at least ten times the target capacity;
3. measures departure-to-target-exit cost, including the admission queue and
   excluding continuation-link travel;
4. simulates the zero-flow probe rather than mixing an analytical continuous
   free-flow point with integer-second simulation output;
5. stores BPR flow and capacity as vehicles per the configured demand window,
   matching the congestion optimizer's cohort-flow units; and
6. keeps requested vehicles/hour and measured active-period throughput only as
   separate diagnostics.

Old link checkpoints cannot satisfy the new calibration-version identity and
will not be resumed as corrected observations.

## Evidence

The bounded study used the existing 49-node, 100-directed-edge sanity network.
No new large end-to-end experiment was run.

### Failure of the previous probe

- All 100 links had at least one travel-time decrease as demand increased.
- Median decreases per link: 4; maximum: 8.
- R²: minimum 0.223, median 0.683, maximum 0.931.
- Only 2 of 100 links reached R² >= 0.90.
- On representative link 23, independent-seed coefficient of variation was
  72% at capacity and 49% at 1.5 times capacity.

The primary cause was downstream-continuation interference. Merely increasing
continuation capacity removed the spikes but exposed a second error: measuring
only time after target entry omitted the target's upstream queue and therefore
created an almost flat response above capacity.

### Corrected measurement

Five links spanning previously poor and strong fits were each run with five
independent seeds and eight demand levels:

- all 25 curves were monotone;
- individual-run R² ranged from 0.975 to 0.992;
- mean-curve R² ranged from 0.976 to 0.989; and
- the formerly unstable link 23 improved to R² 0.983--0.990, with maximum
  cross-seed CV 7.2% rather than more than 70%.

An all-link run then used nine levels
`[0, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2] × capacity`:

- links fitted: 100/100;
- links passing the explicit R² >= 0.95 gate: 100/100;
- minimum R²: 0.953;
- median R²: 0.979;
- mean R²: 0.980; and
- generation plus fitting time: 82.8 seconds with 8 simulation workers.

The fit-only rerun took about 2.2 seconds. The output directory occupies about
48 MB, mostly restartable per-link simulation checkpoints.

### Why nine levels were selected

A separate 17-level run was made on five representative links. Fitting only
the nine alternating levels and comparing with the full 17-level fit changed:

- alpha by at most 1.3%;
- beta by at most 3.0%; and
- the fitted curve by at most 2.8% of the full fitted response range.

Five levels were rejected: despite high training R², interleaved holdout error
near the capacity knee was large. Nine is therefore the smallest tested design
that preserved the fitted two-parameter curve; 17 approximately doubles the
simulation work without materially stabilizing its parameters.

No repeated simulations are part of production calibration. Replications were
used only in this bounded validation study to measure seed sensitivity.

## Important limitation: R² is not the whole story

The corrected queue response is almost flat below capacity and then develops a
sharp finite-horizon queue. A standard smooth BPR function cannot reproduce
that knee exactly. High R² is dominated by the large delays above capacity.
For the nine-level all-link fit, median pointwise relative error was 8.2%, but
the upper-tail relative error around the knee was much larger. These errors are
now recorded per link in `fitter_results.csv`; they are not hidden by the R²
gate.

As a diagnostic, a three-parameter capacity-knee function
`t=t0(1+a*max(v/c-tau,0)^b)` was tested on the same observations. Its median R²
was 0.99987 and its 95th-percentile pointwise relative error was 13.3%, versus
215% for the standard BPR fit. It is a substantially better queue surrogate,
but it is not the standard BPR model used in the paper and would require
changes to the congestion-game potential, derivatives, tests, and manuscript.
It has therefore not been silently adopted for rebuttal runs.

## Remaining scientific choice

The scale configurations interpret demand as a vehicle cohort over a 0.1-hour
(six-minute) demand window. Thus a one-lane simulator capacity of 1,900
vehicles/hour becomes 190 vehicles per optimization window. This fixes the
previous factor-of-ten optimizer-unit mismatch, but the six-minute period is a
modeling assumption that should be stated in the paper and kept identical
across all compared experiments.

For a short rebuttal timeline, the defensible path is to retain standard BPR,
use the corrected nine-level calibration and R² >= 0.95 gate, and report the
capacity-knee mismatch as a limitation. Adopting the shifted model should be a
separate, explicit methodology decision rather than a fit-quality tweak.

## Inspectable outputs

- `results/bpr-validation/corrected-9-level-sanity-network/bpr_fit_samples.png`
- `results/bpr-validation/corrected-9-level-sanity-network/fitter_results.csv`
- `results/bpr-validation/corrected-9-level-sanity-network/bpr_observations.csv.gz`
- `results/bpr-validation/corrected-9-level-sanity-network/bpr_manifest.json`
- `results/bpr-validation/corrected-9-level-sanity-network/link_probe_summary.json`
