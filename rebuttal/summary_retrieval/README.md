# Historical summary-only and summary + DB results

The retained historical aggregates report the following comparisons between
summary-only evaluation and the summary + DB configuration.

On WW Hand, Agent Accuracy is 53.45% versus 61.49%, and exact Step Accuracy is
15.79% versus 17.25%. The corresponding recorded differences favor summary + DB
by 8.04 and 1.46 percentage points.

On WW Algo, Agent Accuracy is 50.00% versus 55.55%, and exact Step Accuracy is
33.33% versus 41.23%. The corresponding recorded differences favor summary + DB
by 5.55 and 7.90 percentage points.

The DB figures are averaged across runs; the summary-only repeat count is not
documented. Per-trace outputs, identical summaries and pools, matching model
allocations, and comparable scoring denominators have not been recovered for
this comparison. These aggregate results therefore do not establish a controlled
paired retrieval ablation, a paired confidence interval, or statistical
significance.

This note reports historical aggregate results. The controlled fixed-pool
experiment and the separate attribution-metric reproduction are linked from
the [common artifact index](../README.md).
