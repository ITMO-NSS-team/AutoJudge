# Historical summary-only versus summary + DB results

The retained experiment workbook explicitly labels two configurations:
`AutoJudge + summ (no DB tool)` and `AutoJudge, summ + DB`.
The following values are transcribed from those columns, not regenerated model
predictions.

| Dataset | Metric | Summary-only | Summary + DB | Difference |
| --- | --- | ---: | ---: | ---: |
| WW Hand | Agent Accuracy | 53.45% | 61.49% | +8.04 pp |
| WW Hand | Exact Step Accuracy | 15.79% | 17.25% | +1.46 pp |
| WW Algo | Agent Accuracy | 50.00% | 55.55% | +5.55 pp |
| WW Algo | Exact Step Accuracy | 33.33% | 41.23% | +7.90 pp |

## Evidence and scope

The worksheet is `Главная 26.03`. Configuration labels are in D20 and J20;
the values are in D21:D24 and J21:J24. The retained source cells and workbook
fingerprint are included in [source_cells.json](source_cells.json).
[comparison.csv](comparison.csv) provides the four comparisons and exact cell
references in machine-readable form.

The summary+DB values are explicitly described in the workbook as averaged
across runs. The summary-only repeat count is not documented. Shared summaries,
generated pools, model allocations, per-trace predictions, and comparable scoring
denominators have not been established from these aggregate records. Actual
retrieval calls are not available in this extract. A paired confidence interval
or statistical-significance claim cannot be recovered from four percentages.

These historical results favor the recorded summary+DB configuration on both
metrics for both WW datasets. They are descriptive evidence, with the provenance
limitations above; they do not establish an isolated causal effect of retrieval.

## Offline check

From the repository root:

```bash
python rebuttal/summary_retrieval/verify.py
```

The checker uses the Python standard library. It verifies checksums, compares
every CSV value to its retained source cell, and recomputes the differences using
decimal arithmetic. It does not invoke a model or independently reproduce the
original judge predictions. See the [common artifact index](../README.md) for
the controlled fixed-pool experiment and the separate attribution-metric check.
