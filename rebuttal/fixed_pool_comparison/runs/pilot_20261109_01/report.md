# WW controlled-6 vs strong fixed pool

Phase: pilot. Complete: True.

## all

| Method | N | Agent Acc | Exact Step Acc | Joint | Coverage | Input tokens | Output tokens (incl. reasoning) | Known cost | Unknown-cost calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| autojudge | 6 | 33.33% | 33.33% | 33.33% | 100.0% | 1546682 | 57039 | $0.606057 | 0 |
| fixed | 6 | 33.33% | 33.33% | 33.33% | 100.0% | 1579119 | 49586 | $0.597701 | 0 |

AJ − fixed (percentage points; paired 95% CI):

- agent: +0.00 pp [-50.00, +50.00]
- step: +0.00 pp [-50.00, +50.00]
- joint: +0.00 pp [-50.00, +50.00]

Cost ratio AJ/fixed: 1.014.

Costs include all saved attempts, including failed/invalid responses. Reasoning is a
subset of output tokens and is not billed twice here. Unknown token counts are shown
in report.json; displayed token totals sum known counts only. Manual billing
reconciliations and pre-generation HTTP rejections are separately counted there.
Do not infer strict budget matching from equal call counts.
