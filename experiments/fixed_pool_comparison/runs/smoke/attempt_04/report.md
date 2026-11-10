# WW controlled-6 vs strong fixed pool

Phase: smoke. Complete: True.

## all

| Method | N | Agent Acc | Exact Step Acc | Joint | Coverage | Input tokens | Output tokens (incl. reasoning) | Known cost | Unknown-cost calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| autojudge | 2 | 50.00% | 50.00% | 50.00% | 100.0% | 259740 | 19438 | $0.126517 | 0 |
| fixed | 2 | 50.00% | 50.00% | 50.00% | 100.0% | 271534 | 16664 | $0.123120 | 0 |

AJ − fixed (percentage points; paired 95% CI):

- agent: +0.00 pp [+0.00, +0.00]
- step: +0.00 pp [+0.00, +0.00]
- joint: +0.00 pp [+0.00, +0.00]

Cost ratio AJ/fixed: 1.028.

Costs include all saved attempts, including failed/invalid responses. Reasoning is a
subset of output tokens and is not billed twice here. Unknown token counts are shown
in report.json; displayed token totals sum known counts only. Manual billing
reconciliations and pre-generation HTTP rejections are separately counted there.
Do not infer strict budget matching from equal call counts.
