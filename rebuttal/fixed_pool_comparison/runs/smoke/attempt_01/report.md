# WW controlled-6 vs strong fixed pool

Phase: smoke. Complete: False.

**INCOMPLETE / PROVISIONAL: pending traces remain in the denominator.**

## all

| Method | N | Agent Acc | Exact Step Acc | Joint | Coverage | Input tokens | Output tokens (incl. reasoning) | Known cost | Unknown-cost calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| autojudge | 2 | 50.00% | 50.00% | 50.00% | 50.0% | 178096 | 16393 | $0.094411 | 2 |
| fixed | 2 | 50.00% | 50.00% | 50.00% | 50.0% | 85956 | 8042 | $0.045892 | 0 |

AJ − fixed (percentage points; paired 95% CI):

- agent: +0.00 pp [+0.00, +0.00]
- step: +0.00 pp [+0.00, +0.00]
- joint: +0.00 pp [+0.00, +0.00]

Cost ratio: unavailable.

Costs include all saved attempts, including failed/invalid responses. Reasoning is a
subset of output tokens and is not billed twice here. Unknown token counts are shown
in report.json; displayed token totals sum known counts only. Manual billing
reconciliations and pre-generation HTTP rejections are separately counted there.
Do not infer strict budget matching from equal call counts.
