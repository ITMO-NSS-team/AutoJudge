# WW controlled-6 vs strong fixed pool

Phase: main. Complete: True.

## all

| Method | N | Agent Acc | Exact Step Acc | Joint | Coverage | Input tokens | Output tokens (incl. reasoning) | Known cost | Unknown-cost calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| autojudge | 126 | 55.56% | 41.27% | 41.27% | 99.2% | 5640015 | 1101090 | $4.419182 | 0 |
| fixed | 126 | 55.56% | 36.51% | 36.51% | 99.2% | 6078356 | 928281 | $4.100588 | 0 |

AJ − fixed (percentage points; paired 95% CI):

- agent: +0.00 pp [-8.73, +8.73]
- step: +4.76 pp [-3.17, +12.70]
- joint: +4.76 pp [-3.17, +12.70]

Cost ratio AJ/fixed: 1.078.

## nonoverlap

| Method | N | Agent Acc | Exact Step Acc | Joint | Coverage | Input tokens | Output tokens (incl. reasoning) | Known cost | Unknown-cost calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| autojudge | 81 | 56.79% | 41.98% | 41.98% | 98.8% | 3598400 | 708488 | $2.832213 | 0 |
| fixed | 81 | 58.02% | 37.04% | 37.04% | 98.8% | 3859979 | 590858 | $2.606535 | 0 |

AJ − fixed (percentage points; paired 95% CI):

- agent: -1.23 pp [-12.35, +9.88]
- step: +4.94 pp [-4.94, +14.81]
- joint: +4.94 pp [-4.94, +14.81]

Cost ratio AJ/fixed: 1.087.

## overlap

| Method | N | Agent Acc | Exact Step Acc | Joint | Coverage | Input tokens | Output tokens (incl. reasoning) | Known cost | Unknown-cost calls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| autojudge | 45 | 53.33% | 40.00% | 40.00% | 100.0% | 2041615 | 392602 | $1.586969 | 0 |
| fixed | 45 | 51.11% | 35.56% | 35.56% | 100.0% | 2218377 | 337423 | $1.494053 | 0 |

AJ − fixed (percentage points; paired 95% CI):

- agent: +2.22 pp [-11.11, +15.56]
- step: +4.44 pp [-6.67, +17.78]
- joint: +4.44 pp [-6.67, +17.78]

Cost ratio AJ/fixed: 1.062.

Costs include all saved attempts, including failed/invalid responses. Reasoning is a
subset of output tokens and is not billed twice here. Unknown token counts are shown
in report.json; displayed token totals sum known counts only. Manual billing
reconciliations and pre-generation HTTP rejections are separately counted there.
Do not infer strict budget matching from equal call counts.
