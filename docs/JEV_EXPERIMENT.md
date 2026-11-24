# Typed Jev experiments across benchmarks

This path evaluates normalized cases through the same Jev runtime regardless of
their source benchmark. It is an implementation scaffold for paired experiments;
its accuracy, latency, and failure rate still require external runs on held-out
benchmark data. The earlier one-trace Who&When pilot is a connectivity check,
not evidence that a specialist pool improves quality.

## Data and execution contract

```text
raw benchmark data
    -> benchmark converter -> cases.jsonl + references.jsonl
cases.jsonl + references.jsonl + evaluation spec
    -> preflight validation
cases.jsonl + evaluation spec
    -> direct Jev, fixed specialist plan, or Meta LLM adaptive plan
    -> JevEvaluator -> typed answers + per-call receipts
typed answers + references.jsonl + metric spec
    -> scoring -> results.jsonl + summary.json
```

`EvaluationCase` contains a `case_id`, the task, and ordered evidence events.
Each event has a stable ID within the case, a normalized one-based `index`,
optional actor and role, content, and optional original `source_step`/`source_id`.
`ReferenceCase` has the same `case_id` and a separate `labels` object. A converter
must not put reference labels into case fields, including metadata or summaries.
The Meta LLM and Jev evaluator receive only `EvaluationCase` and
`EvaluationSpec`. References are used by preflight and scoring, never by the
models.

The versioned `EvaluationSpec` declares the benchmark taxonomy, result targets,
and metrics. A target gets candidates from `events`, distinct `actors`, or
explicit `options`; it also declares one or many selections and whether no
match is valid. Metrics project a selected event's ID, actor, normalized index,
or original source step, or an option ID, into a benchmark label. Supported
metrics are exact accuracy, binary F1, binary precision, set F1, and mean
absolute error. MAE requires an explicit `failure_penalty` so failed cases
remain in its denominator. The contract permits any finite nonnegative
penalty, including zero. A zero penalty rewards failed cases with zero error;
use it only if that is the official rule. Freeze the value before held-out runs.

For example, the [Who&When spec](../examples/jev/specs/who_and_when.json)
selects one event and derives both `agent` and `step` from that same event.
The [WebArena](../examples/jev/specs/webarena.json) and
[Pumpkin](../examples/jev/specs/pumpkin.json) specs show categorical targets.
WebArena reports F1 and precision for its three annotated outcomes. Its
`optimal` question is excluded from the measured spec because the local
annotations have no reference label for it.

The Pumpkin spec is an example of a scoring contract; a raw-to-normalized
Pumpkin converter is not included yet. Such a converter must derive the boolean
reference label `is_ideal` from independent human ground truth (`ht` in the
legacy `HUMAN_GT_PATH` file), not from a previous judge's score. Human labels
`1`/`ideal` map to `true` and `0`/`fair`/`poor` to `false` when those values
have the documented source meaning. Missing or unknown human labels are
conversion errors. The 11 named criteria also need operational rules in the
spec before a competitive Pumpkin evaluation.

To add a benchmark, write a converter that emits this case/reference format
and a spec for its target fields. Jev execution and scoring stay the same.

## Judge modes

- **`direct`**: Jev answers each target from the case and spec. This is the
  baseline for measuring the added value of specialists.
- **`fixed`**: a supplied `JudgePlan` defines reusable specialist questions.
  The plan should express benchmark-level checks without naming a case's answer.
  The plan's offline creation cost is outside the recorded run cost.
- **`adaptive`**: `JudgePlanGenerator` calls a chat Meta LLM for each case. Its
  prompt requests neutral specialists, and validation requires at least one
  specialist per target and at least two in total, with a cap of
  `max(6, target count + 2)`. The runtime constructs candidates and makes the
  final decisions.

The included [Who&When fixed plan](../examples/jev/plans/who_and_when_6.json)
and [WebArena fixed plan](../examples/jev/plans/webarena_6.json) each contain
six benchmark-level specialists. Set `--specialist-count 6` for the adaptive arm
when comparing it with either fixed plan. This matches the planned specialist
count; retries and event-window screening can still change the number of Jev
calls actually made.

The runner defaults to three attempts per Jev request and at most four
simultaneous Decisions requests within a case (`--max-attempts` and
`--max-concurrent-requests`). A 429 or 503 response starts a shared cooldown
for the runner's Jev client; `Retry-After` is honored up to the transport's
120-second retry cap. The ledger records each attempt and its receipt when
available. Before each POST, the runtime checks that the full request is
serializable and within its configured size limit. It also checks all
first-pass target requests before the mode's first Jev call. A later final
request can still exceed that limit after specialist findings are known; its
local failure is preserved in the case result.

For a reported comparison, pass a provider-supported, dated Jev model ID to
`--jev-model` when available and inspect the resolved model IDs in the call
ledger. The default `typesafe/jev-1.14` is an alias, so its backing snapshot may
change. Pin a Meta model snapshot too when the chat provider supports one.

Each mode receives **the same case IDs in the same order**. Within each case,
the runner cyclically rotates the mode order to distribute order effects. It
checks case/reference pairing before calling a model and writes one result row
for every case and mode, including a plan or inference failure. The manifest
records the mode-order policy, input and implementation file hashes, selected
IDs, the predeclared primary metric, requested model IDs, planner prompt
version, resolved Meta/Jev runtime settings, route fingerprints, and installed versions of Pydantic,
HTTPX, PydanticAI, and genai-prices. Every adaptive plan and each Jev call
ledger are kept in the result row for diagnosis. The separate `references.jsonl`
file is used by the runner for preflight and scoring.

## Running a paired comparison

Convert an original Who&When Parquet or JSONL file:

```bash
uv run --group benchmarks python examples/jev/convert_who_and_when.py \
  Algorithm-Generated.parquet --output-dir /tmp/jev-who-when-input
```

The converter writes `cases.jsonl`, `references.jsonl`, `errors.jsonl`, and a
completion `manifest.json` with source and output hashes and row counts. The
runner refuses an incomplete conversion or any conversion errors. For manually
prepared normalized files without a converter manifest, pass
`--allow-unmanifested-input` after checking the files yourself. For Who&When,
`index` and the reference `step` are one-based; `source_step` preserves the
source's zero-based position.

For the Algo subgroup without Hand question overlap, create an explicit
selection list before the run:

```bash
uv run --group benchmarks python examples/jev/build_who_and_when_holdout.py \
  --hand /path/to/Hand.parquet \
  --algo Algorithm-Generated.parquet \
  --output-dir /tmp/jev-who-when-holdout
```

The builder excludes an Algo row when its `question_ID` or normalized question
text occurs in Hand. Its manifest records both source hashes, the exact text
normalization rule, and the selected list hash. The paper does not publish its
exact text-normalization function, so check the actual selected count on the
other computer; do not assume it is the paper's 81 cases. The runner uses the
listed order and hashes `case_ids.txt` in its own manifest. Omit
`--case-ids-file` to evaluate all converted Algo cases.

Check normalized cases, references, spec, and an optional fixed plan before
setting credentials or creating an output directory:

```bash
uv run --group benchmarks python examples/jev/run_experiment.py \
  --cases /tmp/jev-who-when-input/cases.jsonl \
  --references /tmp/jev-who-when-input/references.jsonl \
  --spec examples/jev/specs/who_and_when.json \
  --primary-metric step_accuracy \
  --case-ids-file /tmp/jev-who-when-holdout/case_ids.txt \
  --modes direct fixed adaptive \
  --fixed-plan examples/jev/plans/who_and_when_6.json \
  --specialist-count 6 \
  --validate-only
```

This prints the selected case count and IDs. Preflight also checks that gold
labels are reachable from the declared target candidates and projected event
fields. Preflight checks fixed-plan target coverage. `--limit` selects the same
first N cases for both preflight and run; it cannot be combined with
`--case-ids-file`. The WebArena converter orders trace
files lexicographically, so its first N cases are not a random held-out sample.

Set the chat and Decisions credentials as described in the
[existing Jev setup](../README.md#compare-jev-judges-on-whowhen), then run:

```bash
uv run --group benchmarks python examples/jev/run_experiment.py \
  --cases /tmp/jev-who-when-input/cases.jsonl \
  --references /tmp/jev-who-when-input/references.jsonl \
  --spec examples/jev/specs/who_and_when.json \
  --primary-metric step_accuracy \
  --case-ids-file /tmp/jev-who-when-holdout/case_ids.txt \
  --output-dir /tmp/jev-who-when-run-001 \
  --modes direct fixed adaptive \
  --fixed-plan examples/jev/plans/who_and_when_6.json \
  --specialist-count 6 \
  --meta-temperature 0
```

For WebArena, convert prepared `cleaned/` traces with the AgentRewardBench
annotations, then validate the three-target spec and its fixed plan:

```bash
uv run --group benchmarks python examples/jev/convert_webarena.py \
  /path/to/webarena/cleaned /path/to/annotations.csv \
  --output-dir /tmp/jev-webarena-input

uv run --group benchmarks python examples/jev/run_experiment.py \
  --cases /tmp/jev-webarena-input/cases.jsonl \
  --references /tmp/jev-webarena-input/references.jsonl \
  --spec examples/jev/specs/webarena.json \
  --primary-metric success_f1 \
  --modes direct fixed adaptive \
  --fixed-plan examples/jev/plans/webarena_6.json \
  --specialist-count 6 \
  --validate-only
```

Run the same WebArena comparison after preflight:

```bash
uv run --group benchmarks python examples/jev/run_experiment.py \
  --cases /tmp/jev-webarena-input/cases.jsonl \
  --references /tmp/jev-webarena-input/references.jsonl \
  --spec examples/jev/specs/webarena.json \
  --primary-metric success_f1 \
  --output-dir /tmp/jev-webarena-run-001 \
  --modes direct fixed adaptive \
  --fixed-plan examples/jev/plans/webarena_6.json \
  --specialist-count 6 \
  --meta-temperature 0
```

The output directory must be new unless `--resume` is set. It contains
`manifest.json`, `results.jsonl`, and `summary.json` when the run finishes.
To continue an interrupted run, repeat the command with the same settings and
`--resume`. The runner checks input and source SHA-256 hashes, selected case
IDs, the primary metric, modes, models, planner prompt version, runtime settings, route
fingerprints, dependency versions, and mode order before skipping completed
pairs. It also requires the result rows to form the recorded execution-order
prefix and an unfinished pair to be next in that order. A POSIX file lock
prevents concurrent runs in one output directory on Linux and macOS.

The project currently has dependency lower bounds rather than a lockfile, so
fresh runs can resolve different library versions. Compare the recorded
versions when interpreting results across runs.

Complete result rows stay append-only. Malformed, duplicate, or missing
complete rows stop resume. If only the last line is unterminated, the durable
row count matches the manifest, and `inflight_pair` names the next pair, the
runner saves that line as `results-tail-*.bin` before truncating it and reruns
the pair. The manifest records the quarantined file. A write-ahead
`inflight_pair` marker can also remain after a complete row is written; that
row is retained. A pair rerun without a durable row enters
`untracked_cost_modes` and `untracked_latency_modes`: its complete cost,
latency, and paired cost/speed deltas remain `null`, because the earlier attempt
may have incurred charges and time outside the result ledger. Known partial
amounts remain in `recorded_cost_usd` and `recorded_total_seconds`.
Only provider billing can establish the exact cost after such a crash. The
runner exits nonzero if any attempt is partial or failed, after writing the
reports.

## Reading the summary

Exact accuracy counts a missing or invalid target answer as zero. Binary F1
and precision count a failed positive case as a false negative and a failed
negative case as a false positive. Precision is zero when there are no
predicted positives; binary F1 is zero when TP, FP, and FN are all zero,
matching the existing AgentRewardBench calculation. The primary set F1 is the
mean of per-case F1 scores, with zero for a failed case. Standard micro F1 is
also reported as a diagnostic; an invalid case with empty gold contributes no
micro TP/FP/FN. MAE uses the benchmark-defined failure penalty; a zero
penalty gives a failed case zero absolute error. The summary reports the
selected-case count.

The summary also shows status counts, strict completion rate, per-metric
coverage, planner/judge/total cost and time, joint accuracy across all exact
metrics, case-level p50 and p95 latency, and paired deltas. `jev_calls` counts Decisions calls
with at least one transport attempt; `http_attempts` counts all recorded
attempts, and `retries` counts attempts after the first within each call.
Each metric reports `higher_is_better`; only MAE is lower-is-better. Metric
deltas are raw arm-minus-baseline differences, so a negative MAE delta means an
improvement. `delta_vs_direct` compares each pool with direct Jev, while
`delta_vs_fixed.adaptive` compares adaptive planning directly with a fixed
plan. `exact_metric_case_comparisons` reports wins, losses, and ties separately
for each exact metric on the same paired cases.
`quality_gain_per_extra_usd` divides the direction-adjusted metric delta by a
known, positive run-cost delta. It is `null` when incremental cost is unknown,
zero, or negative. Negative values mean worse quality for additional spend.
It divides a change in mean quality by a change in total run USD, so its
magnitude depends on case count and metric scale. Compare it within one
benchmark and selected set. `cost_usd_per_case` gives incremental run cost
divided by the number of selected cases.
Cost is `null` when a provider receipt or Meta LLM price estimate is missing,
or when an interrupted pair may have untracked charges; recorded partial cost
is shown separately.
Cost deltas compare **run cost only**. Adaptive run cost includes one Meta LLM
plan per case; fixed mode's plan may have a separate offline creation cost,
which must be reported separately for a fair cost comparison.

The direct, fixed, and adaptive modes should be compared on held-out cases from
each benchmark, with repeated runs to measure variance. Report quality,
completion rate, latency, and complete cost together. Large one-choice event
sets require window screening, which can prune the correct candidate; inspect
the recorded calls and compare with direct mode when evaluating that tradeoff.

## Protocol for a competitive result

Freeze the converter, spec, `--primary-metric`, model IDs, evidence policy,
prompt version, and scoring rules before evaluating a held-out split. Use
development cases only
to choose a fixed plan, tune prompts, or decide whether window screening is
acceptable. Do not tune on the held-out labels. Who&When Hand was used for
development in the paper; the 81 Algo cases without overlap by question ID
**or normalized question text** are a more informative subgroup than a random
split that mixes those questions. Our split builder freezes its own explicit
normalization rule and list, which must be checked against the source data.
For other benchmarks, retain the official split when one exists and publish
the selected case IDs and source hashes.

Run `direct`, `fixed`, and `adaptive` on the same normalized cases with the
same Jev model, provider route, evidence, retry policy, and scoring spec.
Use a fixed plan designed from development data only. Record its creation
cost and amortize it over the number of evaluated cases when comparing total
spend. The adaptive arm pays for one Meta LLM plan per case. Compare the
number of specialists and Jev calls as well as quality: a larger pool uses a
different compute budget. Keep one selected configuration per benchmark
across all reported metrics instead of taking the best arm separately for
each metric.

For each benchmark, declare one primary quality metric with `--primary-metric`
before inference; the runner checks it against the spec and records it in the
manifest and summary. Report all secondary metrics, completion rate, mean and
p95 case latency, and total cost. Report unknown cost separately:
`recorded_cost_usd` is only the known
part, while `total_cost_usd: null` means the full cost is unavailable. Include
retry charges, Meta LLM usage, and any shared preprocessing or fixed-plan
creation cost in a separate all-in estimate. Case latency is measured by the
runner's execution pattern; it is not a throughput benchmark at a different
concurrency setting. For a quality-versus-cost claim, show the paired quality
gain and incremental USD on the same cases, including failed attempts. The
primary metric is a study-level declaration fixed by the run manifest.

Repeat model runs on the held-out cases to show variance, even at temperature
zero. Report paired per-case changes and uncertainty intervals for the
primary metric. `delta_vs_direct` and `delta_vs_fixed` are point estimates, not
significance tests. Run the original chat AutoJudge on the same selected cases
as an additional reference when practical. Published benchmark figures are
useful context, but differences in backbone model, split, labels, and failure
handling make them non-controlled comparisons until those settings match.
The WebArena `success_f1` command above declares an internal primary metric;
the paper's AxTree precision result is not a reproduction target for it. The
legacy scorer omits missing predictions, while this runner counts them as
failures and reports coverage. For a published comparison, run its official
scorer on exactly the same selected cases and report coverage alongside this
runner's failure-adjusted result.
Who&When's legacy scripts and paper use different actor-matching conventions;
this spec uses exact actor and source-step matches. Fix one scorer and apply it
to every arm before comparing figures. Legacy WebArena and Pumpkin scripts
also skip some missing outputs, while this runner includes failed cases in the
denominator.
