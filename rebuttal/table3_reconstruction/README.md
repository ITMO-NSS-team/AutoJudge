# Reconstructed few-shot ablation: reproduction

All six conditions use the same 58 Who&When Hand development traces, Gemini 2.5 Flash, six generated intermediate judges, a fixed aggregator and one generation repeat. The supplied few-shot examples vary; two examples are explicitly reconstructed. These are new controlled results, not recovered historical scores.

## Verify saved scores without API access

From the repository root, using Python 3 and its standard library:

```sh
python rebuttal/table3_reconstruction/verify.py
```

The checker verifies frozen execution-source hashes, identical inputs and labels, original successful predictions and recovery targets; it independently reproduces every Agent, exact Step and joint score. Agent names use exact case-sensitive equality. Predicted integer steps are compared to zero-based references plus one. All 58 traces remain in each denominator.

- [Verified scores and counts](verified-results.md), [full numerical results and intervals](verified-results.json).
- [Final predictions, prompts, inputs and labels](runs/recovered/).
- [First-pass results](runs/first_pass/).
- [Recovery plan](runs/recovered/recovery-plan.json), [completion checks](runs/recovered/completion-check.json), [combined accounting](runs/recovered/combined-accounting.json).
- [Runner](../../scripts/who_and_when/table3_ablation.py), [shared concurrency/budget control](../../scripts/who_and_when/table3_parallel_conditions.py), [recovery](../../scripts/who_and_when/table3_recovery.py).
- [Frozen execution source](../../experiments/table3/source/), [reconstructed examples](../../experiments/table3/reconstructed_examples.json), [recorded dependencies](../../experiments/table3/requirements-recorded.txt).

## Recovery and costs

The first pass produced 286 successes and 62 technical failures. One uniform recovery pass targeted all 62 failures, reusing validated stages only after request-identity checks, with up to three fresh attempts per unrecovered stage. All 62 were recovered. The original 286 successful predictions, pools and findings are unchanged and hash-linked. This additional recovery pass is distinct from the first-pass retry limit.

Known charges across both passes: $29.73833733. Conservative budget debit including lost-response reservations: $44.75573033. The exact total is unknown for lost responses. `runs/recovered/report.json` counts NEW recovery charges only; use `combined-accounting.json` for whole-experiment accounting. Paired bootstrap intervals use 10,000 resamples, seed 20261118, and are exploratory and unadjusted.

## Restore detailed receipts and recompute native reports

The [lossless archives](archives/) contain original API receipts, streams, pipeline traces and reconciliation records. First-pass files are split across condition archives; restore all archives together. Monitor snapshots, console logs, lock files and offline test traces are omitted.

```sh
python rebuttal/table3_reconstruction/restore_full_runs.py
```

The standard-library restore helper verifies `SHA256SUMS.json`, restricts extraction to the recorded `results/` directories and refuses to overwrite changed files. With the recorded dependencies available, recompute reports without model calls:

```sh
python scripts/who_and_when/table3_ablation.py report --output results/table3-reconstruction-20261118-olama-v3
python scripts/who_and_when/table3_recovery.py report --output results/table3-reconstruction-20261118-recovery-r1
```

## Optional fresh inference (paid)

With the recorded runtime dependencies and an OpenRouter key in `.env`:

```sh
python scripts/who_and_when/table3_ablation.py prepare --output results/table3-new-run --data rebuttal/fixed_pool_comparison/data/Hand-Crafted.parquet
python scripts/who_and_when/table3_parallel_conditions.py run --output results/table3-new-run --env-file .env --max-cost-usd 100
```

New model calls may produce different stochastic outputs. The commands use checkout-relative paths; historical absolute paths in recorded metadata are not needed for offline scoring. Keep new inference in a separate directory. The recovery script retains the historical first-pass/recovery layout for exact report reproduction.
