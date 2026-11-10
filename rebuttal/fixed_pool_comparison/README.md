# Adaptive versus fixed judge-pool comparison: run artifacts

- [Main run: all 126 Algo traces](runs/main_20261109_01/)
- [Main results](runs/main_20261109_01/report.md)
- [Six-trace technical pilot](runs/pilot_20261109_01/)
- [Initial smoke attempts](runs/smoke/)
- [Pinned public datasets](data/)
- [Artifact checksums](SHA256SUMS.json)

Each run contains:

| Path | Contents |
| --- | --- |
| `results/autojudge/` | Generated judge roles/instructions, individual findings, and final adaptive-pool decisions |
| `results/fixed/` | Fixed judge roles, prepared evidence, individual findings, and final fixed-pool decisions |
| `calls/` | Every API attempt: request, returned response, status, tokens, cost, and timing |
| `streams/` | Original returned SSE events, including returned reasoning and usage |
| `pipeline_traces/` | Native judge message histories and aggregation inputs |
| `inputs/` | Exact model-visible task and trace inputs |
| `references.json` | Ground-truth labels used only for scoring |
| `prompts.json`, `protocol.json` | Frozen prompts, models, settings, scoring rules, and original source fingerprints |
| `manifest.json` | Run identity, task IDs, dataset and input hashes, overlap groups |
| `report.md`, `report.json` | Metrics, paired confidence intervals, failures, and recorded costs |
| `environment.json`, `pricing.json`, `provenance.json` | Execution environment and provenance |

The two public parquet files are stored once in `data/` and shared by all runs.
Run artifacts retain their recorded bytes. Initial smoke attempts include
transport failures and can be incomplete. Predictions and failed attempts were
not filtered out of the main comparison.

## Main result

Both conditions use Gemini 2.5 Flash throughout, six intermediate judges, the
same fixed aggregator, and eight planned calls per trace. The fixed pool gets
an evidence-extraction call instead of pool generation.

| Method | Agent Accuracy | Exact Step Accuracy | Recorded inference cost |
| --- | ---: | ---: | ---: |
| Fixed pool + evidence extraction | 55.56% | 36.51% | $4.10058837 |
| Adaptive pool, controlled six-judge configuration | 55.56% | 41.27% | $4.41918210 |

The paired Step Accuracy difference is +4.76 percentage points, with a 95%
trace-bootstrap interval of [−3.17, +12.70]. Each method has one failed trace,
counted as incorrect. Costs include retries and failed attempts.

## Code and offline verification

The experiment uses the shared library in [`src/autojudge`](../../src/autojudge/).
Its [entry point](../../examples/who_and_when/run_fixed_pool_experiment.py),
[experiment-specific code](../../examples/who_and_when/fixed_pool/),
[configuration](../../docs/experiments/fixed-pool-config.json), and
[tests](../../tests/test_ww_fixed_pool_experiment.py) live in the normal repository
layout. This directory contains run artifacts and supporting data.

From the repository root, verify checksums, input identities, labels, predictions,
and recorded costs with the Python standard library:

```bash
python examples/who_and_when/fixed_pool/verify.py
```

To also regenerate the prompts, metrics, and paired intervals with the experiment
code, install the recorded direct dependency versions (Python 3.12 or newer) and
run the verifier without modifying the saved files:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r examples/who_and_when/fixed_pool/requirements.txt
.venv/bin/python -m examples.who_and_when.fixed_pool.verify --recompute
.venv/bin/python -m pytest tests/test_ww_fixed_pool_experiment.py
```

Verification and the mock-transport tests make no inference requests.
The dataset revision is `59b9fcba1aaed7bbf206b5f4d3c68b8face2f49c`.
For a new experiment, use the entry point's `prepare` and `run` commands;
`run` requires `OPENROUTER_API_KEY` and incurs inference charges.
