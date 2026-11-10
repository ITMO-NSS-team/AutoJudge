# Adaptive versus fixed judge-pool comparison

The experiment uses the repository's **`src/autojudge` library**:
`PoolGenerator`, `AgentPool`, `AgentNode`, `PipelineBuilder`, and `NodeSession`.
It does not carry a separate runtime or reimplement the evaluation graph.

- [Experiment entry point](../../examples/who_and_when/run_fixed_pool_experiment.py)
- [Experiment-specific transport, protocol, and reporting](../../examples/who_and_when/fixed_pool/)
- [Configuration](../../docs/experiments/fixed-pool-config.json)
- [Mock-transport tests](../../tests/test_ww_fixed_pool_experiment.py)
- [Complete experiment archive](fixed_pool_comparison.zip)
- [Main report](report.md) and [machine-readable results](report.json)
- [Offline archive verifier](verify.py)

The paired experiment covers all 126 Who&When Algorithm-Generated traces. Both
methods use Gemini 2.5 Flash throughout, six intermediate judges, the same fixed
aggregator, and eight planned calls per trace. The fixed pool gets an evidence
extraction call instead of pool generation.

| Method | Agent Accuracy | Exact Step Accuracy | Recorded inference cost |
| --- | ---: | ---: | ---: |
| Fixed pool + evidence extraction | 55.56% | 36.51% | $4.10058837 |
| Adaptive pool, controlled six-judge configuration | 55.56% | 41.27% | $4.41918210 |

The paired Step Accuracy difference is +4.76 percentage points, with a 95%
trace-bootstrap interval of [−3.17, +12.70]. Each method has one failed trace,
counted as incorrect. Costs include retries and failed attempts. Matching model
allocation and stagewise limits does not imply identical realized spending.

## Reproduce the report without model calls

Run from the repository root. Python 3.12 or newer is required for the recorded
dependency versions; archive verification itself uses only the standard library.

```bash
python experiments/fixed_pool_comparison/verify.py
python -m zipfile -e experiments/fixed_pool_comparison/fixed_pool_comparison.zip \
  experiments/fixed_pool_comparison/extracted
python -m venv .venv
.venv/bin/python -m pip install -r experiments/fixed_pool_comparison/requirements.txt
.venv/bin/python examples/who_and_when/run_fixed_pool_experiment.py report \
  --output-dir experiments/fixed_pool_comparison/extracted/main_20261109_01/experiment
.venv/bin/python -m pytest tests/test_ww_fixed_pool_experiment.py
```

The entry point imports the library from this checkout's `src/autojudge`.
The test suite simulates model responses with `httpx.MockTransport`. Reporting
and tests require no API credentials and send no inference requests.

## Start a new paired run

From the repository root, using the environment above:

```bash
.venv/bin/python examples/who_and_when/run_fixed_pool_experiment.py prepare \
  --phase pilot --output-dir experiments/fixed_pool_comparison/new_runs/pilot
.venv/bin/python examples/who_and_when/run_fixed_pool_experiment.py run \
  --output-dir experiments/fixed_pool_comparison/new_runs/pilot --max-cost-usd 4
.venv/bin/python examples/who_and_when/run_fixed_pool_experiment.py prepare \
  --phase main --output-dir experiments/fixed_pool_comparison/new_runs/main
.venv/bin/python examples/who_and_when/run_fixed_pool_experiment.py run \
  --output-dir experiments/fixed_pool_comparison/new_runs/main \
  --pilot-dir experiments/fixed_pool_comparison/new_runs/pilot --max-cost-usd 22
```

`prepare` downloads the pinned public dataset without invoking a model. `run`
requires `OPENROUTER_API_KEY` and incurs inference charges. Limits are cumulative
for each output directory. A completed pilot with identical frozen settings and
source hashes is required before the main run. Future predictions and charges
can vary with provider behavior. The dataset revision is
`59b9fcba1aaed7bbf206b5f4d3c68b8face2f49c`.

## Provenance

The ZIP retains the main experiment, its six-trace technical pilot, and all four
initial smoke attempts. It contains requests, responses, SSE streams, generated
judges, evidence, final predictions, pipeline histories, public datasets, frozen
prompts/configuration, source snapshots, and checksums. Historical source
snapshots are retained **inside the archive**, rather than as a second library
in the repository.

Scientific inference artifacts and frozen protocols are byte-identical to the
recorded experiment. Module names, CLI descriptions, session labels, and
administrative paths were normalized for publication; internal planning documents
were omitted. No model-facing prompts, evaluation logic, settings, or recorded
predictions were changed. [source_publication.json](source_publication.json)
distinguishes executed source fingerprints from published fingerprints.

The frozen protocol retains its original execution fingerprints. Use `report`
for offline recomputation and `prepare` to create a fresh run with the published
paths. The saved `artifact_audit.json` describes the pre-publication audit;
`verify.py` checks the public archive and its source against this repository.
Credentials and `.env` files are not included.
