# Adaptive versus fixed judge-pool comparison

This package contains the complete paired comparison on all 126 Who&When
Algorithm-Generated traces, its six-trace technical pilot, and all four initial
smoke attempts. Both methods use Gemini 2.5 Flash for every component, six
intermediate judges, the same fixed aggregator, and eight planned calls per trace.
The fixed pool gets an evidence-extraction call instead of pool generation.

| Method | Agent Accuracy | Exact Step Accuracy | Recorded inference cost |
| --- | ---: | ---: | ---: |
| Fixed pool + evidence extraction | 55.56% | 36.51% | $4.10058837 |
| Adaptive pool, controlled six-judge configuration | 55.56% | 41.27% | $4.41918210 |

The paired Step Accuracy difference is +4.76 percentage points, with a 95%
trace-bootstrap interval of [−3.17, +12.70]. Each method has one failed trace,
counted as incorrect. Costs include retries and failed attempts. Matching the
model allocation and stagewise limits does not imply identical realized spending.

- [fixed_pool_comparison.zip](fixed_pool_comparison.zip): raw requests, responses,
  SSE streams, generated judges, evidence, final predictions, native pipeline
  histories, pinned public datasets, frozen prompts/configuration, source
  snapshots, manifests, and checksums.
- [source/](source/): the main run's execution-time source snapshot, including its
  AutoJudge runtime, CLI, tests, and launcher, under neutral module names.
- [report.md](report.md) and [report.json](report.json): the main results, including
  the 81 traces without Hand overlap and the 45 overlapping traces.
- [source_publication.json](source_publication.json): original execution hashes
  and published hashes for the source files.
- [verify.py](verify.py): offline archive, source, and metric verification.

## Verify the saved experiment without model calls

Requires Python 3.12 or newer. Archive verification itself uses only the standard
library:

```bash
cd experiments/fixed_pool_comparison
python verify.py
python -m zipfile -e fixed_pool_comparison.zip extracted
```

To recompute the metrics and paired intervals with the experiment's own code:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cd source
../.venv/bin/python examples/who_and_when/run_fixed_pool_experiment.py report \
  --output-dir ../extracted/main_20261109_01/experiment
../.venv/bin/python -m pytest tests/test_ww_fixed_pool_experiment.py
```

The test suite simulates model responses with `httpx.MockTransport`. Neither
reporting nor the tests require API credentials or send inference requests.
The dataset revision is `59b9fcba1aaed7bbf206b5f4d3c68b8face2f49c`.

## Start a new paired run

From `source/`, using the same Python environment:

```bash
../.venv/bin/python examples/who_and_when/run_fixed_pool_experiment.py prepare \
  --phase pilot --output-dir ../new_runs/pilot
../.venv/bin/python examples/who_and_when/run_fixed_pool_experiment.py run \
  --output-dir ../new_runs/pilot --max-cost-usd 4
../.venv/bin/python examples/who_and_when/run_fixed_pool_experiment.py prepare \
  --phase main --output-dir ../new_runs/main
../.venv/bin/python examples/who_and_when/run_fixed_pool_experiment.py run \
  --output-dir ../new_runs/main --pilot-dir ../new_runs/pilot --max-cost-usd 22
```

`prepare` downloads the pinned public dataset without invoking a model. `run`
requires `OPENROUTER_API_KEY` and incurs inference charges. Limits are cumulative
for each output directory. A completed pilot with the same frozen settings and
source hashes is required before the main run. Exact future predictions and
charges can vary with provider behavior.

## Source and artifact provenance

Requests, responses, streams, inputs, labels, predictions, datasets, frozen
protocols, run manifests, and result reports are byte-identical to the recorded
experiment. Source module/import paths, CLI descriptions, session labels, and
administrative paths were normalized for publication. The launcher uses a
relative checkout root. Internal planning documents were omitted. No scientific
prompts, evaluation logic, model settings, or recorded predictions were changed.

Each run's `source_publication.json` distinguishes executed source fingerprints
from published fingerprints. The frozen protocol retains its original execution
fingerprints and must not be rewritten to resume an old run with renamed source.
Use `report` for offline recomputation and `prepare` to create a fresh run.
The saved `artifact_audit.json` describes the original pre-publication audit;
`verify.py` checks the published package. Source snapshots for the earlier smoke
attempts are kept separately because their transport implementation differs.

Checksums for the published archive are regenerated at both run and bundle
levels. The bundle's `publication.json` records administrative changes and the
original bundle fingerprint. API keys and `.env` files are not included.
