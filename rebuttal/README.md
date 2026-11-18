# Reproduction artifacts

| Material | Contents | Start here |
| --- | --- | --- |
| Reconstructed few-shot ablation (revised Table 4) | All six conditions, 348 verified predictions, first-pass/recovery provenance, independent scorer and complete receipt archives | [table3_reconstruction/README.md](table3_reconstruction/README.md) |
| Controlled adaptive versus fixed judge-pool comparison | All 126 Algo traces, individual judge findings, final decisions, prompts, labels, API responses, metrics, and costs | [fixed_pool_comparison/README.md](fixed_pool_comparison/README.md) |
| Historical summary-only versus summary + DB comparison | Brief report of retained aggregate results and their limitations | [summary_retrieval/README.md](summary_retrieval/README.md) |
| Who&When attribution metric reproduction | The 58 retained Hand predictions, reference labels, and standalone scoring script for the Table 10 run | [metric_check/README.md](metric_check/README.md) |

The fixed-pool experiment uses the shared library in [`src/autojudge`](../src/autojudge/).
Its [runner](../examples/who_and_when/run_fixed_pool_experiment.py) and
[experiment-specific code](../examples/who_and_when/fixed_pool/) are maintained
in the normal repository layout. This directory contains the reproduction materials. The reconstructed few-shot study additionally preserves its exact executed source snapshot and declares its recovery protocol.

Run the offline checks from the repository root with Python 3:

```bash
python examples/who_and_when/fixed_pool/verify.py
python rebuttal/metric_check/evaluate.py
python rebuttal/table3_reconstruction/verify.py
```

These commands use the Python standard library and require no API access.
The package READMEs describe the scoring conventions, provenance, and optional
full report recomputation for each experiment.
