# Reproduction artifacts

| Material | Contents | Start here |
| --- | --- | --- |
| Controlled adaptive versus fixed judge-pool comparison | All 126 Algo traces, individual judge findings, final decisions, prompts, labels, API responses, metrics, and costs | [fixed_pool_comparison/README.md](fixed_pool_comparison/README.md) |
| Who&When attribution metric reproduction | The 58 retained Hand predictions, reference labels, and standalone scoring script for the Table 10 run | [metric_check/README.md](metric_check/README.md) |

The experiment uses the shared library in [`src/autojudge`](../src/autojudge/).
Its [runner](../examples/who_and_when/run_fixed_pool_experiment.py) and
[experiment-specific code](../examples/who_and_when/fixed_pool/) are maintained
in the normal repository layout. This directory contains the reproduction
materials; it includes no response drafts or separate library copies.

Run the offline checks from the repository root with Python 3:

```bash
python examples/who_and_when/fixed_pool/verify.py
python rebuttal/metric_check/evaluate.py
```

Both commands use the Python standard library and require no API access.
The package READMEs describe the scoring conventions, provenance, and optional
full report recomputation for each experiment.
