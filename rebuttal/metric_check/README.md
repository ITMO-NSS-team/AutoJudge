# Who&When attribution metric reproduction

This package reproduces the retained 58-trace Hand run corresponding to the
Gemini 2.5 Flash / Gemini 2.5 Flash row in Table 10. It documents failure-agent
attribution and exact failure-step localization, not binary task success.

Run with Python 3; no extra packages, network access or API key are needed:

```bash
python evaluate.py
```

Expected results: 37/58 = 63.79% Agent Accuracy; 5/58 = 8.62% Exact Step Accuracy.

Agent names are compared case-insensitively with whitespace ignored. This handles
formatting differences such as `WebSurfer` versus `Websurfer`, which identify the
same agent. The checker does not use substring or semantic matching and does not
remove underscores. Predictions use one-based steps; dataset reference steps are
zero-based. Every manifest task remains in the denominator, and missing or invalid
predictions count as incorrect.

Files:
- `predictions.jsonl`: all 58 retained agent/step/reason prediction objects.
- `references.csv`: agent and step labels for the same 58 public benchmark IDs.
- `manifest.json`: fixed task list, benchmark revision and source-file hashes.
- `evaluate.py`: transparent offline metric calculation and checksum verification.
- `SHA256SUMS.json`: integrity hashes for all other included files.

Benchmark source: https://huggingface.co/datasets/Kevin355/Who_and_When
Revision: 59b9fcba1aaed7bbf206b5f4d3c68b8face2f49c.
The chosen run covers all 58 Hand tasks and reproduces an existing published row;
predictions were not combined across runs or selected individually by correctness.
The original prediction objects are retained unchanged; unrelated metadata is omitted.

Scope: this is a reproduction of one retained Table 10 run, not a reconstruction
of all six prompting conditions in Table 3. Historical API model metadata and
tool-call histories are not available in these verdict-only artifacts. This package
does not establish historical aggregator tool permissions or retrieval usage.
