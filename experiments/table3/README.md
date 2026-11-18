# Reconstructed few-shot experiment

The six conditions vary the examples supplied to the MetaAgent. Every condition generates six intermediate judges and uses a fixed aggregator, full traces, tools disabled, and Gemini 2.5 Flash throughout. There are 58 Hand development traces per condition, one generation repeat and seed 20261118. Two unavailable examples are explicitly reconstructed in `reconstructed_examples.json`.

`source/` preserves the exact executed Python source; hashes are recorded in the campaign manifests. `requirements-recorded.txt` records direct runtime package versions (Python 3.11.9, PydanticAI 1.89.1). The first pass and declared technical-recovery pass are preserved separately, including transport overrides and cost reservations.

Start with [the reproduction instructions](../../rebuttal/table3_reconstruction/README.md). Saved scores can be independently verified with Python's standard library and no API access. That package also documents fresh paid inference, full receipt restoration, uniform recovery and combined cost accounting.
