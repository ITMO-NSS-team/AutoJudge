# Campaign provenance

- [AEGIS](https://huggingface.co/datasets/Fancylalala/AEGIS), official 600-case test split, revision `b936269a5143e4450242d8628bff026bdc5bae18`.
- [TRAIL](https://github.com/patronus-ai/trail-benchmark), official GitHub snapshot `0ffbed9db859b4a66250dc783fa4dccf86869595`, all 148 GAIA and SWE Bench traces. The adapted target is span localization, scored by mean set F1.
- [AgentErrorBench](https://github.com/ulab-uiuc/AgentDebug), annotations and trajectories from the linked dataset folder. Repository revision `7740fe3a5c4822b2143cbde78ecfffeace0bb166`. One annotated GAIA step is beyond the observed turns and is excluded in both arms. The adapted targets are step and module.
- [Who&When](https://huggingface.co/datasets/Kevin355/Who_and_When), revision `59b9fcba1aaed7bbf206b5f4d3c68b8face2f49c`, compatible Hand and Algo cases. Use the case lists next to this file to select the exact rows.
- [AgentRewardBench](https://huggingface.co/datasets/McGill-NLP/agent-reward-bench), revision `b6d17e646009d6cb63d5dd7be78807b680693f61`. The WebArena list records one trajectory per eligible official test task. Eligibility requires a single annotation for the selected model–task pair; the choice of trajectory per task used a deterministic hash.
- Pumpkin requires source execution traces plus independent human labels. Neither was available for this run. No Pumpkin figures are reported.

Metrics are kept per benchmark. The weights and unit of analysis of F1 differ between the benchmark specs. These paired comparisons use direct Jev, while the archived LLM-graph baseline from the original project is a separate run.
