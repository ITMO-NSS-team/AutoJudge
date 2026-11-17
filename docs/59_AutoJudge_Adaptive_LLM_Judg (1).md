# AutoJudge: Adaptive LLM-Judge Pools for Multi-Agent Evaluation

> Текст извлечён из локального PDF на английском языке; это не перевод. Сохранены основной текст, приложения, библиография и номера страниц. Заголовки и числовые таблицы оформлены в Markdown; сложные таблицы дополнены изображениями из PDF. При извлечении возможны артефакты переносов и математической нотации. Утверждения и расхождения оригинала сохранены без исправления по смыслу.

Источник: [PDF](<59_AutoJudge_Adaptive_LLM_Judg (1).pdf>), 3 страницы.

<!-- PDF page 1 -->

Alina Zhidkovskaya1, Kirill Rapatskikh1, Grigorii Barakhsin1, Alexander Serebryakov1, Rodion Berkovich1, Jerzy Kamiński1, Julia Belikova2, Maksim Makarenko1,2, Andrey Savchenko1,2, Anna Kalyuzhnaya1, Nikolay Nikitin1

1ITMO University, Saint Petersburg, Russia; 2Sb AI Lab, Moscow, Russia

## Abstract

Evaluating LLM-based multi-agent systems from execution traces is hard: traces are long, failure modes are diverse, and existing judges are hand-built per benchmark. We demonstrate AutoJudge, a system that builds a trace-conditioned pool of LLM judges inside a fixed evaluation graph. Given a trace, a taxonomy, and an output schema, a MetaAgent writes the roles and instructions of several intermediate judges and one aggregation judge, which then execute to produce a structured, evidence-grounded verdict. AutoJudge handles traces beyond standard context windows through a summary-plus-retrieval mode. Code: https://github.com/ITMO-NSS-team/AutoJudge; demo: https://huggingface.co/spaces/jrzkaminski/autojudge-demo; video: https://youtu.be/I-8M8lY7VAQ.

## AutoJudge

Why. Coordination, tool-use, and intermediate-reasoning failures in multi-agent systems are invisible to end-to-end success metrics, so trace-level evaluation is needed – but today this means hand-designing a judge for every new benchmark, taxonomy, and trace format. New agent benchmarks now appear faster than custom judges can be built by hand.

What AutoJudge does. AutoJudge removes this per-benchmark engineering step. A researcher supplies only an evaluation taxonomy and an output schema; the system adapts its judge pool to the trace at hand while keeping the evaluation graph–several intermediate judges running in parallel and feeding one aggregation judge – fixed and inspectable in every configuration we report. A MetaAgent (Gemini 3 Flash Preview) writes each judge’s role and instructions from the trace, taxonomy, and schema; the judges and aggregator (Gemini 2.5 Flash) are then executed as schema-constrained calls with automatic retries on format violations, ensuring robust and consistent structured outputs. As shown in Figure 1, the MetaAgent runs in one of two modes – a dedicated pipeline per trace (default), or one pipeline generated from a small batch and reused across a dataset (budget mode) – trading off cost against per-trace adaptation (see Cost vs. accuracy).

Scaling to long traces. Full-trace mode passes the raw trace to every judge for maximal fidelity. For traces beyond standard context windows (we have processed traces past 2M tokens), a summary mode compresses each step, stores the full trace in PostgreSQL, and lets judges retrieve individual steps by identifier on demand.

## Demonstration

At the booth, visitors pick a preset GAIA trace or upload their own JSON, set an evaluation objective with a markdown taxonomy and structured output schema, and launch AutoJudge (demo: https://huggingface.co/spaces/jrzkaminski/autojudge-demo; video: https://youtu.be/I-8M8lY7VAQ). The interface visualizes the generated judge pipeline as a directed graph – named judges like source_factuality and constraint_completion feed into a final_aggregator – validates the design before execution, and surfaces each run’s structured verdict on the Runs page, making the path from raw trace through the judge pipeline to final verdict directly inspectable rather than a black box.

![demo-figure-1](images/papers/demo-figure-1.png)

Figure 1: Per-trace vs. per-batch pipeline generation: the MetaAgent either builds one dedicated pipeline per trace, or observes a small batch of traces and generates a single reusable pipeline – trading off per-trace adaptation against cost (Cost vs. accuracy).

---

<!-- PDF page 2 -->

![demo-figure-2](images/papers/demo-figure-2.png)

Figure 2: AutoJudge Workbench: the web interface provides an overview of evaluation runs, completed pipelines, and their verdicts, allowing researchers to launch new evaluations and inspect results.

Table 1: Controlled Who&When comparison; all systems use Gemini 2.5 Flash.

| System | Hand Agent | Hand Step | Algo Agent | Algo Step |
| --- | --- | --- | --- | --- |
| WW judge (all-at-once) | 58.62 | 3.45 | 66.67 | 28.57 |
| WW judge (step-by-step) | 44.83 | 10.34 | 39.68 | 28.57 |
| AutoJudge (summ + DB) | 61.49 | 17.25 | 55.55 | 41.23 |
| AutoJudge (full trace) | 59.19 | 12.07 | 63.23 | 43.38 |


## Evaluation

Controlled comparison. Table 1 isolates pipeline design from backbone choice on Who&When (Zhang et al. 2025): every system below uses Gemini 2.5 Flash as its judge model. AutoJudge leads on step accuracy in every setting and on hand-subset agent accuracy; the paper judge’s all-at-once mode remains stronger on algo-subset agent accuracy (66.67 vs. 63.23), a result we report as-is rather than omit.

Across benchmarks. On seven benchmarks – Who&When (Zhang et al. 2025), TRAIL (Deshpande et al. 2025), Aegis (Kong et al. 2026), AgentErrorBench (Zhu et al. 2025), AgentRewardBench (Lù et al. 2025), AgenTracer (Zhang et al. 2026), and Pumpkin (Zhidkovskaya et al. 2026) – spanning attribution, localization, error categorization, and reward-style judging, AutoJudge outperforms benchmark-specific evaluators on 14 of 19 within-benchmark metrics; these comparisons use each benchmark’s own metric, backbone, and protocol, so we report per-benchmark wins rather than an aggregate ranking.

Cost vs. accuracy. Generating one judge pool per trace (Figure 1, upper) is not uniformly more cost-efficient than reusing one pool generated from a batch of traces (Figure 1, lower). On WW Hand, per-trace generation costs $2.18 for 28.2 agent-accuracy points per dollar, vs. $1.82 and 31.8 points per dollar for dataset-level reuse. On WW Algo, where traces are more heterogeneous, per-trace generation wins on both accuracy and points per dollar (23.5 vs. 23.2). Dataset-level reuse is the better default only when traces are homogeneous and budget is tight.

## Related Work

Unlike automatic prompt-engineering methods such as APE (Zhou et al. 2022), OPRO (Yang et al. 2024), and DSPy (Khattab et al. 2023), which iteratively optimize instructions or pipeline parameters against a metric and labeled development examples, AutoJudge performs trace-conditioned judge-pool construction at inference time, with no optimization loop and no labeled development data required. It also differs from benchmark-specific judges such as Who&When (Zhang et al. 2025) and TRAIL (Deshpande et al. 2025), which fix both the evaluation criteria and the judge architecture to one benchmark: AutoJudge instead takes the taxonomy and schema as input, so the same graph serves a new benchmark without being redesigned by hand.

## Limitations

Validation relies on benchmark ground truth rather than a dedicated human-agreement study, and causal attribution can still confuse root cause with manifestation. Highly domain-specific criteria can still benefit from a manually written judge.

---

<!-- PDF page 3 -->

## References

Deshpande, D.; Gangal, V.; Mehta, H.; Krishnan, J.; Kannappan, A.; and Qian, R. 2025. Trail: Trace reasoning and agentic issue localization. arXiv preprint arXiv:2505.08638. Khattab, O.; Singhvi, A.; Maheshwari, P.; Zhang, Z.; Santhanam, K.; Vardhamanan, S.; Haq, S.; Sharma, A.; Joshi, T. T.; Moazam, H.; et al. 2023. Dspy: Compiling declarative language model calls into self-improving pipelines. arXiv preprint arXiv:2310.03714. Kong, F.; Zhang, R.; Yin, H.; Zhang, G.; Zhang, X.; Chen, Z.; Zhang, Z.; Zhang, X.; Zhu, S.-C.; and Feng, X. 2026. Aegis: Automated error generation and attribution for multi-agent systems. In International Conference on Learning Representations, volume 2026, 52994–53028. Lù, X. H.; Kazemnejad, A.; Meade, N.; Patel, A.; Shin, D.; Zambrano, A.; Stańczak, K.; Shaw, P.; Pal, C. J.; and Reddy, S. 2025. Agentrewardbench: Evaluating automatic evaluations of web agent trajectories. arXiv preprint arXiv:2504.08942. Yang, C.; Wang, X.; Lu, Y.; Liu, H.; Le, Q. V.; Zhou, D.; and Chen, X. 2024. Large language models as optimizers. In International Conference on Learning Representations, volume 2024, 12028–12068. Zhang, G.; Wang, J.; Chen, J.; Zhou, W.; Wang, K.; and Yan, S. 2026. Agentracer: Who is inducing failure in the llm agentic systems? In International Conference on Learning Representations, volume 2026, 11377–11399. Zhang, S.; Yin, M.; Zhang, J.; Liu, J.; Han, Z.; Zhang, J.; Li, B.; Wang, C.; Wang, H.; Chen, Y.; et al. 2025. Which agent causes task failures and when? on automated failure attribution of llm multi-agent systems. arXiv preprint arXiv:2505.00212. Zhidkovskaya, A.; Rapatskikh, K.; Kaminski, J.; Barakhsin, G.; Kalyuzhnaya, A. V.; Druzhinin, A.; Savchenko, A.; Belikova, J.; Polev, K.; and Nikitin, N. O. 2026. Towards Design of an Automated Judge for Multi-Agent Systems. In Trustworthy Agentic AI Workshop at the AAAI Conference on Artificial Intelligence (AAAI). Zhou, Y.; Muresanu, A. I.; Han, Z.; Paster, K.; Pitis, S.; Chan, H.; and Ba, J. 2022. Large language models are human-level prompt engineers. arXiv preprint arXiv:2211.01910. Zhu, K.; Liu, Z.; Li, B.; Tian, M.; Yang, Y.; Zhang, J.; Han, P.; Xie, Q.; Cui, F.; Zhang, W.; et al. 2025. Where llm agents fail and how they can learn from failures. arXiv preprint arXiv:2509.25370.


