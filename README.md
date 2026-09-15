---
title: AutoJudge
emoji: ⚖️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
fullWidth: true
header: mini
short_description: Research workspace for evaluating multi-agent execution traces with LLM judges.
---

# AutoJudge: Dynamic Generation of LLM Judge Pipelines for Multi-Agent System Evaluation

Official anonymous repository for the paper:

> **AutoJudge: Automatic Generation of LLM-Based Judges from Execution Traces**

---

# Overview

Modern LLM-based multi-agent systems (MAS) generate complex execution traces containing:

- reasoning steps,
- tool calls,
- agent interactions,
- environment feedback,
- planning decisions,
- orchestration logic.

Evaluating such systems is difficult because existing judge frameworks are typically:

- benchmark-specific,
- prompt-static,
- manually engineered,
- difficult to generalize across MAS architectures.

Most current approaches rely on fixed judge pipelines manually designed for a single benchmark, taxonomy, or trace format.

---

# AutoJudge

AutoJudge introduces a different paradigm:

> **judge pipelines are generated dynamically at inference time.**

Instead of using a predefined evaluator, AutoJudge automatically synthesizes a tailored multi-judge evaluation pipeline conditioned on:

- execution traces,
- evaluation taxonomies,
- desired output schemas.

The framework adapts automatically to:

- arbitrary MAS architectures,
- arbitrary evaluation criteria,
- arbitrary failure taxonomies,
- arbitrary structured outputs.

No benchmark-specific prompt engineering or retraining is required.

---

# Core Idea

Traditional evaluation:

```text
Fixed Prompt / Fixed Judge
            ↓
      Fixed Evaluation
```

AutoJudge:

```text
Trace + Taxonomy + Output Schema
                  ↓
              MetaAgent
                  ↓
     Dynamic Judge Pipeline
                  ↓
       Structured Evaluation
```

The evaluation logic itself becomes adaptive.

---

# What Makes AutoJudge Different

## Dynamic Pipeline Generation

AutoJudge dynamically generates:

- specialized intermediate judges,
- decomposition strategies,
- aggregation judges,
- reasoning chains,
- schema-constrained outputs.

Pipelines are synthesized either:

- per trace,
- per dataset,
- or per benchmark.

---

## Adaptation to Arbitrary Evaluation Criteria

Users are not restricted to predefined metrics.

Instead, they provide:

1. an evaluation taxonomy,
2. a desired output schema.

AutoJudge constructs the evaluation pipeline automatically.

Example taxonomy:

```python
taxonomy = """
├── Reasoning Errors
│   ├── Hallucinations
│   │   ├── Language-only
│   │   └── Tool-related
│   ├── Information Processing
│   │   ├── Poor Information Retrieval
│   │   └── Tool Output Misinterpretation
│   ├── Decision Making
│   │   ├── Incorrect Problem Identification
│   │   └── Tool Selection Errors
│   └── Output Generation
│       ├── Formatting Errors
│       └── Instruction Non-compliance
├── System Execution Errors
│   ├── Configuration
│   ├── API Issues
│   └── Resource Management
├── Planning and Coordination Errors
│   ├── Context Management
│   └── Task Management
"""
```

Example output schema:

```python
output_schema = """
{
    "errors": [
        {
            "category": "...",
            "location": "...",
            "evidence": "...",
            "description": "...",
            "impact": "HIGH|MEDIUM|LOW"
        }
    ],
    "scores": [
        {
            "reliability_score": 0-5,
            "security_score": 0-5,
            "instruction_adherence_score": 0-5,
            "plan_opt_score": 0-5,
            "overall": 0-5
        }
    ]
}
"""
```

AutoJudge then generates:

- specialized judges for taxonomy branches,
- aggregation logic,
- structured reasoning,
- validation pipelines,
- schema-constrained outputs.

Additional schema examples are available in:

```text
src/autojudge/meta_agents/prompts/output_schema_prompts/
```

---

# Generalization Across Arbitrary MAS Architectures

AutoJudge supports heterogeneous systems including:

- ReAct-style agents,
- tool-augmented agents,
- coding agents,
- browser/web agents,
- planning agents,
- debate-based MAS,
- orchestration pipelines,
- heterogeneous multi-agent systems.

The framework dynamically adapts evaluation logic to the structure of the trace itself.

---

# Long-Context Trace Evaluation

AutoJudge supports:

- full-trace evaluation,
- summary + retrieval evaluation.

For ultra-long traces:

- traces are summarized,
- raw trajectories are stored externally,
- judges retrieve relevant steps dynamically.

This avoids:

- context truncation,
- "lost in the middle" failures,
- prohibitive full-context costs.

---

# Structured Multi-Judge Reasoning

Evaluation is decomposed into multiple stages:

1. intermediate specialized judges,
2. evidence extraction,
3. aggregation reasoning,
4. schema validation.

The final aggregation judge synthesizes:

- trace evidence,
- intermediate outputs,
- taxonomy structure,
- schema requirements.

---

# Benchmarks

AutoJudge is evaluated across heterogeneous MAS trace benchmarks:

| Benchmark | Evaluation Objective |
|---|---|
| Who&When | Agent and step attribution |
| TRAIL | Failure localization |
| Aegis | Coordination and validation |
| AgentErrorBench | Fine-grained error categorization |
| AgentRewardBench | Reward-style evaluation |
| Pumpkin | Ensemble trajectory classification |

These benchmarks span:

- QA agents,
- coding systems,
- planning agents,
- browser agents,
- debate systems,
- heterogeneous orchestration pipelines.

---

# Repository Structure

```text
.
├── src/autojudge/
│   ├── meta_agents/         # Dynamic pipeline generation
│   ├── judges/              # Judge execution
│   ├── aggregation/         # Aggregation logic
│   ├── schemas/             # Structured outputs
│   ├── retrieval/           # Long-trace retrieval
│   ├── summarization/       # Trace summarization
│   └── pipelines/           # Runtime-generated judge graphs
│
├── benchmarks/              # Benchmark adapters
├── experiments/             # Experiment scripts
├── configs/                 # Configurations
└── README.md
```

---

# Installation

Clone repository:

```bash
git clone https://anonymous.4open.science/r/AutoJudge-6644/
cd AutoJudge-6644
```

Create environment:

```bash
conda create -n autojudge python=3.11
conda activate autojudge
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Experimental Findings

AutoJudge demonstrates strong transferability across:

- evaluation taxonomies,
- trace formats,
- judge objectives,
- MAS architectures.

Key findings include:

- dynamic judge generation outperforms static judge reuse,
- adaptive pipelines generalize without retraining,
- summary + retrieval mode scales to ultra-long traces,
- structured multi-judge reasoning improves robustness.

---

# Reproducibility

The repository includes:

- prompts,
- benchmark adapters,
- evaluation schemas,
- experiment configurations,
- runtime-generated pipelines,
- trace processing utilities.

All major experiments from the paper are reproducible using the provided scripts.

---

# Citation

```bibtex
@article{anonymous2026autojudge,
  title={AutoJudge: Automatic Generation of LLM-Based Judges from Execution Traces},
  author={Anonymous Authors},
  journal={Under Review},
  year={2026}
}
```

---

# License

Anonymous repository for double-blind review.

Code and assets will be publicly released upon publication.
