"""
count_dataset_tokens.py

Estimates LLM-as-a-judge input and output token usage for evaluation cost projection.

Datasets supported:
- who_and_when  -> Kevin355/Who_and_When (Algorithm-Generated.parquet, public)
- trail         -> PatronusAI/TRAIL (gated; requires HF auth)
- aegis         -> Fancylalala/AEGIS (public)

Usage:
    uv run python examples/count_dataset_tokens.py --dataset who_and_when
    uv run python examples/count_dataset_tokens.py --dataset trail --sample 20
    uv run python examples/count_dataset_tokens.py --dataset who_and_when \\
        --model gemini-2.5-flash --price-per-1m-input 0.3 --price-per-1m-output 2.5 --static-overhead 1500
    uv run python examples/count_dataset_tokens.py --dataset aegis \\
        --model cl100k_base --price-per-1m-input 2.5 --price-per-1m-output 10

`--static-overhead` is the fixed token count of taxonomy + output_schema +
examples that the launch scripts prepend to every judge call. Add it if you
want a realistic per-call total instead of just the per-trace content.

Output token estimates are based on typical judge outputs (verdict schemas).

Token counting: uses Gemini 2.5 Flash by default (requires GOOGLE_API_KEY env var).
Fallback: tiktoken encodings or char/4 estimate.
"""

import argparse
import json
import logging
import os
import statistics
from pathlib import Path
from typing import Any, Callable, Iterable

from dotenv import load_dotenv

load_dotenv(".env")

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logger = logging.getLogger("count_dataset_tokens")


# ---------- text builder --------------------------------------------------


def build_text_generic(row: dict) -> str:
    """Concatenate all string-ish fields. For datasets with unknown schemas."""
    parts = []
    for key, value in row.items():
        if value is None:
            continue
        if isinstance(value, str):
            text = value
        elif isinstance(value, (int, float, bool)):
            continue
        else:
            try:
                text = json.dumps(value, ensure_ascii=False, default=str)
            except Exception:
                text = str(value)
        parts.append(f"[{key}]\n{text}")
    return "\n\n".join(parts)


# dataset loaders
def load_who_and_when(sample: int | None) -> Iterable[dict]:
    from datasets import load_dataset

    logger.info("Loading Kevin355/Who_and_When (Algorithm-Generated)")
    ds = load_dataset(
        "Kevin355/Who_and_When",
        # data_files="Algorithm-Generated.parquet",
        data_files="Hand-Crafted.parquet",
        split="train",
    )
    if sample is not None:
        ds = ds.select(range(min(sample, len(ds))))
    logger.info(f"Loaded {len(ds)} rows")
    return list(ds)


def load_trail(sample: int | None) -> Iterable[dict]:
    from datasets import get_dataset_split_names, load_dataset

    logger.info("Loading PatronusAI/TRAIL (gated; requires HF auth)")
    try:
        splits = get_dataset_split_names("PatronusAI/TRAIL")
    except Exception as exc:
        raise RuntimeError(
            "Could not enumerate TRAIL splits. PatronusAI/TRAIL is gated — "
            "make sure you have accepted its terms on Hugging Face and that "
            "HF_TOKEN is set in your environment.\n"
            f"Original error: {exc}"
        ) from exc

    logger.info(f"TRAIL splits: {splits}")
    rows: list[dict] = []
    for split in splits:
        ds = load_dataset("PatronusAI/TRAIL", split=split)
        logger.info(f"  split={split}: {len(ds)} rows")
        if sample is not None:
            remaining = sample - len(rows)
            if remaining <= 0:
                break
            ds = ds.select(range(min(remaining, len(ds))))
        rows.extend(list(ds))
    logger.info(f"Loaded {len(rows)} rows from TRAIL")
    return rows


def load_aegis(sample: int | None) -> Iterable[dict]:
    from datasets import load_dataset

    logger.info("Loading Fancylalala/AEGIS")
    ds = load_dataset("Fancylalala/AEGIS", split="test")
    if sample is not None:
        ds = ds.select(range(min(sample, len(ds))))
    logger.info(f"Loaded {len(ds)} rows")
    return list(ds)


DATASETS: dict[str, dict[str, Any]] = {
    "who_and_when": {
        "loader": load_who_and_when,
        "build_text": build_text_generic,
    },
    "trail": {
        "loader": load_trail,
        "build_text": build_text_generic,
    },
    "aegis": {
        "loader": load_aegis,
        "build_text": build_text_generic,
    },
}


# ---------- tokenization --------------------------------------------------


def get_token_counter(model: str) -> tuple[Callable[[str], int], str]:
    """Get token counter for the specified model.

    Supports:
    - gemini-2.5-flash: uses google.generativeai
    - other tiktoken encodings: uses tiktoken (e.g., cl100k_base for GPT-4)
    """
    if model.startswith("gemini"):
        try:
            from google import genai

            client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

            def count_tokens_gemini(text: str) -> int:
                try:
                    response = client.models.count_tokens(model=model, contents=text)
                    return response.total_tokens
                except Exception as e:
                    logger.warning(
                        f"Gemini token counting failed: {e}; falling back to len/4"
                    )
                    return max(1, len(text) // 4)

            return count_tokens_gemini, f"gemini/{model}"
        except ImportError:
            logger.warning("google-genai not installed; falling back to len(text)//4")
            return (lambda text: max(1, len(text) // 4)), "char/4 estimate"
    else:
        # Assume it's a tiktoken encoding name
        try:
            import tiktoken

            enc = tiktoken.get_encoding(model)
            return (
                lambda text: len(enc.encode(text, disallowed_special=()))
            ), f"tiktoken/{model}"
        except ImportError:
            logger.warning(
                "tiktoken not installed; falling back to len(text)//4 estimate"
            )
            return (lambda text: max(1, len(text) // 4)), "char/4 estimate"


# output token estimation


def estimate_output_tokens(dataset: str, count_fn: Callable[[str], int]) -> int:
    """
    Estimate typical output tokens for a dataset based on its output schema.
    Uses the provided count_fn to count example outputs (respects model's tokenization).
    """
    outputs = {
        "who_and_when": json.dumps(
            {
                "verdict": "fair",
                "agent": "AgentName",
                "justification": "Agent made a reasonable attempt but missed some edge case in the logic flow.",
            }
        ),
        "trail": json.dumps(
            {
                "errors": [
                    {
                        "category": "Insufficient Validation",
                        "location": "span-42",
                        "evidence": "System accepted invalid input without checking constraints",
                        "description": "The system failed to validate user input before processing, leading to potential data corruption.",
                        "impact": "HIGH",
                    },
                    {
                        "category": "Missing Error Handling",
                        "location": "span-67",
                        "evidence": "Exception thrown without recovery mechanism",
                        "description": "An error occurred during processing but the system had no recovery mechanism.",
                        "impact": "MEDIUM",
                    },
                ],
                "scores": [
                    {
                        "reliability_score": 3,
                        "reliability_reasoning": "System has some fault tolerance but lacks comprehensive error handling mechanisms.",
                        "security_score": 2,
                        "security_reasoning": "Input validation is missing in critical paths and authentication checks are incomplete.",
                        "instruction_adherence_score": 3,
                        "instruction_adherence_reasoning": "System partially follows specified instructions but deviates in error scenarios.",
                        "plan_opt_score": 3,
                        "plan_opt_reasoning": "Plan is reasonable but has redundant steps and inefficient resource allocation.",
                        "overall": 2.75,
                    }
                ],
            }
        ),
        "aegis": json.dumps(
            {
                "faulty_agents": [
                    {
                        "agent_name": "Solver",
                        "error_type": "FM-3.2",
                        "evidence": "The agent failed to validate the response against the constraint.",
                        "description": "Agent produced output that violated explicitly stated requirements without attempting verification.",
                    },
                    {
                        "agent_name": "Reviewer",
                        "error_type": "FM-1.5",
                        "evidence": "Review process did not catch obvious inconsistency in the response.",
                        "description": "Verification step was bypassed or performed superficially, missing critical issues.",
                    },
                ],
                "scores": [
                    {
                        "reliability_score": 3,
                        "reliability_reasoning": "System handled most cases correctly but had gaps in error recovery mechanisms.",
                        "coordination_score": 2,
                        "coordination_reasoning": "Agent communication had delays and some information was not properly shared between components.",
                        "verification_score": 2,
                        "verification_reasoning": "Verification mechanisms were present but not applied consistently across all critical paths.",
                        "overall": 2.33,
                    }
                ],
            }
        ),
    }

    if dataset not in outputs:
        logger.warning(f"No output estimate for {dataset}; using fallback (500 tokens)")
        return 500

    return count_fn(outputs[dataset])


def summarize(
    counts: list[int],
    output_tokens_per_trace: int,
    price_per_1m_input: float,
    price_per_1m_output: float,
    static_overhead: int,
) -> dict:
    if not counts:
        return {"traces": 0}
    counts_with_overhead = [c + static_overhead for c in counts]
    total_input = sum(counts_with_overhead)
    total_output = len(counts) * output_tokens_per_trace
    total_tokens = total_input + total_output

    input_cost = total_input / 1_000_000 * price_per_1m_input
    output_cost = total_output / 1_000_000 * price_per_1m_output
    total_cost = input_cost + output_cost

    return {
        "traces": len(counts),
        "static_overhead_per_call": static_overhead,
        "total_input_tokens": total_input,
        "mean_input_per_trace": round(statistics.mean(counts_with_overhead), 1),
        "median_input_per_trace": int(statistics.median(counts_with_overhead)),
        "min_input_per_trace": min(counts_with_overhead),
        "max_input_per_trace": max(counts_with_overhead),
        "stdev_input_per_trace": round(statistics.pstdev(counts_with_overhead), 1),
        "estimated_output_tokens_per_trace": output_tokens_per_trace,
        "total_output_tokens": total_output,
        "total_tokens": total_tokens,
        "input_cost_usd": round(input_cost, 4),
        "output_cost_usd": round(output_cost, 4),
        "total_cost_usd": round(total_cost, 4),
        "price_per_1m_input_usd": price_per_1m_input,
        "price_per_1m_output_usd": price_per_1m_output,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS.keys()))
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Limit to first N traces (faster, partial estimate).",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Model for token counting: 'gemini-2.5-flash' or tiktoken encoding (default: gemini-2.5-flash).",
    )
    parser.add_argument(
        "--price-per-1m-input",
        type=float,
        default=0.3,
        help="USD per 1M input tokens (default: 0.3 for Gemini-2.5-flash).",
    )
    parser.add_argument(
        "--price-per-1m-output",
        type=float,
        default=2.5,
        help="USD per 1M output tokens (default: 2.5 for Gemini-2.5-flash).",
    )
    parser.add_argument(
        "--static-overhead",
        type=int,
        default=0,
        help="Constant tokens added per judge call (taxonomy + output_schema + examples).",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to dump per-trace counts as JSON.",
    )
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    count_fn, backend = get_token_counter(args.model)
    logger.info(f"Token counter: {backend}")

    rows = cfg["loader"](args.sample)
    build_text = cfg["build_text"]

    counts: list[int] = []
    for row in rows:
        text = build_text(row)
        counts.append(count_fn(text))

    output_tokens = estimate_output_tokens(args.dataset, count_fn)
    stats = summarize(
        counts,
        output_tokens_per_trace=output_tokens,
        price_per_1m_input=args.price_per_1m_input,
        price_per_1m_output=args.price_per_1m_output,
        static_overhead=args.static_overhead,
    )
    stats["dataset"] = args.dataset
    stats["model"] = args.model
    stats["counter_backend"] = backend

    print(json.dumps(stats, indent=2))

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"stats": stats, "per_trace_input_tokens": counts}, indent=2)
        )
        logger.info(f"Wrote per-trace counts to {out}")


if __name__ == "__main__":
    main()
