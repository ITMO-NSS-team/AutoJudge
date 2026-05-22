import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import asyncio
import os
from dotenv import load_dotenv

load_dotenv(".env")

from autojudge.utils import get_logger
import json
import pandas as pd
from toon_format import encode
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel

logger = get_logger(__name__)

taxonomy = """
1) Guilty agent
2) Step of error
"""

output_schema = """
**OUTPUT FORMAT - STRICTLY REQUIRED:**
You must determine the most guilty agent in the evaluated 'history_for_evaluating', based on what the other judges wrote. You MUST return ONLY a valid JSON object with exactly these three fields:
{
  "agent": "Guilty Agent name from trace here",
  "step": "integer number (1, 2, 3...) of the message in trace sequence",
  "reason": "reason of your prediction"
}

**CRITICAL RULES:**
- Return ONLY the JSON object, nothing else
- NO markdown code fences (no ```json or ```)
- NO explanatory text before or after the JSON
- NO additional fields (no score, no confidence, no metadata)
- agent must be the exact agent name as it appears in the trace
- step must be a plain integer string: "1", "2", "3", etc.

**STEP RULE:** This is the sequential position of the message in 'history_for_evaluating' (1 = first message, 2 = second message, etc). Use ONLY integers. Do NOT use IDs, UUIDs, strings, or any other identifiers.

**VALID EXAMPLE:**
{
  "agent": "File_Surfer",
  "step": "1", 
  "reason": "The agent fails to collect price data for the daily tickets and season passes for California's Great America in 2024."
}


INVALID EXAMPLES (DO NOT USE THIS FORMAT):
```json
{
  "agent": "Orchestrator",
  "step": "21",
  "reason": "The Orchestrator is the most guilty agent. Despite the WebSurfer's repeated failures to find clear Vudu listings for 'The Tenant' and 'Nosferatu the Vampyre' (as noted in steps 13 and 17), and the subsequent 'ResponsibleAIPolicyViolation' error in step 21, the Orchestrator still allowed the final answer to be 'The Tenant' without any verified evidence of its availability on Vudu. This indicates a failure in the Orchestrator's decision-making process to ensure all constraints were met before providing a final answer. The Orchestrator also repeatedly asked the WebSurfer to check for Vudu availability without changing its strategy, indicating a lack of progress and looping, as highlighted by the Search Integrity Judge."
}
```

**INVALID STEP EXAMPLES:** "step1", "abc-123", "task_id_45", "first" — ONLY USE: "1", "2", "3", etc.
"""


def build_single_judge_prompt(judge_input: dict) -> str:
    return f"""Evaluate the trace using only this taxonomy:
{taxonomy}

Follow this strict output schema:
{output_schema}

Choose one most guilty agent and the exact step where the key mistake happened.
Return ONLY valid JSON with this format:
{{"agent":"<exact agent name from trace>","step":"<1-based integer as string>","reason":"<short reason>"}}

query: {judge_input["query"]}
history_for_evaluating: {judge_input["history_for_evaluating"]}
"""


async def run_single_judge(agent: Agent, prompt: str, max_attempts: int = 3) -> str:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = await agent.run(prompt)
            raw_output = str(response.output).strip()
            if raw_output:
                return raw_output
            raise ValueError("Empty model output")
        except Exception as exc:
            last_error = exc
            logger.warning(
                f"Single judge attempt {attempt}/{max_attempts} failed: {exc}"
            )
    if last_error is not None:
        raise last_error
    raise RuntimeError("Single judge failed without explicit exception")


async def main(
    save_folder: str, df, num_traces: int | None = None
):
    logger.info("===Starting Who&When evaluation===")

    judge_model = OpenAIChatModel(
        "google/gemini-2.5-flash",
        provider="openrouter",
        settings={
            "temperature": 0.0,
            "reasoning": False,
        },
    )
    single_judge_agent = Agent(
        name="WHO_AND_WHEN_SINGLE_JUDGE",
        model=judge_model,
        instrument=True,
        retries=3,
    )
    logger.info("Initialized single judge agent")

    # continue processing that was already started
    done_traces = []
    # local_results_dir = Path(__file__).resolve().parent / "results" / save_folder
    local_results_dir = Path(__file__).resolve().parent / "single_agent_results(for_paper)" / save_folder

    results_dir = local_results_dir

    if results_dir.exists():
        for res in os.listdir(results_dir):
            res_cropped = res.split(".")[0]
            done_traces.append(res_cropped)

    # skip failed traces
    failed_traces_ids = []
    failed_traces = []

    if local_results_dir.exists():
        if "failed_traces.txt" in os.listdir(local_results_dir):
            with open(local_results_dir / "failed_traces.txt", "r") as f:
                for line in f:
                    if line.startswith("Task ID:"):
                        failed_traces_ids.append(line.split(":")[1].strip())

    if num_traces is not None:
        df = df[:num_traces]

    for idx in range(len(df)):
        id = df.iloc[idx]["question_ID"]
        if id in done_traces:
            question_id = id
            logger.info(f"Task {question_id} already processed, skipping...")
            continue

        if id in failed_traces_ids:
            logger.info(f"Task {id} already failed, skipping...")
            continue

        logger.info(f"Processing task {idx + 1}/{len(df)}: {id}")
        serializable_results = {}

        try:
            trace_data = {
                "history": df.iloc[idx]["history"],
                "question": df.iloc[idx]["question"],
                "task_id": id,
                "trace_id": id,
            }

            trace_metadata = {
                "task_id": trace_data["task_id"],
                "trace_id": trace_data["trace_id"],
            }

            if "groundtruth" in df.iloc[idx].keys():
                trace_metadata["ground_truth"] = df.iloc[idx]["groundtruth"]
                trace_metadata["correct_answer"] = df.iloc[idx]["is_corrected"]
            else:
                trace_metadata["ground_truth"] = df.iloc[idx]["ground_truth"]
                trace_metadata["correct_answer"] = df.iloc[idx]["is_correct"]

            judge_input = {
                "query": encode(trace_data["question"]),
                "history_for_evaluating": trace_data["history"],
            }

            prompt = build_single_judge_prompt(judge_input)

            logger.info("Executing single-agent evaluation...")
            result = await run_single_judge(single_judge_agent, prompt)
            logger.info("Single-agent evaluation completed")

            result_dict = json.loads(
                result.replace("```json", "").replace("```", "").strip()
            )

            serializable_results["summarizer_score"] = {
                "metric_name": "summarizer_score",
                "scores": [
                    {
                        "item_id": "overall_score",
                        "score": result_dict,
                        "idx": idx,
                        "task_id": id,
                        "ground_truth": trace_metadata["ground_truth"],
                        "correct_answer": str(trace_metadata["correct_answer"]),
                        "gt_agent": df.iloc[idx]["mistake_agent"],
                        "gt_step": df.iloc[idx]["mistake_step"],
                        "gt_mistake_reason": df.iloc[idx]["mistake_reason"],
                    }
                ],
            }

            output_dir = local_results_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / Path(f"{df.iloc[idx]['question_ID']}.json")

            with open(output_file, "w") as f:
                json.dump(serializable_results, f, indent=2)

            logger.info(f"\nResults saved to: {output_file}\n\n")

            print(f"Result: {result}")

        except Exception as e:
            error_msg = str(e)
            safe_error_msg = error_msg.replace("{", "{{").replace("}", "}}")
            task_id = id
            logger.error(
                f"Error processing task {task_id}: {safe_error_msg}", exc_info=True
            )

            failed_traces.append(
                {
                    "task_id": task_id,
                    "task_index": idx + 1,
                    "error": error_msg,
                    "error_type": type(e).__name__,
                }
            )

            print(f"\n!  Failed task {idx + 1}/{len(df)}: {task_id}")
            print(f"   Error: {error_msg}\n")
            continue

    output_dir = local_results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    failed_file = output_dir / "failed_traces.txt"

    if failed_traces:
        first_run = not failed_file.exists()
        with open(failed_file, "a") as f:
            if first_run:
                f.write(f"Failed traces: {len(failed_traces)} out of {len(df)}\n")
                f.write("=" * 80 + "\n\n")

            for failed in failed_traces:
                f.write(f"Task ID: {failed['task_id']}\n")
                f.write(f"Index: {failed['task_index']}/{len(df)}\n")
                f.write(f"Error Type: {failed['error_type']}\n")
                f.write(f"Error Message: {failed['error']}\n")
                f.write("-" * 80 + "\n\n")

    if len(failed_traces) > 0:
        logger.warning(
            f"\n!  {len(failed_traces)} traces failed. Details saved to: {failed_file}\n"
        )

    if failed_traces_ids:
        logger.info(
            f"Completed evaluation: {len(df) - (len(failed_traces) + len(failed_traces_ids))}/{len(df)} successful, {len(failed_traces) + len(failed_traces_ids)} failed"
        )
    else:
        logger.info(
            f"Completed evaluation: {len(df) - (len(failed_traces))}/{len(df)} successful, {len(failed_traces)} failed"
        )


if __name__ == "__main__":
    # handcrafted dataset
    # df_handcrafted = pd.read_parquet(
    #     "hf://datasets/Kevin355/Who_and_When/Hand-Crafted.parquet"
    # )
    # or llm-generated dataset
    df_algorithm = pd.read_parquet(
        "hf://datasets/Kevin355/Who_and_When/Algorithm-Generated.parquet"
    )

    asyncio.run(
        main(
            save_folder="who_and_when_single_agent_algo_gemini_2.5_flash_run_2(20.05.26)",
            df=df_algorithm,
            # num_traces=1,
        )
    )
