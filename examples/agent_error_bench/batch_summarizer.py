import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import asyncio
import os
from dotenv import load_dotenv

load_dotenv(".env")

from automas.meta_agents import StepsBatchSummarizer
from automas.utils import get_logger
import json
import pandas as pd
from pydantic_ai.models.openai import OpenAIChatModel

logger = get_logger(__name__)


async def _summarize_chunk_with_retries(
    model, chunk, chunk_ids, task_id, prev_summary_text, logger, max_retries=3
):
    """Summarize a chunk with retries. Falls back to one-by-one on persistent count mismatch."""
    last_exc = None
    for attempt in range(max_retries):
        try:
            summarizer = StepsBatchSummarizer(model=model)
            batch_result = await summarizer.summarize(
                states=chunk,
                state_ids=chunk_ids,
                task_id=task_id,
                previous_summary=prev_summary_text,
            )
            if len(batch_result) != len(chunk):
                raise ValueError(
                    f"Expected {len(chunk)} summaries, got {len(batch_result)}"
                )
            return batch_result
        except Exception as e:
            last_exc = e
            if attempt < max_retries - 1:
                delay = min(2 ** (attempt + 1), 30)
                logger.warning(
                    f"Summarizer attempt {attempt + 1}/{max_retries} failed for "
                    f"{task_id} chunk {chunk_ids[0]}-{chunk_ids[-1]}: {e}. Retrying in {delay}s..."
                )
                await asyncio.sleep(delay)

    if len(chunk) == 1:
        raise last_exc

    logger.warning(
        f"Batch summarization failed after {max_retries} attempts for {task_id} "
        f"chunk {chunk_ids[0]}-{chunk_ids[-1]}. Falling back to one-by-one..."
    )
    results = []
    for state, sid in zip(chunk, chunk_ids):
        single_result = await _summarize_chunk_with_retries(
            model, [state], [sid], task_id, prev_summary_text, logger, max_retries
        )
        results.extend(single_result)
    return results


async def main(save_folder: str, df):
    logger.info(f"===Starting Who&When evaluation===")

    summarizer_model = OpenAIChatModel(
        "google/gemini-2.5-flash",
        provider="openrouter",
        settings={
            "temperature": 0.0,
            "reasoning": False,
        },
    )

    # continue processing that was already started
    done_traces = []
    local_results_dir = Path(__file__).resolve().parent / "results" / save_folder
    results_dir = local_results_dir

    if results_dir.exists():
        for res in os.listdir(results_dir):
            res_path = Path(results_dir) / res
            done_traces.append(res_path.stem) 

    # skip failed traces
    failed_traces_ids = []
    failed_traces = []

    if local_results_dir.exists():
        if "failed_traces.txt" in os.listdir(local_results_dir):
            with open(local_results_dir / "failed_traces.txt", "r") as f:
                for line in f:
                    if line.startswith("Task ID:"):
                        failed_traces_ids.append(line.split(":")[1].strip())

    for idx in range(len(df)):
        logger.info(
            f"Processing task {idx + 1}/{len(df)}: {str(df.iloc[idx]['filename'])}"
        )
        summary_file = (
            Path(__file__).resolve().parent
            / "summaries_agent_error_v2"
            / "ALFWorld"
            / f"{str(df.iloc[idx]['filename'])}.json"
        )
        if os.path.exists(summary_file):
            logger.info(
                f"Trace summary already exists for {str(df.iloc[idx]['filename'])}, skipping..."
            )
            summary = json.load(open(summary_file))
        else:
            summary = {"step_by_step_summary": []}
            all_summaries = []
            prev_summary_text = None
            chunk_size = 5

            for i in range(0, len(df.iloc[idx]["messages"]), chunk_size):
                chunk = df.iloc[idx]["messages"][i : i + chunk_size]
                chunk_ids = [
                    f"{str(df.iloc[idx]['filename'])}_{j}"
                    for j in range(i + 1, i + 1 + len(chunk))
                ]

                batch_result = await _summarize_chunk_with_retries(
                    summarizer_model,
                    chunk,
                    chunk_ids,
                    str(df.iloc[idx]['filename']),
                    prev_summary_text,
                    logger,
                )
                for idx_in_chunk, s in enumerate(batch_result):
                    s.id = chunk_ids[idx_in_chunk]
                    print(s)
                    all_summaries.append(s)
                    summary["step_by_step_summary"].append(s.model_dump(mode="json"))

                prev_summary_text = "\n".join(
                    f"[{s.id}] {s.name} ({s.role}): {s.content_summary}"
                    for s in all_summaries[-3:]
                )

            logger.info("Trace summary generated")
            os.makedirs(
                Path(__file__).resolve().parent
                / "summaries_agent_error_v2"
                / "ALFWorld",
                exist_ok=True,
            )
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Trace summary saved to {summary_file}")


if __name__ == "__main__":
    import os
    import json
    import pandas as pd
    import glob

    folder = "/home/alina/Desktop/AutoJudge/examples/agent_error_bench/AgentErrorBench/Original_Failure_Trajectory/ALFWorld"

    data = []

    for file in glob.glob(folder + "/*.json"):
        with open(file, "r", encoding="utf-8") as f:
            content = json.load(f)
            filename = os.path.basename(file).replace(".json", "")
            if not ('Llama' in filename):
                continue

            if isinstance(content, dict):
                data.append({"filename": filename, **content})
            elif isinstance(content, list):
                for item in content:
                    data.append({"filename": filename, **item})

    df = pd.json_normalize(data)

    print(df.head())

    asyncio.run(
        main(
            save_folder="summary",
            df=df[:],
        )
    )
