import json
from pathlib import Path

from dotenv import load_dotenv
import pandas as pd

from pydantic_ai import ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge

# DATASET = "handcrafted"   
DATASET = "algorithm" 

SAVE_FOLDER = None       

MODEL_NAME = "google/gemini-2.5-flash"
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

def history_to_str(history) -> str:
    if not isinstance(history, list):
        history = history.tolist()

    lines = []
    for i, msg in enumerate(history, 1):
        name = msg.get("name")
        role = msg.get("role", "unknown")
        speaker = name if name else role
        content = msg.get("content", "")

        if isinstance(content, str) and len(content) > 3000:
            content = content[:3000] + "..."

        lines.append(f"[{i}] {speaker}: {content}")

    return "\n\n".join(lines)


urls = {
    "handcrafted": "hf://datasets/Kevin355/Who_and_When/Hand-Crafted.parquet",
    "algorithm": "hf://datasets/Kevin355/Who_and_When/Algorithm-Generated.parquet",
}

df = pd.read_parquet(urls[DATASET])

model = OpenAIChatModel(
    MODEL_NAME,
    provider=OpenRouterProvider(),
    settings=ModelSettings(temperature=0.0),
)

RUBRIC = """
You are a forensic evaluator.

You must extract the root-cause failure.

You MUST follow these rules:

REASON FORMAT (STRICT - MUST FOLLOW EXACTLY):

AGENT: <agent_name>
STEP: <step_number>

RULES:
- REASON ONLY these 2 lines
- No sentences
- No explanations
- No punctuation
- No additional text


IMPORTANT DISTINCTION:
- Do NOT select where the error is noticed
- Do NOT select where error propagates
- ONLY select the FIRST step that introduces incorrect information or wrong action

IMPORTANT:
You are NOT allowed to describe the error.
You are NOT allowed to explain reasoning.
You are ONLY allowed to extract fields.

---

GOOD EXAMPLE:

AGENT: filesurfer
STEP: 44

BAD EXAMPLE (forbidden):
"The FileSurfer agent failed..."

Now extract from the trace.
"""


judge = LLMJudge(
    model=model,
    rubric=RUBRIC,
    include_expected_output=False,
    assertion=False,
    score={"evaluation_name": "ww_analysis", "include_reason": True},
    model_settings=ModelSettings(temperature=0.0),
)

cases = []
meta_list = []

for _, row in df.iterrows():
    task_id = row["question_ID"]

    cases.append(
        Case(
            inputs={
                "query": str(row["question"]),
                "trace": history_to_str(row["history"]),
            },
            name=task_id,
        )
    )

    meta_list.append({
        "task_id": task_id,
        "gt_agent": str(row["mistake_agent"]).strip().lower(),
        "gt_step": int(row["mistake_step"]),
    })


dataset = Dataset(
    name="who_when_forensic",
    cases=cases,
    evaluators=[judge],
)
report = dataset.evaluate_sync(
    lambda x: x,
    name="forensic_judge",
    max_concurrency=3,
    progress=True,
)

def parse_reason(reason: str):
    agent = None
    step = None

    for line in reason.splitlines():
        line = line.strip()

        if line.startswith("AGENT:"):
            agent = line.split(":", 1)[1].strip().lower()

        elif line.startswith("STEP:"):
            try:
                step = int(line.split(":", 1)[1].strip())
            except:
                step = None

    return agent, step


correct_agent = 0
correct_step = 0
total = 0

results_dir = None
if SAVE_FOLDER:
    results_dir = Path(__file__).resolve().parent / "results" / SAVE_FOLDER
    results_dir.mkdir(parents=True, exist_ok=True)


for cr, meta in zip(report.cases, meta_list):
    total += 1

    gt_agent = meta["gt_agent"]
    gt_step = meta["gt_step"]

    score = cr.scores.get("ww_analysis")
    reason = score.reason if score else ""
    print("RAW REASON:", reason, flush=True)

    pred_agent, pred_step = parse_reason(reason)

    agent_match = (pred_agent == gt_agent) if pred_agent else False
    step_match = (pred_step == gt_step) if pred_step else False

    if agent_match:
        correct_agent += 1
    if step_match:
        correct_step += 1

    print(
        f"{cr.name}: "
        f"agent={'✓' if agent_match else '✗'} "
        f"step={'✓' if step_match else '✗'} "
        f"pred={pred_agent}:{pred_step} "
        f"gt={gt_agent}:{gt_step}",
        flush=True
    )

    if results_dir:
        with open(results_dir / f"{cr.name}.json", "w") as f:
            json.dump({
                "task_id": cr.name,
                "pred_agent": pred_agent,
                "pred_step": pred_step,
                "gt_agent": gt_agent,
                "gt_step": gt_step,
                "agent_match": agent_match,
                "step_match": step_match,
                "reason": reason,
            }, f, indent=2)


print(f"\nResults: {DATASET}")
print(f"Agent accuracy: {correct_agent}/{total} = {correct_agent/total*100:.1f}%")
print(f"Step accuracy:  {correct_step}/{total} = {correct_step/total*100:.1f}%")

if report.failures:
    print(f"\nFailures: {len(report.failures)}")