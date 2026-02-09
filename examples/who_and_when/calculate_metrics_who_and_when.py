import json
import glob
import pandas as pd
from pathlib import Path

df = pd.read_parquet("hf://datasets/Kevin355/Who_and_When/Hand-Crafted.parquet")
df = df[:30]
# df = pd.read_parquet("hf://datasets/Kevin355/Who_and_When/Algorithm-Generated.parquet")

path = str(Path(__file__).resolve().parent / "results" / "who_and_when_res_30_traces" / "*.json")
files = glob.glob(path)

tp = tn = fp = fn = 0
correct_mistake_agent, correct_mistake_agent_judge = 0, 0
processed_files = 0

for file in files:
    question_id = file.split("/")[-1].replace(".json", "")
    
    if question_id == "failed_traces":
        continue
    
    row = df[df['question_ID'] == question_id]
    
    if row.empty:
        print(f"Warning: question_ID {question_id} not found in DataFrame, skipping...")
        continue
    
    row = row.iloc[0]
    found = False
    mistake_agent_true = row['mistake_agent']

    if 'is_corrected' in row.keys():
        gt = int(row["is_corrected"])  
    elif 'is_correct' in row.keys():
        gt = int(row["is_correct"])
    else:
        print(f"Warning: no is_correct/is_corrected column for {question_id}, skipping...")
        continue

    data = json.load(open(file))

    score = data['summarizer_score']['scores'][0]['score']
    
    pred = 0 if score == 'poor' or score == 'fair'  else 1
    
    print(f"File: {question_id}, GT: {gt}, Pred: {pred}, Score: {score}")
    if pred == 0 and score == 'poor':
        for llm_metric in data.keys():
            if found:
                break
            if data.get(llm_metric):
                if llm_metric not in ['gt', 'label_answer']:
                    for score in data[llm_metric]['scores']:
                        if score['score'] == 'poor' or score['score'] == 'fair':
                            if score['justification']:
                                if mistake_agent_true.lower().replace(" ", "") in score['justification'].lower().replace(" ", ""):
                                    correct_mistake_agent += 1
                                    found = True
                                    break
        judge_justification = data['summarizer_score']['scores'][0]['justification']
        if mistake_agent_true.lower().replace(" ", "") in judge_justification.lower().replace(" ", ""):
            correct_mistake_agent_judge += 1
            
    if gt == 1 and pred == 1:
        tp += 1
    elif gt == 0 and pred == 0:
        tn += 1
    elif gt == 0 and pred == 1:
        fp += 1
    elif gt == 1 and pred == 0:
        fn += 1
    
    processed_files += 1

print(f"\nProcessed files: {processed_files}")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")

total = tp + tn + fp + fn
if total > 0:
    accuracy = (tp + tn) / total
    print(f"Accuracy: {accuracy:.4f}")
else:
    print("Accuracy: N/A (no valid predictions)")

who_accuracy = correct_mistake_agent / processed_files if processed_files > 0 else 0
who_judge_accuracy = correct_mistake_agent_judge / processed_files if processed_files > 0 else 0
print(f"Who llm-metrics Accuracy: {who_accuracy:.4f}")
print(f"Who Judge Accuracy: {who_judge_accuracy:.4f}")