import json
from pathlib import Path

def evaluate_metrics(folder_path: str):
    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"Folder {folder_path} does not exist")

    files = list(folder.glob("*.json"))
    if not files:
        raise ValueError(f"No JSON files found in {folder_path}")

    total_traces = 0
    step_correct = 0
    step_module_correct = 0
    all_correct = 0
    lost_gt = 0

    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        scores = data.get("summarizer_score", {}).get("scores", [])[0].get("score", {})
        anno = data.get("summarizer_score", {}).get("scores", [])[0]
        
        gt_step = anno.get("critical_failure_step")
        gt_module = anno.get("critical_failure_module")
        gt_type = anno.get("step_annotations")[0].get(gt_module).get("failure_type")
        
        if gt_module == "" or gt_step == "":
            lost_gt += 1
            print(f"Skip sample, lost GT!, lost count: {lost_gt}")
            continue
    
        pred_step = scores.get("critical_failure_step")
        pred_module = scores.get("critical_failure_module")
        pred_type = scores.get("failure_type")

        total_traces += 1
        if str(pred_step) == str(gt_step):
            step_correct += 1
        if str(pred_step) == str(gt_step) and str(pred_module) == str(gt_module):
            step_module_correct += 1
            
        # if GT not exist in annotation 
        if gt_type != "":
            if (str(pred_step) == str(gt_step)
                and str(pred_module) == str(gt_module)
                and str(pred_type) == str(gt_type)):
                all_correct += 1
        else:
            if (str(pred_step) == str(gt_step)
                and str(pred_module) == str(gt_module)):
                all_correct += 1
            
    total_traces = total_traces - lost_gt

    print(f"Total traces: {total_traces}")
    print(f"Step Accuracy: {step_correct/total_traces:.2%}")
    print(f"Step+Module Accuracy: {step_module_correct/total_traces:.2%}")
    print(f"All Correct Accuracy: {all_correct/total_traces:.2%}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python eval_on_aeb.py <path_to_json_folder>")
        sys.exit(1)
    evaluate_metrics(sys.argv[1])