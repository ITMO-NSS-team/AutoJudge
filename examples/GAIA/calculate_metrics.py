"""
This file can be used to calculate the F1 score for each LLM-metric and judge. 
The script supports input in JSON log-format. Example is 
"""
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from ftfy import fix_text

load_dotenv(".env")

errors = []
SCORE_MAPPING = {"ideal": 1, "fair": 0, "poor": 0}


def normalize_for_comparison(text):
    if not isinstance(text, str):
        text = str(text)

    text = text.strip()

    replacements = {
        "\u2019": "'",  # ' → '
        "\u2018": "'",  # ' → '
        "\u201c": '"',  # " → "
        "\u201d": '"',  # " → "
        "\u2013": "-",  # – → -
        "\u2014": "-",  # — → -
        "\u00a0": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    import re

    text = re.sub(r"\s+", " ", text)

    return text


def calculate_f1_score(TP: int, TN: int, FP: int, FN: int):
    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)

    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)

    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score


def parse_gaia_files_to_dataframe(
    directory_path, name_of_metric_high_level: str, name_of_metric_low_level: str
):
    """
    Parse JSON files and create a DataFrame with specified columns
    """
    results = []

    directory = Path(directory_path)
    json_files = list(directory.glob("*.json"))

    print(f"Found {len(json_files)} JSON files to process")
    missing_score_count = 0

    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = json.load(file)
                content["mas_answer"] = fix_text(content.get("mas_answer", ""))

            filename = file_path.stem
            id = filename[-32:]

            low_level_score = []
            if (
                name_of_metric_low_level in content
                and "scores" in content[name_of_metric_low_level]
            ):
                for score_item in content[name_of_metric_low_level]["scores"]:
                    low_level_score.append(score_item.get("score", ""))

            high_level_score = ""
            if (
                name_of_metric_high_level in content
                and "scores" in content[name_of_metric_high_level]
            ):
                mas_scores = content[name_of_metric_high_level]["scores"]
                if mas_scores and len(mas_scores) > 0:
                    high_level_score = mas_scores[0].get("score", "")
                    if high_level_score == "":
                        missing_score_count += 1
                else:
                    missing_score_count += 1
            else:
                missing_score_count += 1

            # skip files with missing score
            if high_level_score == "":
                errors.append(filename)
                continue

            row_data = {
                "id": id,
                name_of_metric_low_level: low_level_score,
                name_of_metric_high_level: high_level_score,
                "gt": content.get("gt", ""),
                "mas_answer": content.get("mas_answer", ""),
                "label_answer": content.get("label_answer", ""),
            }

            results.append(row_data)

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue

    df = pd.DataFrame(results)

    print(f"Files with missing scores: {errors}")
    print(f"Total files with missing scores: {missing_score_count}")
    return df


def calculate_high_level_metric(df, enc="ISO-8859-1"):
    """
    Calculate evaluation metrics and create confusion matrix
    """
    # SHOULD CONSIST OF NEXT COLUMNS (!): name	gt	mas_answer	label_answer	human_score
    gt_store = pd.read_csv(os.environ["HUMAN_GT_PATH"], encoding=enc).values.tolist()
    evaluation_data = []
    evaluation_columns = [
        "id",
        "gt_answer",
        "sys_answer",
        "judge_answer",
        "answer_in_bin",
        "gt",
        "accuracy_per_task",
    ]

    correct_predictions = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0

    print("Evaluating high level score...")

    for idx, row in enumerate(df.values.tolist()):
        ground_truth_for_judge = None

        task_id = row[0]
        model_answer = row[4]
        correct_answer = row[5]

        # judge score
        raw_score = row[2]
        judge_score = SCORE_MAPPING[raw_score]

        for i in gt_store:
            if task_id == str(i[0]):
                ground_truth_for_judge = i[-1]

        if ground_truth_for_judge == None:
            print("Skip one row! Continue process data...")
            continue

        correct_predictions += judge_score == ground_truth_for_judge

        if judge_score is None:
            continue

        if judge_score == ground_truth_for_judge and ground_truth_for_judge == 1:
            true_positives += 1
        elif judge_score == 1 and ground_truth_for_judge == 0:
            false_positives += 1
        elif judge_score == 0 and ground_truth_for_judge == 1:
            false_negatives += 1
        else:
            true_negatives += 1

        evaluation_data.append(
            [
                task_id,
                correct_answer,
                model_answer,
                raw_score,
                judge_score,
                ground_truth_for_judge,
                judge_score == ground_truth_for_judge,
            ]
        )

    return (
        evaluation_data,
        evaluation_columns,
        correct_predictions,
        true_positives,
        false_positives,
        false_negatives,
        true_negatives,
    )


def calculate_low_level_metric(df, enc="ISO-8859-1"):
    """
    Calculate evaluation metrics
    """
    # SHOULD CONSIST OF NEXT COLUMNS (!): name	gt	mas_answer	label_answer	human_score
    gt_store = pd.read_csv(os.environ["HUMAN_GT_PATH"], encoding=enc).values.tolist()
    evaluation_data = []
    evaluation_columns = [
        "id",
        "gt_answer",
        "sys_answer",
        "task_scores",
        "avg_task_score",
        "rounded_score",
        "gt",
        "accuracy_per_task",
    ]
    cnt = 0
    correct_predictions = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0

    print("Evaluating low level scores...")

    for idx, row in enumerate(df.values.tolist()):
        cnt += 1
        task_id = row[0]
        task_scores = row[1]
        model_answer = row[4]
        correct_answer = row[5]

        # convert task scores to num values and calculate average
        numerical_scores = [
            SCORE_MAPPING[score] for score in task_scores if score in SCORE_MAPPING
        ]

        if numerical_scores:
            avg_score = sum(numerical_scores) / len(numerical_scores)
            # round to 0 or 1
            rounded_score = 1 if avg_score >= 0.5 else 0
        else:
            avg_score = 0
            rounded_score = 0

        ground_truth_for_judge = None

        for i in gt_store:
            if task_id == str(i[0]):
                ground_truth_for_judge = i[-1]

        if ground_truth_for_judge == None:
            print("Skip one row! Continue process data...")
            continue

        correct_predictions += rounded_score == ground_truth_for_judge

        if rounded_score == ground_truth_for_judge and ground_truth_for_judge == 1:
            true_positives += 1
        elif rounded_score == 1 and ground_truth_for_judge == 0:
            false_positives += 1
        elif rounded_score == 0 and ground_truth_for_judge == 1:
            false_negatives += 1
        else:
            true_negatives += 1

        evaluation_data.append(
            [
                task_id,
                correct_answer,
                model_answer,
                task_scores,
                avg_score,
                rounded_score,
                ground_truth_for_judge,
                rounded_score == ground_truth_for_judge,
            ]
        )

    return (
        evaluation_data,
        evaluation_columns,
        correct_predictions,
        true_positives,
        false_positives,
        false_negatives,
        true_negatives,
    )


if __name__ == "__main__":
    RESULT_FILE_NAME = "metrics"

    agent_metrics = [
        "task_completeness",
        "tool_parameter_extraction",
        "tool_selection",
        "state_consistency",
        "observation_alignment",
    ]
    system_metrics = [
        "summarizer_score",
        "mas_task_completion",
        "mas_roles_distribution",
        "mas_task_transfer",
        "mas_complexity",
    ]

    directory_path = "/home/user/Desktop/AutoMAS/AutoJudge/examples/GAIA/results/gaia_res_30_traces_with_summarizer_as_judge_input"

    for name_of_metric_high_level, name_of_metric_low_level in zip(
        system_metrics, agent_metrics
    ):
        print("Parsing files...")
        df = parse_gaia_files_to_dataframe(
            directory_path, name_of_metric_high_level, name_of_metric_low_level
        )
        output_path = "gaia_scores.csv"

        df.to_csv(output_path, index=False)

        print(f"\nDataFrame info:")
        print(df.info())

        print("\n" + "=" * 50)
        print("METRICS FOR ", name_of_metric_low_level)
        print("=" * 50)

        eval_data, eval_columns, correct, tp, fp, fn, tn = calculate_low_level_metric(df)

        print("Calculating correlation statistics...")
        pd.DataFrame(eval_data, columns=eval_columns).to_csv(
            f"{RESULT_FILE_NAME}_low_level_few_shot.csv"
        )

        total = tp + tn + fn + fp
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        print(f"F1: {calculate_f1_score(tp, tn, fp, fn)}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"TP: {tp}")
        print(f"FP: {fp}")
        print(f"FN: {fn}")
        print(f"TN: {tn}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")

        # Calculate metrics for task_completeness
        print("\n" + "=" * 50)
        print("METRICS FOR ", name_of_metric_high_level)
        print("=" * 50)

        (
            eval_data_task,
            eval_columns_task,
            correct_task,
            tp_task,
            fp_task,
            fn_task,
            tn_task,
        ) = calculate_high_level_metric(df)

        print("Calculating statistics...")
        pd.DataFrame(eval_data_task, columns=eval_columns_task).to_csv(
            f"{RESULT_FILE_NAME}_high_level_few_shot.csv"
        )

        total_task = tp_task + tn_task + fn_task + fp_task
        accuracy_task = (tp_task + tn_task) / total_task if total_task > 0 else 0
        precision_task = tp_task / (tp_task + fp_task) if (tp_task + fp_task) > 0 else 0
        recall_task = tp_task / (tp_task + fn_task) if (tp_task + fn_task) > 0 else 0

        print(f"F1: {calculate_f1_score(tp_task, tn_task, fp_task, fn_task)}")
        print(f"Accuracy: {accuracy_task:.4f}")
        print(f"TP: {tp_task}")
        print(f"FP: {fp_task}")
        print(f"FN: {fn_task}")
        print(f"TN: {tn_task}")
        print(f"Precision: {precision_task:.4f}")
        print(f"Recall: {recall_task:.4f}")