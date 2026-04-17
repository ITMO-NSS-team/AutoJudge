from autojudge.judge_eval.eval_on_aeb import evaluate_metrics

if __name__ == "__main__":
    folder_path = "/home/alina/Desktop/AutoJudge/examples/agent_error_bench/results/alfworld_sm_db_16_04"  
    evaluate_metrics(folder_path)