from automas.judge_eval.eval_on_aeb import evaluate_metrics

if __name__ == "__main__":
    folder_path = "/home/alina/Desktop/AutoJudge/examples/agent_error_bench/results/gaia_full_trace"  
    evaluate_metrics(folder_path)