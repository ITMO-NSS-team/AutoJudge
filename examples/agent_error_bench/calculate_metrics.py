from automas.judge_eval.eval_on_aeb import evaluate_metrics

if __name__ == "__main__":
    folder_path = "/home/alina/Desktop/AutoJudge/examples/agent_error_bench/results/full_trace_webshop"  
    evaluate_metrics(folder_path)