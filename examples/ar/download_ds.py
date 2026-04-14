from huggingface_hub import snapshot_download

# Download the dataset to ./trajectories/
snapshot_download(
    repo_id="McGill-NLP/agent-reward-bench",
    repo_type="dataset",
    local_dir="./trajectories/"
)
