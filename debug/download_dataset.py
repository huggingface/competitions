from huggingface_hub import snapshot_download
import os

snapshot_download(
    repo_id=os.environ.get("SAFE_DATASET_REPO"),
    local_dir=os.environ.get("DATASET_PATH"),
    token=os.environ.get("HF_TOKEN"),
    repo_type="dataset",
)