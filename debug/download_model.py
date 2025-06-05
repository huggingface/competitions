from huggingface_hub import snapshot_download
import os

snapshot_download(
    repo_id=os.environ.get("MODEL_REPO"),
    local_dir=os.environ.get("MODEL_PATH"),
    token=os.environ.get("HF_TOKEN"),
    repo_type="model",
)