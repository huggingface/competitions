import argparse
import json

from huggingface_hub import snapshot_download
from loguru import logger

from competitions import utils
from competitions.compute_metrics import compute_metrics
from competitions.params import EvalParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def generate_submission_file(params):
    logger.info("Downloading submission dataset")
    snapshot_download(
        repo_id=params.data_path,
        local_dir=params.output_path,
        token=params.token,
        repo_type="dataset",
    )


@utils.monitor
def run(params):
    if isinstance(params, dict):
        params = EvalParams(**params)

    utils.update_submission_status(params, "processing")

    if params.competition_type == "code":
        generate_submission_file(params)

    public_score, private_score = compute_metrics(params)

    utils.update_submission_score(params, public_score, private_score)
    utils.update_submission_status(params, "success")
    utils.pause_space(params)


if __name__ == "__main__":
    args = parse_args()
    _params = json.load(open(args.config, encoding="utf-8"))
    _params = EvalParams(**_params)
    run(_params)
