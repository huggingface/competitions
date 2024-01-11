import argparse
import json
import os
import shutil
import subprocess

from huggingface_hub import HfApi, snapshot_download
from loguru import logger

from competitions import utils
from competitions.compute_metrics import compute_metrics
from competitions.params import EvalParams


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def upload_submission_file(params, file_path):
    logger.info("Uploading submission file")
    pass


def generate_submission_file(params):
    logger.info("Downloading submission dataset")
    submission_dir = snapshot_download(
        repo_id=params.submission_repo,
        local_dir=params.output_path,
        token=os.environ.get("USER_TOKEN"),
        repo_type="model",
    )
    # submission_dir has a script.py file
    # start a subprocess to run the script.py
    # the script.py will generate a submission.csv file in the submission_dir
    # push the submission.csv file to the repo using upload_submission_file
    logger.info("Generating submission file")

    # Copy socket-kit.so to submission_dir
    shutil.copyfile("socket-kit.so", f"{submission_dir}/socket-kit.so")

    # Define your command
    cmd = "python script.py"
    socket_kit_path = os.path.abspath(f"{submission_dir}/socket-kit.so")

    # Copy the current environment and modify it
    env = os.environ.copy()
    env["LD_PRELOAD"] = socket_kit_path

    # Start the subprocess
    process = subprocess.Popen(cmd, cwd=submission_dir, shell=True, env=env)

    # Wait for the process to complete or timeout
    try:
        process.wait(timeout=params.time_limit)
    except subprocess.TimeoutExpired:
        logger.info(f"Process exceeded {params.time_limit} seconds time limit. Terminating...")
        process.kill()
        process.wait()

    # Check if process terminated due to timeout
    if process.returncode and process.returncode != 0:
        logger.error("Subprocess didn't terminate successfully")

    api = HfApi(token=params.token)
    api.upload_file(
        path_or_fileobj=f"{submission_dir}/submission.csv",
        path_in_repo=f"submissions/{params.team_id}-{params.submission_id}.csv",
        repo_id=params.competition_id,
        repo_type="dataset",
    )


@utils.monitor
def run(params):
    logger.info(params)
    if isinstance(params, dict):
        params = EvalParams(**params)

    utils.update_submission_status(params, "processing")

    if params.competition_type == "code":
        generate_submission_file(params)

    evaluation = compute_metrics(params)

    utils.update_submission_score(params, evaluation["public_score"], evaluation["private_score"])
    utils.update_submission_status(params, "success")
    utils.pause_space(params)


if __name__ == "__main__":
    args = parse_args()
    _params = json.load(open(args.config, encoding="utf-8"))
    _params = EvalParams(**_params)
    run(_params)
