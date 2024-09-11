import argparse
import json
import os
import shlex
import shutil
import subprocess

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils._errors import EntryNotFoundError
from loguru import logger

from competitions import utils
from competitions.compute_metrics import compute_metrics
from competitions.enums import SubmissionStatus
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

    # invalidate USER_TOKEN env var
    os.environ["USER_TOKEN"] = ""

    # Copy sandbox to submission_dir
    shutil.copyfile("sandbox", f"{submission_dir}/sandbox")
    sandbox_path = f"{submission_dir}/sandbox"
    os.chmod(sandbox_path, 0o755)

    # Define your command
    cmd = "./sandbox python script.py"
    cmd = shlex.split(cmd)

    # Copy the current environment and modify it
    env = os.environ.copy()

    # Start the subprocess
    process = subprocess.Popen(cmd, cwd=submission_dir, env=env)

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
    else:
        logger.info("Subprocess terminated successfully")

    logger.info("contents of submission_dir")
    logger.info(os.listdir(submission_dir))

    api = HfApi(token=params.token)
    for sub_file in params.submission_filenames:
        logger.info(f"Uploading {sub_file} to the repository")
        sub_file_ext = sub_file.split(".")[-1]
        api.upload_file(
            path_or_fileobj=f"{submission_dir}/{sub_file}",
            path_in_repo=f"submissions/{params.team_id}-{params.submission_id}.{sub_file_ext}",
            repo_id=params.competition_id,
            repo_type="dataset",
        )


@utils.monitor
def run(params):
    logger.info(params)
    if isinstance(params, dict):
        params = EvalParams(**params)

    utils.update_submission_status(params, SubmissionStatus.PROCESSING.value)

    if params.competition_type == "script":
        try:
            requirements_fname = hf_hub_download(
                repo_id=params.competition_id,
                filename="requirements.txt",
                token=params.token,
                repo_type="dataset",
            )
        except EntryNotFoundError:
            requirements_fname = None

        if requirements_fname:
            logger.info("Installing requirements")
            utils.uninstall_requirements(requirements_fname)
            utils.install_requirements(requirements_fname)
        if len(str(params.dataset).strip()) > 0:
            # _ = Repository(local_dir="/tmp/data", clone_from=params.dataset, token=params.token)
            _ = snapshot_download(
                repo_id=params.dataset,
                local_dir="/tmp/data",
                token=params.token,
                repo_type="dataset",
            )
        generate_submission_file(params)

    evaluation = compute_metrics(params)

    utils.update_submission_score(params, evaluation["public_score"], evaluation["private_score"])
    utils.update_submission_status(params, SubmissionStatus.SUCCESS.value)
    utils.delete_space(params)


if __name__ == "__main__":
    args = parse_args()
    _params = json.load(open(args.config, encoding="utf-8"))
    _params = EvalParams(**_params)
    run(_params)
