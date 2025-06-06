import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils import EntryNotFoundError
from loguru import logger

from competitions import utils
from competitions.compute_metrics import compute_metrics
from competitions.enums import SubmissionStatus
from competitions.params import EvalParams

#TODO figure how to avoid logging token file
# LOG_FILE="/app/logs/evaluate.log"

# os.makedirs("logs", exist_ok=True)

# Configure logger to write logs to a file with rotation and retention.
# logger.add(LOG_FILE, rotation="10 MB", retention="10 days", level="INFO")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def upload_submission_file(params, file_path):
    logger.info("Uploading submission file")
    pass


def generate_submission_file(params, conda_env=None):
    logger.info("Downloading submission mode repo")
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
    logger.info("Downloaded submission mode repo")

    logger.info("Running script.py")

    # invalidate USER_TOKEN env var
    os.environ["USER_TOKEN"] = ""

    # Copy sandbox to submission_dir
    shutil.copyfile("sandbox", f"{submission_dir}/sandbox")
    sandbox_path = f"{submission_dir}/sandbox"
    os.chmod(sandbox_path, 0o755)
    os.chown(sandbox_path, os.getuid(), os.getgid())

    # Define your command

    if conda_env:
        cmd = f"{sandbox_path} conda run -p {conda_env} --no-capture-output python script.py"
    else:
        cmd = f"{sandbox_path} python script.py"

    cmd = shlex.split(cmd)

    # Copy the current environment and modify it
    env = os.environ.copy()
    env["PARAMS"] = ""

    # Start the subprocess
    stdout_log_path = os.path.join(submission_dir, "stdout.log")
    stderr_log_path = os.path.join(submission_dir, "stderr.log")

    with open(stdout_log_path, "w") as stdout_log, open(
        stderr_log_path, "w"
    ) as stderr_log:
        process = subprocess.Popen(
            cmd,
            cwd=submission_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        def stream_copy(src, *dests):
            for line in src:
                for dest in dests:
                    dest.write(line)
                    dest.flush()

        threads = [
            threading.Thread(
                target=stream_copy, args=(process.stdout, sys.stdout, stdout_log)
            ),
            threading.Thread(
                target=stream_copy, args=(process.stderr, sys.stderr, stderr_log)
            ),
        ]
        for t in threads:
            t.start()
        # Wait for the process to complete or timeout
        try:
            process.wait(timeout=params.time_limit)
        except subprocess.TimeoutExpired:
            logger.info(
                f"Process exceeded {params.time_limit} seconds time limit. Terminating..."
            )
            process.kill()
            process.wait()
        for t in threads:
            t.join()

    LOG_FILE = "submission.log"

    # After process ends, log the captured stdout and stderr

    with open(LOG_FILE, "w") as log_file, open(stdout_log_path) as stdout_log, open(
        stderr_log_path
    ) as stderr_log:
        log_file.write(f"Submission STDOUT:\n{stdout_log.read()}\n")
        log_file.write(f"Submission STDERR:\n{stderr_log.read()}\n")

    utils.upload_submission_logs(params, LOG_FILE)


    # # Wait for the process to complete or timeout
    # try:
    #     process.wait(timeout=params.time_limit)
    # except subprocess.TimeoutExpired:
    #     logger.info(f"Process exceeded {params.time_limit} seconds time limit. Terminating...")
    #     process.kill()
    #     process.wait()

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
        file_path = f"{submission_dir}/{sub_file}"

        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File {file_path} does not exist.")
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=f"submissions/{params.team_id}-{params.submission_id}.{sub_file_ext}",
                repo_id=params.competition_id,
                repo_type="dataset",
            )
            logger.info(f"Successfully uploaded {file_path}")
        except Exception as e:
            raise e


@utils.monitor
def run(params):
    logger.info(params)
    if isinstance(params, dict):
        params = EvalParams(**params)

    utils.update_submission_status(params, SubmissionStatus.PROCESSING.value)

    if params.competition_type == "script":

        try:
            requirements_fname = hf_hub_download(
                repo_id=params.submission_repo,
                filename="requirements.txt",
                token=os.environ.get("USER_TOKEN"),
                repo_type="model",
            )
            logger.info(
                f"found custom requirments.txt file in {params.submission_repo}"
            )

        except EntryNotFoundError:
            logger.info(
                f"custom requirments.txt file not found in {params.submission_repo}"
            )
            try:
                requirements_fname = hf_hub_download(
                    repo_id=params.competition_id,
                    filename="requirements.txt",
                    token=params.token,
                    repo_type="dataset",
                )

                logger.info(f"using a default requirments file instead")

            except EntryNotFoundError:
                requirements_fname = None

        if requirements_fname:
            logger.info("Installing requirements")
            # utils.uninstall_requirements(requirements_fname)
            utils.install_requirements(
                requirements_fname,
                conda_env=os.environ.get("CONDA_ENV_MODEL", "/app/model_default"),
            )
        if len(str(params.dataset).strip()) > 0:
            logger.info("downloading dataset.")
            # _ = Repository(local_dir="/tmp/data", clone_from=params.dataset, token=params.token)
            _ = snapshot_download(
                repo_id=params.dataset,
                local_dir="/tmp/data",
                token=params.token,
                repo_type="dataset",
            )
            logger.info("downloaded dataset.")
        generate_submission_file(
            params, conda_env=os.environ.get("CONDA_ENV_MODEL", "/app/model_default")
        )

    evaluation = compute_metrics(params)

    utils.update_submission_score(
        params, evaluation["public_score"], evaluation["private_score"]
    )
    utils.update_submission_status(params, SubmissionStatus.SUCCESS.value)
    # utils.upload_submission_logs(params, log_file=LOG_FILE)
    utils.delete_space(params)


if __name__ == "__main__":
    args = parse_args()
    _params = json.load(open(args.config, encoding="utf-8"))
    _params = EvalParams(**_params)
    run(_params)
