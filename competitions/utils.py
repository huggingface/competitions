import io
import json
import os
import subprocess
import traceback

import requests
from huggingface_hub import HfApi, hf_hub_download
from loguru import logger

from competitions.params import EvalParams

from . import MOONLANDING_URL


def user_authentication(token):
    headers = {}
    cookies = {}
    if token.startswith("hf_"):
        headers["Authorization"] = f"Bearer {token}"
    else:
        cookies = {"token": token}
    try:
        response = requests.get(
            MOONLANDING_URL + "/api/whoami-v2",
            headers=headers,
            cookies=cookies,
            timeout=3,
        )
    except (requests.Timeout, ConnectionError) as err:
        logger.error(f"Failed to request whoami-v2 - {repr(err)}")
        raise Exception("Hugging Face Hub is unreachable, please try again later.")
    return response.json()


def make_clickable_user(user_id):
    link = "https://huggingface.co/" + user_id
    return f'<a  target="_blank" href="{link}">{user_id}</a>'


def run_evaluation(params, local=False, wait=False):
    params = json.loads(params)
    if isinstance(params, str):
        params = json.loads(params)
    params = EvalParams(**params)
    if not local:
        params.output_path = "/tmp/model"
    params.save(output_dir=params.output_path)
    cmd = [
        "python",
        "-m",
        "competitions.evaluate",
        "--config",
        os.path.join(params.output_path, "params.json"),
    ]

    cmd = [str(c) for c in cmd]
    logger.info(cmd)
    env = os.environ.copy()
    process = subprocess.Popen(" ".join(cmd), shell=True, env=env)
    if wait:
        process.wait()
    return process.pid


def pause_space(params):
    if "SPACE_ID" in os.environ:
        if os.environ["SPACE_ID"].split("/")[-1].startswith("comp-"):
            logger.info("Pausing space...")
            api = HfApi(token=params.token)
            api.pause_space(repo_id=os.environ["SPACE_ID"])


def download_submission_info(params):
    user_fname = hf_hub_download(
        repo_id=params.competition_id,
        filename=f"submission_info/{params.team_id}.json",
        token=params.token,
        repo_type="dataset",
    )
    with open(user_fname, "r", encoding="utf-8") as f:
        user_submission_info = json.load(f)

    return user_submission_info


def upload_submission_info(params, user_submission_info):
    user_submission_info_json = json.dumps(user_submission_info, indent=4)
    user_submission_info_json_bytes = user_submission_info_json.encode("utf-8")
    user_submission_info_json_buffer = io.BytesIO(user_submission_info_json_bytes)
    api = HfApi(token=params.token)
    api.upload_file(
        path_or_fileobj=user_submission_info_json_buffer,
        path_in_repo=f"submission_info/{params.team_id}.json",
        repo_id=params.competition_id,
        repo_type="dataset",
    )


def update_submission_status(params, status):
    user_submission_info = download_submission_info(params)
    for submission in user_submission_info["submissions"]:
        if submission["submission_id"] == params.submission_id:
            submission["status"] = status
            break
    upload_submission_info(params, user_submission_info)


def update_submission_score(params, public_score, private_score):
    user_submission_info = download_submission_info(params)
    for submission in user_submission_info["submissions"]:
        if submission["submission_id"] == params.submission_id:
            submission["public_score"] = public_score
            submission["private_score"] = private_score
            submission["status"] = "done"
            break
    upload_submission_info(params, user_submission_info)


def monitor(func):
    def wrapper(*args, **kwargs):
        params = kwargs.get("params", None)
        if params is None and len(args) > 0:
            params = args[0]

        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_message = f"""{func.__name__} has failed due to an exception: {traceback.format_exc()}"""
            logger.error(error_message)
            logger.error(str(e))
            update_submission_status(params, "failed")
            pause_space(params)

    return wrapper
