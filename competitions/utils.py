import io
import json
import os
import shlex
import subprocess
import traceback

import requests
from fastapi import Request
from huggingface_hub import HfApi, hf_hub_download
from loguru import logger

from competitions.enums import SubmissionStatus
from competitions.params import EvalParams

from . import HF_URL


USER_TOKEN = os.environ.get("USER_TOKEN")


def token_information(token):
    if token.startswith("hf_oauth"):
        _api_url = HF_URL + "/oauth/userinfo"
    else:
        _api_url = HF_URL + "/api/whoami-v2"
    headers = {}
    cookies = {}
    if token.startswith("hf_"):
        headers["Authorization"] = f"Bearer {token}"
    else:
        cookies = {"token": token}
    try:
        response = requests.get(
            _api_url,
            headers=headers,
            cookies=cookies,
            timeout=3,
        )
    except (requests.Timeout, ConnectionError) as err:
        logger.error(f"Failed to request whoami-v2 - {repr(err)}")
        raise Exception("Hugging Face Hub is unreachable, please try again later.")

    if response.status_code != 200:
        logger.error(f"Failed to request whoami-v2 - {response.status_code}")
        raise Exception("Invalid token.")

    resp = response.json()
    user_info = {}

    if token.startswith("hf_oauth"):
        user_info["id"] = resp["sub"]
        user_info["name"] = resp["preferred_username"]
        user_info["orgs"] = [resp["orgs"][k]["preferred_username"] for k in range(len(resp["orgs"]))]
    else:
        user_info["id"] = resp["id"]
        user_info["name"] = resp["name"]
        user_info["orgs"] = [resp["orgs"][k]["name"] for k in range(len(resp["orgs"]))]
    return user_info


def user_authentication(request: Request):
    auth_header = request.headers.get("Authorization")
    bearer_token = None

    if auth_header and auth_header.startswith("Bearer "):
        bearer_token = auth_header.split(" ")[1]

    if bearer_token:
        try:
            _ = token_information(token=bearer_token)
            return bearer_token
        except Exception as e:
            logger.error(f"Failed to verify token: {e}")
            return None

    if USER_TOKEN is not None:
        try:
            _ = token_information(token=USER_TOKEN)
            return USER_TOKEN
        except Exception as e:
            logger.error(f"Failed to verify token: {e}")
            return None

    if "oauth_info" in request.session:
        try:
            _ = token_information(token=request.session["oauth_info"]["access_token"])
            return request.session["oauth_info"]["access_token"]
        except Exception as e:
            request.session.pop("oauth_info", None)
            logger.error(f"Failed to verify token: {e}")
            return None

    return None


def user_authentication_dep(token, return_raw=False):
    if token.startswith("hf_oauth"):
        _api_url = HF_URL + "/oauth/userinfo"
    else:
        _api_url = HF_URL + "/api/whoami-v2"
    headers = {}
    cookies = {}
    if token.startswith("hf_"):
        headers["Authorization"] = f"Bearer {token}"
    else:
        cookies = {"token": token}
    try:
        response = requests.get(
            _api_url,
            headers=headers,
            cookies=cookies,
            timeout=3,
        )
    except (requests.Timeout, ConnectionError) as err:
        logger.error(f"Failed to request whoami-v2 - {repr(err)}")
        raise Exception("Hugging Face Hub is unreachable, please try again later.")

    resp = response.json()
    if return_raw:
        return resp

    user_info = {}
    if "error" in resp:
        return resp
    if token.startswith("hf_oauth"):
        user_info["id"] = resp["sub"]
        user_info["name"] = resp["preferred_username"]
        user_info["orgs"] = [resp["orgs"][k]["preferred_username"] for k in range(len(resp["orgs"]))]
    else:

        user_info["id"] = resp["id"]
        user_info["name"] = resp["name"]
        user_info["orgs"] = [resp["orgs"][k]["name"] for k in range(len(resp["orgs"]))]
    return user_info


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
    cmd = shlex.split(" ".join(cmd))
    process = subprocess.Popen(cmd, env=env)
    if wait:
        process.wait()
    return process.pid


def pause_space(params):
    if "SPACE_ID" in os.environ:
        if os.environ["SPACE_ID"].split("/")[-1].startswith("comp-"):
            logger.info("Pausing space...")
            api = HfApi(token=params.token)
            api.pause_space(repo_id=os.environ["SPACE_ID"])


def delete_space(params):
    if "SPACE_ID" in os.environ:
        if os.environ["SPACE_ID"].split("/")[-1].startswith("comp-"):
            logger.info("Deleting space...")
            api = HfApi(token=params.token)
            api.delete_repo(repo_id=os.environ["SPACE_ID"], repo_type="space")


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
            submission["status"] = SubmissionStatus.SUCCESS.value
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
            update_submission_status(params, SubmissionStatus.FAILED.value)
            pause_space(params)

    return wrapper


def uninstall_requirements(requirements_fname):
    if os.path.exists(requirements_fname):
        # read the requirements.txt
        uninstall_list = []
        with open(requirements_fname, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("-"):
                    uninstall_list.append(line[1:])

        # create an uninstall.txt
        with open("uninstall.txt", "w", encoding="utf-8") as f:
            for line in uninstall_list:
                f.write(line)

        pipe = subprocess.Popen(
            [
                "pip",
                "uninstall",
                "-r",
                "uninstall.txt",
                "-y",
            ],
        )
        pipe.wait()
        logger.info("Requirements uninstalled.")
        return


def install_requirements(requirements_fname):
    # check if params.project_name has a requirements.txt
    if os.path.exists(requirements_fname):
        # install the requirements using subprocess, wait for it to finish
        install_list = []

        with open(requirements_fname, "r", encoding="utf-8") as f:
            for line in f:
                # if line startswith - then skip but dont skip if line startswith --
                if line.startswith("-"):
                    if not line.startswith("--"):
                        continue
                install_list.append(line)

        with open("install.txt", "w", encoding="utf-8") as f:
            for line in install_list:
                f.write(line)

        pipe = subprocess.Popen(
            [
                "pip",
                "install",
                "-r",
                "install.txt",
            ],
        )
        pipe.wait()
        logger.info("Requirements installed.")
        return
    logger.info("No requirements.txt found. Skipping requirements installation.")
    return


def is_user_admin(user_token, competition_organization):
    user_info = token_information(token=user_token)
    user_orgs = user_info.get("orgs", [])
    for org in user_orgs:
        if org == competition_organization:
            return True
    return False


def get_team_name(user_token, competition_id, hf_token):
    user_info = token_information(token=user_token)
    user_id = user_info["id"]
    user_team = hf_hub_download(
        repo_id=competition_id,
        filename="user_team.json",
        token=hf_token,
        repo_type="dataset",
    )
    with open(user_team, "r", encoding="utf-8") as f:
        user_team = json.load(f)

    if user_id not in user_team:
        return None

    team_id = user_team[user_id]

    team_metadata = hf_hub_download(
        repo_id=competition_id,
        filename="teams.json",
        token=hf_token,
        repo_type="dataset",
    )
    with open(team_metadata, "r", encoding="utf-8") as f:
        team_metadata = json.load(f)

    team_name = team_metadata[team_id]["name"]
    return team_name


def update_team_name(user_token, new_team_name, competition_id, hf_token):
    user_info = token_information(token=user_token)
    user_id = user_info["id"]
    user_team = hf_hub_download(
        repo_id=competition_id,
        filename="user_team.json",
        token=hf_token,
        repo_type="dataset",
    )
    with open(user_team, "r", encoding="utf-8") as f:
        user_team = json.load(f)

    if user_id not in user_team:
        raise Exception("User is not part of a team")

    team_id = user_team[user_id]

    team_metadata = hf_hub_download(
        repo_id=competition_id,
        filename="teams.json",
        token=hf_token,
        repo_type="dataset",
    )
    with open(team_metadata, "r", encoding="utf-8") as f:
        team_metadata = json.load(f)

    team_metadata[team_id]["name"] = new_team_name
    team_metadata_json = json.dumps(team_metadata, indent=4)
    team_metadata_json_bytes = team_metadata_json.encode("utf-8")
    team_metadata_json_buffer = io.BytesIO(team_metadata_json_bytes)
    api = HfApi(token=hf_token)
    api.upload_file(
        path_or_fileobj=team_metadata_json_buffer,
        path_in_repo="teams.json",
        repo_id=competition_id,
        repo_type="dataset",
    )
    return new_team_name
