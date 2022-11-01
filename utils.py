import datetime
import glob
import io
import json
import os
import time

import pandas as pd
import requests
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.utils._errors import EntryNotFoundError

import config


def get_auth_headers(token: str, prefix: str = "Bearer"):
    return {"Authorization": f"{prefix} {token}"}


def http_post(path: str, token: str, payload=None, domain: str = None, params=None) -> requests.Response:
    """HTTP POST request to the AutoNLP API, raises UnreachableAPIError if the API cannot be reached"""
    try:
        response = requests.post(
            url=domain + path, json=payload, headers=get_auth_headers(token=token), allow_redirects=True, params=params
        )
    except requests.exceptions.ConnectionError:
        print("❌ Failed to reach AutoNLP API, check your internet connection")
    response.raise_for_status()
    return response


def http_get(path: str, token: str, domain: str = None) -> requests.Response:
    """HTTP POST request to the AutoNLP API, raises UnreachableAPIError if the API cannot be reached"""
    try:
        response = requests.get(url=domain + path, headers=get_auth_headers(token=token), allow_redirects=True)
    except requests.exceptions.ConnectionError:
        print("❌ Failed to reach AutoNLP API, check your internet connection")
    response.raise_for_status()
    return response


def create_project(project_id, submission_dataset, model, dataset):
    project_config = {}
    project_config["dataset_name"] = "lewtun/imdb-dummy"
    project_config["dataset_config"] = "lewtun--imdb-dummy"
    project_config["dataset_split"] = "train"
    project_config["col_mapping"] = {"text": "text", "label": "target"}

    payload = {
        "username": config.AUTOTRAIN_USERNAME,
        "proj_name": project_id,
        "task": 1,
        "config": {
            "language": "en",
            "max_models": 5,
            "benchmark": {
                "dataset": dataset,
                "model": model,
                "submission_dataset": submission_dataset,
            },
        },
    }

    project_json_resp = http_post(
        path="/projects/create", payload=payload, token=config.AUTOTRAIN_TOKEN, domain=config.AUTOTRAIN_BACKEND_API
    ).json()
    time.sleep(5)
    # Upload data
    payload = {
        "split": 4,
        "col_mapping": project_config["col_mapping"],
        "load_config": {"max_size_bytes": 0, "shuffle": False},
        "dataset_id": project_config["dataset_name"],
        "dataset_config": project_config["dataset_config"],
        "dataset_split": project_config["dataset_split"],
    }

    _ = http_post(
        path=f"/projects/{project_json_resp['id']}/data/dataset",
        payload=payload,
        token=config.AUTOTRAIN_TOKEN,
        domain=config.AUTOTRAIN_BACKEND_API,
    ).json()
    print("💾💾💾 Dataset creation done 💾💾💾")

    # Process data
    _ = http_post(
        path=f"/projects/{project_json_resp['id']}/data/start_processing",
        token=config.AUTOTRAIN_TOKEN,
        domain=config.AUTOTRAIN_BACKEND_API,
    ).json()

    print("⏳ Waiting for data processing to complete ...")
    is_data_processing_success = False
    while is_data_processing_success is not True:
        project_status = http_get(
            path=f"/projects/{project_json_resp['id']}",
            token=config.AUTOTRAIN_TOKEN,
            domain=config.AUTOTRAIN_BACKEND_API,
        ).json()
        # See database.database.enums.ProjectStatus for definitions of `status`
        if project_status["status"] == 3:
            is_data_processing_success = True
            print("✅ Data processing complete!")
        time.sleep(10)

    # Approve training job
    _ = http_post(
        path=f"/projects/{project_json_resp['id']}/start_training",
        token=config.AUTOTRAIN_TOKEN,
        domain=config.AUTOTRAIN_BACKEND_API,
    ).json()


def user_authentication(token):
    headers = {}
    cookies = {}
    if token.startswith("hf_"):
        headers["Authorization"] = f"Bearer {token}"
    else:
        cookies = {"token": token}
    try:
        response = requests.get(
            config.MOONLANDING_URL + "/api/whoami-v2",
            headers=headers,
            cookies=cookies,
            timeout=3,
        )
    except (requests.Timeout, ConnectionError) as err:
        print(f"Failed to request whoami-v2 - {repr(err)}")
        raise Exception("Hugging Face Hub is unreachable, please try again later.")
    return response.json()


def add_new_user(user_info):
    api = HfApi()
    user_submission_info = {}
    user_submission_info["name"] = user_info["name"]
    user_submission_info["id"] = user_info["id"]
    user_submission_info["submissions"] = []
    # convert user_submission_info to BufferedIOBase file object
    user_submission_info_json = json.dumps(user_submission_info)
    user_submission_info_json_bytes = user_submission_info_json.encode("utf-8")
    user_submission_info_json_buffer = io.BytesIO(user_submission_info_json_bytes)

    api.upload_file(
        path_or_fileobj=user_submission_info_json_buffer,
        path_in_repo=f"{user_info['id']}.json",
        repo_id=config.COMPETITION_ID,
        repo_type="dataset",
        token=config.AUTOTRAIN_TOKEN,
    )


def check_user_submission_limit(user_info):
    user_id = user_info["id"]
    try:
        user_fname = hf_hub_download(
            repo_id=config.COMPETITION_ID,
            filename=f"{user_id}.json",
            use_auth_token=config.AUTOTRAIN_TOKEN,
            repo_type="dataset",
        )
    except EntryNotFoundError:
        add_new_user(user_info)
        user_fname = hf_hub_download(
            repo_id=config.COMPETITION_ID,
            filename=f"{user_id}.json",
            use_auth_token=config.AUTOTRAIN_TOKEN,
            repo_type="dataset",
        )
    except Exception as e:
        print(e)
        raise Exception("Hugging Face Hub is unreachable, please try again later.")

    with open(user_fname, "r") as f:
        user_submission_info = json.load(f)

    todays_date = datetime.datetime.now().strftime("%Y-%m-%d")
    if len(user_submission_info["submissions"]) == 0:
        user_submission_info["submissions"] = []

    # count the number of times user has submitted today
    todays_submissions = 0
    for sub in user_submission_info["submissions"]:
        if sub["date"] == todays_date:
            todays_submissions += 1
    if todays_submissions >= config.SUBMISSION_LIMIT:
        return False
    return True


def increment_submissions(user_id, submission_id, submission_comment):
    user_fname = hf_hub_download(
        repo_id=config.COMPETITION_ID,
        filename=f"{user_id}.json",
        use_auth_token=config.AUTOTRAIN_TOKEN,
        repo_type="dataset",
    )
    with open(user_fname, "r") as f:
        user_submission_info = json.load(f)
    todays_date = datetime.datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    # here goes all the default stuff for submission
    user_submission_info["submissions"].append(
        {
            "date": todays_date,
            "time": current_time,
            "submission_id": submission_id,
            "submission_comment": submission_comment,
            "status": "pending",
            "selected": False,
            "public_score": -1,
            "private_score": -1,
        }
    )
    # count the number of times user has submitted today
    todays_submissions = 0
    for sub in user_submission_info["submissions"]:
        if sub["date"] == todays_date:
            todays_submissions += 1
    # convert user_submission_info to BufferedIOBase file object
    user_submission_info_json = json.dumps(user_submission_info)
    user_submission_info_json_bytes = user_submission_info_json.encode("utf-8")
    user_submission_info_json_buffer = io.BytesIO(user_submission_info_json_bytes)
    api = HfApi()
    api.upload_file(
        path_or_fileobj=user_submission_info_json_buffer,
        path_in_repo=f"{user_id}.json",
        repo_id=config.COMPETITION_ID,
        repo_type="dataset",
        token=config.AUTOTRAIN_TOKEN,
    )
    return todays_submissions


def verify_submission(bytes_data):
    return True


def fetch_submissions(user_id):
    user_fname = hf_hub_download(
        repo_id=config.COMPETITION_ID,
        filename=f"{user_id}.json",
        use_auth_token=config.AUTOTRAIN_TOKEN,
        repo_type="dataset",
    )
    with open(user_fname, "r") as f:
        user_submission_info = json.load(f)
    return user_submission_info["submissions"]


def fetch_leaderboard(private=False):
    submissions_folder = snapshot_download(
        repo_id=config.COMPETITION_ID,
        allow_patterns="*.json",
        use_auth_token=config.AUTOTRAIN_TOKEN,
        repo_type="dataset",
    )
    submissions = []
    for submission in glob.glob(os.path.join(submissions_folder, "*.json")):
        with open(submission, "r") as f:
            submission_info = json.load(f)
        if config.EVAL_HIGHER_IS_BETTER:
            submission_info["submissions"].sort(
                key=lambda x: x["private_score"] if private else x["public_score"], reverse=True
            )
        else:
            submission_info["submissions"].sort(key=lambda x: x["private_score"] if private else x["public_score"])
        # select only the best submission
        submission_info["submissions"] = submission_info["submissions"][0]
        temp_info = {
            "id": submission_info["id"],
            "name": submission_info["name"],
            "submission_id": submission_info["submissions"]["submission_id"],
            "submission_comment": submission_info["submissions"]["submission_comment"],
            "status": submission_info["submissions"]["status"],
            "selected": submission_info["submissions"]["selected"],
            "public_score": submission_info["submissions"]["public_score"],
            "private_score": submission_info["submissions"]["private_score"],
            "submission_date": submission_info["submissions"]["date"],
            "submission_time": submission_info["submissions"]["time"],
        }
        submissions.append(temp_info)

    df = pd.DataFrame(submissions)
    # convert submission date and time to datetime
    df["submission_datetime"] = pd.to_datetime(
        df["submission_date"] + " " + df["submission_time"], format="%Y-%m-%d %H:%M:%S"
    )
    # sort by submission datetime
    # sort by public score and submission datetime
    if config.EVAL_HIGHER_IS_BETTER:
        df = df.sort_values(by=["public_score", "submission_datetime"], ascending=[False, True])
    else:
        df = df.sort_values(by=["public_score", "submission_datetime"], ascending=[True, True])
    # reset index
    df = df.reset_index(drop=True)
    df["rank"] = df.index + 1

    if private:
        columns = ["rank", "name", "private_score", "submission_datetime"]
    else:
        columns = ["rank", "name", "public_score", "submission_datetime"]
    return df[columns]
