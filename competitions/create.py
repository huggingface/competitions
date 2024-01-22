import io
import json

import gradio as gr
from huggingface_hub import HfApi
from loguru import logger
from sklearn.metrics import get_scorer_names

from competitions.utils import user_authentication


COMPETITION_DESC = """Sample competition description"""
DATASET_DESC = """Sample dataset description"""
SUBMISSION_DESC = """Sample submission description"""
SOLUTION_CSV = """
id,pred,split
0,1,public
1,0,private
2,0,private
3,1,private
4,0,public
5,1,private
6,1,public
7,1,private
8,0,public
9,0,private
10,0,private
11,0,private
12,1,private
13,0,private
14,1,public
15,1,private
16,1,private
17,0,private
18,0,private
19,0,public
20,0,private
21,0,private
22,1,private
23,1,public
24,0,private
25,0,private
26,0,public
27,1,private
28,1,private
29,0,private
30,0,public
"""
SOLUTION_CSV = SOLUTION_CSV.strip()

DOCKERFILE = """
FROM huggingface/competitions:latest

CMD uvicorn competitions.app:app --host 0.0.0.0 --port 7860 --workers 1
"""
DOCKERFILE = DOCKERFILE.replace("\n", " ").replace("  ", "\n").strip()

HARDWARE_CHOICES = [
    "cpu-basic",
    "cpu-upgrade",
    "t4-small",
    "t4-medium",
    "a10g-small",
    "a10g-large",
    "a10g-largex2",
    "a10g-largex4",
    "a100-large",
]
METRIC_CHOICES = get_scorer_names() + ["custom"]


def check_if_user_can_create_competition(user_token):
    """
    Check if the user can create a competition
    :param user_token: the user's token
    :return: True if the user can create a competition, False otherwise
    """
    user_info = user_authentication(user_token)
    return_msg = None
    if "error" in user_info:
        return_msg = "Invalid token. You can find your HF token here: https://huggingface.co/settings/tokens"

    elif user_info["auth"]["accessToken"]["role"] != "write":
        return_msg = "Please provide a token with write access"

    if return_msg is not None:
        return gr.Dropdown()

    orgs = user_info["orgs"]
    valid_orgs = [org for org in orgs if org["canPay"] is True]

    if len(valid_orgs) == 0:
        return_msg = """You are not a member of any organization with a valid payment method.
        Please add a valid payment method for your organization in order to create competitions."""
        return gr.Dropdown()

    valid_orgs = [org for org in valid_orgs if org["roleInOrg"] in ("admin", "write")]

    if len(valid_orgs) == 0:
        return_msg = """You dont have write access for any organization.
        Please contact your organization's admin to add you as a member with write privilages."""
        return gr.Dropdown()

    valid_entities = {org["name"]: org["id"] for org in valid_orgs}

    return gr.Dropdown(
        choices=list(valid_entities.keys()),
        visible=True,
        value=list(valid_entities.keys())[0],
    )


def _create_readme(competition_name):
    _readme = "---\n"
    _readme += f"title: {competition_name}\n"
    _readme += "emoji: 🚀\n"
    _readme += "colorFrom: green\n"
    _readme += "colorTo: indigo\n"
    _readme += "sdk: docker\n"
    _readme += "pinned: false\n"
    _readme += "duplicated_from: autotrain-projects/autotrain-advanced\n"
    _readme += "---\n"
    _readme = io.BytesIO(_readme.encode())
    return _readme


def _create(
    user_token,
    organization,
    competition_name,
    competition_logo,
    hardware,
    competition_type,
    time_limit,
    metric,
    metric_higher_is_better,
    submission_limit,
    selection_limit,
    end_date,
    submission_id_column,
    submission_columns,
    submission_rows,
):
    """
    Create a competition
    """

    # make sure competition name is alphanumeric
    competition_name = "".join([c for c in competition_name if c.isalnum()])
    if len(competition_name) == 0:
        raise gr.Error("Please provide a valid alphanumeric competition name")

    conf_json = {
        "COMPETITION_TYPE": competition_type,
        "SUBMISSION_LIMIT": int(submission_limit),
        "TIME_LIMIT": int(time_limit),
        "SELECTION_LIMIT": int(selection_limit),
        "HARDWARE": hardware,
        "END_DATE": end_date,
        "EVAL_HIGHER_IS_BETTER": metric_higher_is_better == "True",
        "SUBMISSION_ID_COLUMN": submission_id_column,
        "SUBMISSION_COLUMNS": submission_columns,
        "SUBMISSION_ROWS": int(submission_rows),
        "EVAL_METRIC": metric,
        "LOGO": competition_logo,
    }
    teams_json = {}
    user_team_json = {}

    logger.info(f"Creating competition: {competition_name}")

    api = HfApi()
    api.create_repo(
        repo_id=f"{organization}/{competition_name}",
        repo_type="dataset",
        private=True,
    )

    conf_json = json.dumps(conf_json, indent=4)
    conf_json_bytes = conf_json.encode("utf-8")
    conf_json_buffer = io.BytesIO(conf_json_bytes)
    api.upload_file(
        path_or_fileobj=conf_json_buffer,
        path_in_repo="conf.json",
        repo_id=f"{organization}/{competition_name}",
        repo_type="dataset",
    )

    teams_json = json.dumps(teams_json, indent=4)
    teams_json_bytes = teams_json.encode("utf-8")
    teams_json_buffer = io.BytesIO(teams_json_bytes)
    api.upload_file(
        path_or_fileobj=teams_json_buffer,
        path_in_repo="teams.json",
        repo_id=f"{organization}/{competition_name}",
        repo_type="dataset",
    )

    user_team_json = json.dumps(user_team_json, indent=4)
    user_team_json_bytes = user_team_json.encode("utf-8")
    user_team_json_buffer = io.BytesIO(user_team_json_bytes)
    api.upload_file(
        path_or_fileobj=user_team_json_buffer,
        path_in_repo="user_team.json",
        repo_id=f"{organization}/{competition_name}",
        repo_type="dataset",
    )

    comp_desc = io.BytesIO(COMPETITION_DESC.encode())
    api.upload_file(
        path_or_fileobj=comp_desc,
        path_in_repo="COMPETITION_DESC.md",
        repo_id=f"{organization}/{competition_name}",
        repo_type="dataset",
    )

    dataset_desc = io.BytesIO(DATASET_DESC.encode())
    api.upload_file(
        path_or_fileobj=dataset_desc,
        path_in_repo="DATASET_DESC.md",
        repo_id=f"{organization}/{competition_name}",
        repo_type="dataset",
    )

    submission_desc = io.BytesIO(SUBMISSION_DESC.encode())
    api.upload_file(
        path_or_fileobj=submission_desc,
        path_in_repo="SUBMISSION_DESC.md",
        repo_id=f"{organization}/{competition_name}",
        repo_type="dataset",
    )

    solution_csv = io.BytesIO(SOLUTION_CSV.encode())
    api.upload_file(
        path_or_fileobj=solution_csv,
        path_in_repo="solution.csv",
        repo_id=f"{organization}/{competition_name}",
        repo_type="dataset",
    )

    # create competition space
    api.create_repo(
        repo_id=f"{organization}/{competition_name}",
        repo_type="space",
        space_sdk="docker",
        space_hardware="cpu-basic" if competition_type == "script" else hardware,
        private=True,
    )
    api.add_space_secret(repo_id=f"{organization}/{competition_name}", key="HF_TOKEN", value=user_token)
    api.add_space_secret(
        repo_id=f"{organization}/{competition_name}",
        key="COMPETITION_ID",
        value=f"{organization}/{competition_name}",
    )
    readme = _create_readme(competition_name)
    api.upload_file(
        path_or_fileobj=readme,
        path_in_repo="README.md",
        repo_id=f"{organization}/{competition_name}",
        repo_type="space",
    )

    _dockerfile = io.BytesIO(DOCKERFILE.encode())
    api.upload_file(
        path_or_fileobj=_dockerfile,
        path_in_repo="Dockerfile",
        repo_id=f"{organization}/{competition_name}",
        repo_type="space",
    )

    return gr.Markdown(
        value=f"""Created private dataset and competition space.
        To make competition public, you should make the space public.
        Please note that the dataset should always be kept private.

        Private dataset: https://huggingface.co/datasets/{organization}/{competition_name}

        Competition space: https://huggingface.co/spaces/{organization}/{competition_name}

        Note: there's still some work left. Now you must change the solution.csv file to your own solution,
        and make changes to *_desc.md files to reflect your competition. You may also change conf.json
        to suit your needs. Please refer to the [documentation](https://hf.co/docs/competitions) for more information.
        """
    )


def main():
    with gr.Blocks() as demo:
        gr.Markdown("# Hugging Face Competition Creator")
        token = gr.Textbox(label="Your Hugging Face write token", lines=1, type="password")
        with gr.Row():
            organization = gr.Dropdown(label="Organization name", choices=[""])
            competition_name = gr.Textbox(label="Competition name", lines=1)
            competition_logo = gr.Textbox(label="Competition logo", value="https://mysite.com/mylogo.png", lines=1)
        with gr.Row():
            hardware = gr.Dropdown(label="Hardware to use", choices=HARDWARE_CHOICES, value=HARDWARE_CHOICES[0])
            competition_type = gr.Dropdown(label="Competition type", choices=["generic", "script"], value="generic")
            time_limit = gr.Textbox(label="Time limit (s). Only used for script competitions", lines=1, value="3600")
        with gr.Row():
            metric = gr.Dropdown(label="Metric to use", choices=METRIC_CHOICES, value=METRIC_CHOICES[0])
            metric_higher_is_better = gr.Dropdown(label="Is higher metric better?", choices=[True, False], value=True)
        with gr.Row():
            submission_limit = gr.Textbox(label="Submission limit per day", lines=1, value="5")
            selection_limit = gr.Textbox(label="Final selection limit", lines=1, value="2")
            end_date = gr.Textbox(label="End date (YYYY-MM-DD)", lines=1, value="2024-12-31")
        with gr.Row():
            submission_id_column = gr.Textbox(label="Submission id column", lines=1, value="id")
            submission_columns = gr.Textbox(label="Submission columns", lines=1, value="id,pred")
            submission_rows = gr.Textbox(label="Submission total rows (exclusing header)", lines=1, value="10000")

        output_md = gr.Markdown("Click the button below to create the competition")
        create_competition = gr.Button(value="Create competition")
        token.change(check_if_user_can_create_competition, inputs=token, outputs=organization)

        create_competition.click(
            _create,
            inputs=[
                token,
                organization,
                competition_name,
                competition_logo,
                hardware,
                competition_type,
                time_limit,
                metric,
                metric_higher_is_better,
                submission_limit,
                selection_limit,
                end_date,
                submission_id_column,
                submission_columns,
                submission_rows,
            ],
            outputs=output_md,
        )
    return demo


if __name__ == "__main__":
    main().launch()
