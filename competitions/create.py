import io
import json
import uuid

import gradio as gr
import pandas as pd
from huggingface_hub import HfApi, create_repo

from . import BOT_TOKEN
from .utils import user_authentication


def verify_sample_and_solution(sample_submission, solution, eval_metric):
    sample_submission = pd.read_csv(sample_submission.name)
    solution = pd.read_csv(solution.name)

    # check if both contain an id column
    if "id" not in sample_submission.columns:
        raise Exception("Sample submission should contain an id column")

    if "id" not in solution.columns:
        raise Exception("Solution file should contain an id column")

    if eval_metric != "map-iou":
        # check if both files have the same ids
        if not (sample_submission["id"] == solution["id"]).all():
            raise Exception("Sample submission and solution should have the same ids")

        # check if both files have the same number of rows
        if sample_submission.shape[0] != solution.shape[0]:
            raise Exception("Sample submission and solution should have the same number of rows")

    # check if solution contains a split column
    if "split" not in solution.columns:
        raise Exception("Solution file should contain a split column")

    # check if split column contains only two unique values
    if len(solution["split"].unique()) != 2:
        raise Exception("Split column should contain only two unique values: public and private")

    # check if unique values are public and private
    if not set(solution["split"].unique()) == set(["public", "private"]):
        raise Exception("Split column should contain only two unique values: public and private")

    if eval_metric != "map-iou":
        # except the `split` column, all other columns should be the same
        solution_columns = solution.columns.tolist()
        solution_columns.remove("split")
        if not (sample_submission.columns == solution_columns).all():
            raise Exception("Sample submission and solution should have the same columns, except for the split column")

    return True


def create_competition(
    user_token,
    who_pays,
    competition_type,
    competition_name,
    eval_metric,
    submission_limit,
    selection_limit,
    end_date,
    sample_submission_file,
    solution_file,
    is_public,
):
    # verify sample submission and solution
    try:
        verify_sample_and_solution(sample_submission_file, solution_file, eval_metric)
    except Exception as e:
        return gr.Markdown.update(
            value=f"""
        <div style="text-align: center">
            <h4>Invalid sample submission or solution file</h4>
            <p>{e}</p>
        </div>
        """,
            visible=True,
        )

    # check if end_date is valid format: YYYY-MM-DD and in the future
    try:
        if len(end_date.split("-")) != 3:
            raise Exception("End date should be in the format YYYY-MM-DD")
        end_date_pd = pd.to_datetime(end_date)
        if end_date_pd == pd.NaT:
            raise Exception("End date should be in the format YYYY-MM-DD")
        if end_date_pd <= pd.to_datetime("today"):
            raise Exception("End date should be in the future")
    except Exception as e:
        return gr.Markdown.update(
            value=f"""
        <div style="text-align: center">
            <h4>Invalid end date</h4>
            <p>{e}</p>
        </div>
        """,
            visible=True,
        )

    is_public = is_public == "Public"
    suffix = str(uuid.uuid4())
    private_dataset_name = f"{who_pays}/{competition_name}{suffix}"
    public_dataset_name = f"{who_pays}/{competition_name}"
    if is_public:
        space_name = f"competitions/{competition_name}"
    else:
        space_name = f"{who_pays}/{competition_name}"

    sample_submission_df = pd.read_csv(sample_submission_file.name)
    submission_columns = ",".join(sample_submission_df.columns)

    conf = {
        "COMPETITION_TYPE": competition_type,
        "SUBMISSION_LIMIT": submission_limit,
        "SELECTION_LIMIT": selection_limit,
        "END_DATE": end_date,
        "EVAL_HIGHER_IS_BETTER": 1 if eval_metric != "logloss" else 0,
        "COMPETITION_NAME": competition_name,
        "SUBMISSION_ID_COLUMN": "id",
        "SUBMISSION_COLUMNS": submission_columns,
        "SUBMISSION_ROWS": len(sample_submission_df),
        "EVAL_METRIC": eval_metric,
    }
    if eval_metric == "map-iou":
        conf["IOU_THRESHOLD"] = 0.5

    api = HfApi()

    # create private dataset repo
    try:
        create_repo(
            repo_id=private_dataset_name,
            repo_type="dataset",
            private=True,
            token=user_token,
            exist_ok=False,
        )
    except Exception as e:
        return gr.Markdown.update(
            value=f"""
        <div style="text-align: center">
            <h4>Failed to create private dataset repo</h4>
            <p>{e}</p>
        </div>
        """,
            visible=True,
        )
    competition_desc = f"""
    # Welcome to {competition_name}

    This is a competition description.

    You can use markdown to format your description.
    """

    dataset_desc = f"""
    # Dataset Description

    This is a dataset description.

    You can use markdown to format your description.

    Dataset can be downloaded from [here](https://hf.co/datasets/{public_dataset_name})
    """

    conf_json = json.dumps(conf)
    conf_bytes = conf_json.encode("utf-8")
    conf_buffer = io.BytesIO(conf_bytes)

    api.upload_file(
        path_or_fileobj=conf_buffer,
        path_in_repo="conf.json",
        repo_id=private_dataset_name,
        repo_type="dataset",
        token=user_token,
    )

    # convert competition description to bytes
    competition_desc_bytes = competition_desc.encode("utf-8")
    competition_desc_buffer = io.BytesIO(competition_desc_bytes)

    api.upload_file(
        path_or_fileobj=competition_desc_buffer,
        path_in_repo="COMPETITION_DESC.md",
        repo_id=private_dataset_name,
        repo_type="dataset",
        token=user_token,
    )

    # convert dataset description to bytes
    dataset_desc_bytes = dataset_desc.encode("utf-8")
    dataset_desc_buffer = io.BytesIO(dataset_desc_bytes)

    api.upload_file(
        path_or_fileobj=dataset_desc_buffer,
        path_in_repo="DATASET_DESC.md",
        repo_id=private_dataset_name,
        repo_type="dataset",
        token=user_token,
    )

    with open(solution_file.name, "rb") as f:
        solution_bytes_data = f.read()
    # upload solution file
    api.upload_file(
        path_or_fileobj=solution_bytes_data,
        path_in_repo="solution.csv",
        repo_id=private_dataset_name,
        repo_type="dataset",
        token=user_token,
    )

    # create public dataset repo
    try:
        create_repo(
            repo_id=public_dataset_name,
            repo_type="dataset",
            private=False,
            token=user_token,
            exist_ok=False,
        )
    except Exception as e:
        return gr.Markdown.update(
            value=f"""
        <div style="text-align: center">
            <h4>Failed to create public dataset repo</h4>
            <p>{e}</p>
        </div>
        """,
            visible=True,
        )

    # upload sample submission file
    with open(sample_submission_file.name, "rb") as f:
        sample_submission_bytes_data = f.read()

    api.upload_file(
        path_or_fileobj=sample_submission_bytes_data,
        path_in_repo="sample_submission.csv",
        repo_id=public_dataset_name,
        repo_type="dataset",
        token=user_token,
    )

    dockerfile = """
    FROM huggingface/competitions:latest
    CMD competitions run
    """
    dockerfile = dockerfile.strip()
    dockerfile = dockerfile.replace("    ", "")

    # create competition space
    create_repo(
        repo_id=space_name,
        repo_type="space",
        private=False,
        token=BOT_TOKEN if is_public else user_token,
        space_sdk="docker",
        exist_ok=False,
    )

    # upload dockerfile
    dockerfile_bytes = dockerfile.encode("utf-8")
    dockerfile_buffer = io.BytesIO(dockerfile_bytes)

    api.upload_file(
        path_or_fileobj=dockerfile_buffer,
        path_in_repo="Dockerfile",
        repo_id=space_name,
        repo_type="space",
        token=BOT_TOKEN if is_public else user_token,
    )

    space_readme = f"""
    ---
    title: {competition_name}
    emoji: 🏆
    colorFrom: blue
    colorTo: gray
    sdk: docker
    pinned: false
    ---
    """
    space_readme = space_readme.strip()
    space_readme = space_readme.replace("    ", "")

    # upload space readme
    space_readme_bytes = space_readme.encode("utf-8")
    space_readme_buffer = io.BytesIO(space_readme_bytes)

    api.upload_file(
        path_or_fileobj=space_readme_buffer,
        path_in_repo="README.md",
        repo_id=space_name,
        repo_type="space",
        token=BOT_TOKEN if is_public else user_token,
    )

    api.add_space_secret(
        repo_id=space_name,
        key="COMPETITION_ID",
        value=private_dataset_name,
        token=BOT_TOKEN if is_public else user_token,
    )
    api.add_space_secret(
        repo_id=space_name,
        key="AUTOTRAIN_USERNAME",
        value=who_pays,
        token=BOT_TOKEN if is_public else user_token,
    )
    api.add_space_secret(
        repo_id=space_name,
        key="AUTOTRAIN_TOKEN",
        value=user_token,
        token=BOT_TOKEN if is_public else user_token,
    )

    return gr.Markdown.update(
        value=f"""
        <div style="text-align: center">
            <h4>Competition created successfully!</h4>
            <p>Private dataset: <a href="https://hf.co/datasets/{private_dataset_name}">{private_dataset_name}</a></p>
            <p>Public dataset: <a href="https://hf.co/datasets/{public_dataset_name}">{public_dataset_name}</a></p>
            <p>Competition space: <a href="https://hf.co/spaces/{space_name}">{space_name}</a></p>
            <p>NOTE: for private competitions, please add `autoevaluator` user to your org: {who_pays}.</p>
            <p>NOTE: Do NOT share the private dataset or link with anyone else.</p>
        </div>
        """,
        visible=True,
    )


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
        return [
            gr.Box.update(visible=False),
            gr.Markdown.update(value=return_msg, visible=True),
            gr.Dropdown.update(visible=False),
        ]

    orgs = user_info["orgs"]
    valid_orgs = [org for org in orgs if org["canPay"] is True]

    if len(valid_orgs) == 0:
        return_msg = """You are not a member of any organization with a valid payment method.
        Please add a valid payment method for your organization in order to create competitions."""
        return [
            gr.Box.update(visible=False),
            gr.Markdown.update(
                value=return_msg,
                visible=True,
            ),
            gr.Dropdown.update(visible=False),
        ]

    valid_orgs = [org for org in valid_orgs if org["roleInOrg"] in ("admin", "write")]

    if len(valid_orgs) == 0:
        return_msg = """You dont have write access for any organization.
        Please contact your organization's admin to add you as a member with write privilages."""
        return [
            gr.Box.update(visible=False),
            gr.Markdown.update(
                value=return_msg,
                visible=True,
            ),
            gr.Dropdown.update(visible=False),
        ]

    valid_entities = {org["name"]: org["id"] for org in valid_orgs}

    return [
        gr.Box.update(visible=True),
        gr.Markdown.update(value="", visible=False),
        gr.Dropdown.update(
            choices=list(valid_entities.keys()),
            visible=True,
            value=list(valid_entities.keys())[0],
        ),
    ]


with gr.Blocks() as demo:
    gr.Markdown(
        """
        <div style="text-align: center">
        <h1>Hugging Face Competition Creator</h1>
        """
    )
    user_token = gr.Textbox(
        max_lines=1,
        value="",
        label="Please enter your Hugging Face token (write access needed)",
        type="password",
    )
    login_button = gr.Button("Login")

    message_box = gr.Markdown(visible=False)

    with gr.Box(visible=False) as create_box:
        who_pays = gr.Dropdown(
            ["Me", "My Organization"],
            label="Who Pays",
            value="Me",
            visible=False,
            interactive=True,
        )
        competition_type = gr.Radio(
            ["Generic"],
            label="Competition Type",
            value="Generic",
        )

        with gr.Row():
            competition_name = gr.Textbox(
                max_lines=1,
                value="",
                label="Competition Name",
                placeholder="my-awesome-competition",
            )
            eval_metric = gr.Dropdown(
                ["accuracy", "auc", "f1", "logloss", "map-iou", "precision", "recall"],
                label="Evaluation Metric",
                value="accuracy",
            )
        with gr.Row():
            submission_limit = gr.Slider(
                minimum=1,
                maximum=100,
                value=5,
                step=1,
                label="Submission Limit Per Day",
            )
            selection_limit = gr.Slider(
                minimum=1,
                maximum=100,
                value=2,
                step=1,
                label="Selection Limit For Final Leaderboard",
            )
            end_date = gr.Textbox(
                max_lines=1,
                value="",
                label="End Date (YYYY-MM-DD), Private LB will be visible on this date",
            )
        with gr.Box():
            with gr.Row():
                with gr.Column():
                    sample_submission_file = gr.File(
                        label="sample_submission.csv",
                    )
                with gr.Column():
                    solution_file = gr.File(
                        label="solution.csv",
                    )
            gr.Markdown(
                """Please note that you will need to upload training and test
            data separately to the public repository that will be created.
            You can also change sample_submission and solution files later.
            """
            )
        with gr.Box():
            with gr.Row():
                is_public = gr.Dropdown(
                    ["Public", "Private"],
                    label="Competition Visibility. Private competitions are only visible to you and your organization members and are created inside your organization. Public competitions are available at hf.co/competitions.",
                    value="Public",
                )
            with gr.Row():
                create_button = gr.Button("Create Competition")

    final_output = gr.Markdown(visible=True)

    login_button.click(
        check_if_user_can_create_competition, inputs=[user_token], outputs=[create_box, message_box, who_pays]
    )

    create_inputs = [
        user_token,
        who_pays,
        competition_type,
        competition_name,
        eval_metric,
        submission_limit,
        selection_limit,
        end_date,
        sample_submission_file,
        solution_file,
        is_public,
    ]
    create_button.click(create_competition, inputs=create_inputs, outputs=[final_output])
