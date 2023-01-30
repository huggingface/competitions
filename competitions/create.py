import uuid

import gradio as gr
import pandas as pd

from .utils import user_authentication


"""
To create a competition, follow these steps:
1. create a private dataset which has the following structure:
    - conf.json
    - solution.csv
    - COMPETITION_DESC.md
    - DATASET_DESC.md
2. create a public dataset which consists of the following files:
    - sample_submission.csv
    - test.csv
    - train.csv
    - anything else
3. create a competition space
"""


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
):
    # generate a random id
    suffix = str(uuid.uuid4())
    private_dataset_name = f"{competition_name}{suffix}"
    public_dataset_name = f"{competition_name}"
    space_name = f"{competition_name}"

    sample_submission_df = pd.read_csv(sample_submission_file, nrows=10)
    submission_columns = ",".join(sample_submission_df.columns)

    conf = {
        "SUBMISSION_LIMIT": submission_limit,
        "SELECTION_LIMIT": selection_limit,
        "END_DATE": end_date,
        "EVAL_HIGHER_IS_BETTER": True,
        "COMPETITION_NAME": competition_name,
        "SUBMISSION_COLUMNS": submission_columns,
        "EVAL_METRIC": eval_metric,
    }

    pass


def check_if_user_can_create_competition(user_token):
    """
    Check if the user can create a competition
    :param user_token: the user's token
    :return: True if the user can create a competition, False otherwise
    """
    user_info = user_authentication(user_token)
    print(user_info)
    return_msg = None
    if "error" in user_info:
        return_msg = "Invalid token. You can find your HF token here: https://huggingface.co/settings/tokens"

    elif user_info["auth"]["accessToken"]["role"] != "write":
        return_msg = "Please provide a token with write access"

    elif user_info["canPay"] is False:
        return_msg = "Please add a valid payment method in order to create and manage a competition"

    if return_msg is not None:
        return [
            gr.Box.update(visible=False),
            gr.Markdown.update(value=return_msg, visible=True),
            gr.Dropdown.update(visible=False),
        ]

    username = user_info["name"]
    user_id = user_info["id"]

    orgs = user_info["orgs"]
    valid_orgs = [org for org in orgs if org["canPay"] is True]
    valid_orgs = [org for org in valid_orgs if org["roleInOrg"] in ("admin", "write")]

    valid_entities = {org["id"]: org["name"] for org in valid_orgs}
    valid_entities[user_id] = username

    # reverse the dictionary
    valid_entities = {v: k for k, v in valid_entities.items()}

    return [
        gr.Box.update(visible=True),
        gr.Markdown.update(value="", visible=False),
        gr.Dropdown.update(choices=list(valid_entities.keys()), visible=True, value=username),
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
                ["accuracy", "auc", "f1", "logloss", "precision", "recall"],
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
                label="End Date (YYYY-MM-DD)",
            )
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
            """
        <p style="text-align: center">
        <h4>Please note that you will need to upload training and test
        data separately to the public repository that will be created.
        You can also change sample_submission and solution files later.</h4>
        </p>
        """
        )
        with gr.Row():
            create_button = gr.Button("Create Competition")

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
    ]
    create_button.click(create_competition, inputs=create_inputs, outputs=[message_box])
