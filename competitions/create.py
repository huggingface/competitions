import gradio as gr

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


def check_if_user_can_create_competition(user_token):
    """
    Check if the user can create a competition
    :param user_token: the user's token
    :return: True if the user can create a competition, False otherwise
    """
    user_info = user_authentication(user_token)

    if "error" in user_info:
        raise Exception("Invalid token. You can find your HF token here: https://huggingface.co/settings/tokens")

    if user_info["auth"]["accessToken"]["role"] != "write":
        raise Exception("Please provide a token with write access")

    if user_info["canPay"] is False:
        raise Exception("Please add a valid payment method in order to create and manage a competition")

    return [gr.Box.update(visible=True)]


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
    org_name = gr.Textbox(
        max_lines=1,
        value="",
        label="Please enter your username/organization name where the competition datasets will be hosted. Leave blank if you want to create a competition space in your personal account.",
    )
    with gr.Box():
        gr.Markdown(
            """
            Pricing:
            - Generic: $0.50 per submission
            - Hub Model: Coming Soon!
            """
        )
        competition_type = gr.Radio(
            ["Generic"],
            label="Competition Type",
            value="Generic",
        )

    with gr.Box():
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
        Please note that you will need to upload training and test
        data separately to the public repository that will be created.
        You can also change sample_submission and solution files later.
        """
        )
        with gr.Row():
            gr.Button("Create Competition")
