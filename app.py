import uuid
from datetime import datetime
from functools import partial

import gradio as gr
import numpy as np
import pandas as pd
from huggingface_hub import HfApi
from huggingface_hub.utils._errors import EntryNotFoundError

import config
import utils

SUBMISSION_TEXT = f"""You can make upto {config.competition_info.submission_limit} submissions per day.
The test data has been divided into public and private splits.
Your score on the public split will be shown on the leaderboard.
Your final score will be based on your private split performance.
The final rankings will be based on the private split performance.
"""

SUBMISSION_ERROR = """Submission is not in a proper format.
Please check evaluation instructions for more details."""

SUBMISSION_LIMIT_TEXT = f"""You can select upto {config.competition_info.selection_limit}
 submissions for private leaderboard."""


def get_subs(user_info, private=False):
    # get user submissions
    user_id = user_info["id"]
    try:
        user_submissions = utils.fetch_submissions(user_id)
    except EntryNotFoundError:
        return_value = "No submissions found"
        return [gr.Textbox.update(visible=True, value=return_value), gr.DataFrame.update(visible=False)]
    submissions_df = pd.DataFrame(user_submissions)
    if not private:
        submissions_df = submissions_df.drop(columns=["private_score"])
        submissions_df = submissions_df[
            ["date", "submission_id", "public_score", "submission_comment", "selected", "status"]
        ]
    else:
        submissions_df = submissions_df[
            ["date", "submission_id", "public_score", "private_score", "submission_comment", "selected", "status"]
        ]
    return [gr.Textbox.update(visible=False), gr.DataFrame.update(visible=True, value=submissions_df)]


def my_submissions(user_token):
    if user_token != "":
        user_info = utils.user_authentication(token=user_token)
        print(user_info)
        if "error" in user_info:
            return_value = "Invalid token"
            return [gr.Textbox.update(visible=True, value=return_value), gr.DataFrame.update(visible=False)]

        if user_info["emailVerified"] is False:
            return_value = "Please verify your email on Hugging Face Hub"
            return [gr.Textbox.update(visible=True, value=return_value), gr.DataFrame.update(visible=False)]

        current_date_time = datetime.now()
        private = False
        if current_date_time >= config.competition_info.end_date:
            private = True
        subs = get_subs(user_info, private=private)
        return subs
    return [gr.Textbox.update(visible=True, value="Invalid token"), gr.DataFrame.update(visible=False)]


def new_submission(user_token):
    gr.Markdown(SUBMISSION_TEXT)
    uploaded_file = st.file_uploader("Choose a file")
    submit_button = st.button("Submit")
    if uploaded_file is not None and user_token != "" and submit_button:
        # verify token
        user_info = utils.user_authentication(token=user_token)
        if "error" in user_info:
            gr.Markdown("Invalid token")
            return

        if user_info["emailVerified"] is False:
            gr.Markdown("Please verify your email on Hugging Face Hub")
            return

        # check if user can submit to the competition
        if utils.check_user_submission_limit(user_info) is False:
            gr.Markdown("You have reached your submission limit for today")
            return

        bytes_data = uploaded_file.getvalue()
        # verify file is valid
        if not utils.verify_submission(bytes_data):
            gr.Markdown("Invalid submission")
            gr.Markdown(SUBMISSION_ERROR)
            # write a horizontal html line
            gr.Markdown("<hr/>", unsafe_allow_html=True)
        else:
            # TODO: add spinner here
            user_id = user_info["id"]
            submission_id = str(uuid.uuid4())
            file_extension = uploaded_file.name.split(".")[-1]
            # upload file to hf hub
            api = HfApi()
            api.upload_file(
                path_or_fileobj=bytes_data,
                path_in_repo=f"submissions/{user_id}-{submission_id}.{file_extension}",
                repo_id=config.COMPETITION_ID,
                repo_type="dataset",
                token=config.AUTOTRAIN_TOKEN,
            )
            # update submission limit
            submissions_made = utils.increment_submissions(
                user_id=user_id,
                submission_id=submission_id,
                submission_comment="",
            )
            # schedule submission for evaluation
            utils.create_project(
                project_id=f"{submission_id}",
                dataset=f"{config.COMPETITION_ID}",
                submission_dataset=user_id,
                model="generic_competition",
            )
        gr.Markdown("Submission scheduled for evaluation")
        gr.Markdown(
            f"You have {config.competition_info.submission_limit - submissions_made} submissions left for today."
        )


with gr.Blocks() as demo:
    with gr.Tab("Overview"):
        gr.Markdown(f"# Welcome to {config.competition_info.competition_name}! 👋")

        gr.Markdown(f"{config.competition_info.competition_description}")

        gr.Markdown("## Dataset")
        gr.Markdown(f"{config.competition_info.dataset_description}")

    with gr.Tab("Public Leaderboard"):
        lb = utils.fetch_leaderboard(private=False)
        gr.Markdown(lb.to_markdown())
    with gr.Tab("Private Leaderboard"):
        current_date_time = datetime.now()
        if current_date_time >= config.competition_info.end_date:
            lb = utils.fetch_leaderboard(private=True)
            gr.Markdown(lb.to_markdown())
        else:
            gr.Markdown("Private Leaderboard will be available after the competition ends")
    with gr.Tab("New Submission"):
        text_input = gr.Textbox()
        text_output = gr.Textbox()
        text_button = gr.Button("Flip")
    with gr.Tab("My Submissions"):
        gr.Markdown(SUBMISSION_LIMIT_TEXT)
        user_token = gr.Textbox(max_lines=1, value="hf_XXX", label="Please enter your Hugging Face token")
        output_text = gr.Textbox(visible=True, show_label=False)
        empty_df = pd.DataFrame(
            columns=[
                "date",
                "submission_id",
                "public_score",
                "private_score",
                "submission_comment",
                "selected",
                "status",
            ]
        )
        output_df = gr.Dataframe(visible=False, value=empty_df)
        my_subs_button = gr.Button("Fetch Submissions")
        my_subs_button.click(fn=my_submissions, inputs=[user_token], outputs=[output_text, output_df])

if __name__ == "__main__":
    demo.launch()
