import uuid
from datetime import datetime
from functools import partial

import config
import gradio as gr
import pandas as pd
import utils
from huggingface_hub import HfApi
from huggingface_hub.utils._errors import EntryNotFoundError


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
        return [gr.Markdown.update(visible=True, value=return_value), gr.DataFrame.update(visible=False)]
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
    return [gr.Markdown.update(visible=False), gr.DataFrame.update(visible=True, value=submissions_df)]


def my_submissions(user_token):
    if user_token != "":
        user_info = utils.user_authentication(token=user_token)
        if "error" in user_info:
            return_value = "Invalid token"
            return [gr.Markdown.update(visible=True, value=return_value), gr.DataFrame.update(visible=False)]

        if user_info["emailVerified"] is False:
            return_value = "Please verify your email on Hugging Face Hub"
            return [gr.Markdown.update(visible=True, value=return_value), gr.DataFrame.update(visible=False)]

        current_date_time = datetime.now()
        private = False
        if current_date_time >= config.competition_info.end_date:
            private = True
        subs = get_subs(user_info, private=private)
        return subs
    return [gr.Markdown.update(visible=True, value="Invalid token"), gr.DataFrame.update(visible=False)]


def new_submission(user_token, uploaded_file):
    if uploaded_file is not None and user_token != "":
        # verify token
        user_info = utils.user_authentication(token=user_token)
        if "error" in user_info:
            return "Invalid token"

        if user_info["emailVerified"] is False:
            return "Please verify your email on Hugging Face Hub"

        # check if user can submit to the competition
        if utils.check_user_submission_limit(user_info) is False:
            return "You have reached your submission limit for today"
        with open(uploaded_file.name, "rb") as f:
            bytes_data = f.read()
        # verify file is valid
        if not utils.verify_submission(bytes_data):
            return "Invalid submission"
            # gr.Markdown(SUBMISSION_ERROR)
            # write a horizontal html line
            # gr.Markdown("<hr/>", unsafe_allow_html=True)
        else:
            user_id = user_info["id"]
            submission_id = str(uuid.uuid4())
            file_extension = uploaded_file.orig_name.split(".")[-1]
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
        return_text = f"Submission scheduled for evaluation. You have {config.competition_info.submission_limit - submissions_made} submissions left for today."
        return return_text
    return "Error"


with gr.Blocks() as demo:
    with gr.Tab("Overview"):
        gr.Markdown(f"# Welcome to {config.competition_info.competition_name}! 👋")

        gr.Markdown(f"{config.competition_info.competition_description}")

        gr.Markdown("## Dataset")
        gr.Markdown(f"{config.competition_info.dataset_description}")

    with gr.Tab("Public Leaderboard"):
        output_markdown = gr.Markdown("")
        fetch_lb = gr.Button("Fetch Leaderboard")
        fetch_lb_partial = partial(utils.fetch_leaderboard, private=False)
        fetch_lb.click(fn=fetch_lb_partial, outputs=[output_markdown])
        # lb = utils.fetch_leaderboard(private=False)
        # gr.Markdown(lb.to_markdown())
    with gr.Tab("Private Leaderboard"):
        current_date_time = datetime.now()
        if current_date_time >= config.competition_info.end_date:
            lb = utils.fetch_leaderboard(private=True)
            gr.Markdown(lb)
        else:
            gr.Markdown("Private Leaderboard will be available after the competition ends")
    with gr.Tab("New Submission"):
        gr.Markdown(SUBMISSION_TEXT)
        user_token = gr.Textbox(max_lines=1, value="hf_XXX", label="Please enter your Hugging Face token")
        uploaded_file = gr.File()
        output_text = gr.Markdown(visible=True, show_label=False)
        new_sub_button = gr.Button("Upload Submission")
        new_sub_button.click(fn=new_submission, inputs=[user_token, uploaded_file], outputs=[output_text])

    with gr.Tab("My Submissions"):
        gr.Markdown(SUBMISSION_LIMIT_TEXT)
        user_token = gr.Textbox(max_lines=1, value="hf_XXX", label="Please enter your Hugging Face token")
        output_text = gr.Markdown(visible=True, show_label=False)
        output_df = gr.Dataframe(visible=False)
        my_subs_button = gr.Button("Fetch Submissions")
        my_subs_button.click(fn=my_submissions, inputs=[user_token], outputs=[output_text, output_df])

if __name__ == "__main__":
    demo.launch()
