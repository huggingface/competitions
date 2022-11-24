from datetime import datetime
from functools import partial

import gradio as gr

from . import AUTOTRAIN_BACKEND_API, AUTOTRAIN_TOKEN, AUTOTRAIN_USERNAME, COMPETITION_ID, competition_info
from .errors import SubmissionError, SubmissionLimitError
from .leaderboard import Leaderboard
from .submissions import Submissions
from .text import (
    NO_SUBMISSIONS,
    SUBMISSION_LIMIT_REACHED,
    SUBMISSION_SELECTION_TEXT,
    SUBMISSION_SUCCESS,
    SUBMISSION_TEXT,
)


leaderboard = Leaderboard(
    end_date=competition_info.end_date,
    eval_higher_is_better=competition_info.eval_higher_is_better,
    max_selected_submissions=competition_info.selection_limit,
    competition_id=COMPETITION_ID,
    autotrain_token=AUTOTRAIN_TOKEN,
)

submissions = Submissions(
    competition_id=competition_info.competition_id,
    submission_limit=competition_info.submission_limit,
    end_date=competition_info.end_date,
    autotrain_username=AUTOTRAIN_USERNAME,
    autotrain_token=AUTOTRAIN_TOKEN,
    autotrain_backend_api=AUTOTRAIN_BACKEND_API,
)


def _new_submission(user_token, submission_file):
    try:
        remaining_subs = submissions.new_submission(user_token, submission_file)
        return SUBMISSION_SUCCESS.format(remaining_subs)
    except SubmissionLimitError:
        return SUBMISSION_LIMIT_REACHED
    except SubmissionError:
        return "Something went wrong. Please try again later."


def _my_submissions(user_token):
    df = submissions.my_submissions(user_token)
    if len(df) == 0:
        return [
            gr.Markdown.update(visible=True, value=NO_SUBMISSIONS),
            gr.DataFrame.update(visible=False),
            gr.TextArea.update(visible=False),
            gr.Button.update(visible=False),
        ]
    selected_submission_ids = df[df["selected"] == True]["submission_id"].values.tolist()
    if len(selected_submission_ids) > 0:
        return [
            gr.Markdown.update(visible=True),
            gr.DataFrame.update(visible=True, value=df),
            gr.TextArea.update(visible=True, value="\n".join(selected_submission_ids), interactive=True),
            gr.Button.update(visible=True),
        ]
    return [
        gr.Markdown.update(visible=False),
        gr.DataFrame.update(visible=True, value=df),
        gr.TextArea.update(visible=True, interactive=True),
        gr.Button.update(visible=True),
    ]


def _update_selected_submissions(user_token, submission_ids):
    submission_ids = submission_ids.split("\n")
    submission_ids = [sid.strip() for sid in submission_ids]
    submission_ids = [sid for sid in submission_ids if len(sid) > 0]
    if len(submission_ids) > competition_info.selection_limit:
        raise ValueError(
            f"You can select only {competition_info.selection_limit} submissions. You selected {len(submission_ids)} submissions."
        )
    submissions.update_selected_submissions(user_token, submission_ids)
    return _my_submissions(user_token)


def _fetch_leaderboard(private):
    if private:
        current_date_time = datetime.now()
        if current_date_time < competition_info.end_date:
            return [
                gr.DataFrame.update(visible=False),
                gr.Markdown.update(
                    visible=True, value="Private Leaderboard will be available after the competition ends"
                ),
            ]
    df = leaderboard.fetch(private=private)
    num_teams = len(df)
    return [
        gr.DataFrame.update(visible=True, value=df),
        gr.Markdown.update(visible=True, value=f"Number of teams: {num_teams}"),
    ]


with gr.Blocks() as demo:
    with gr.Tabs() as tab_container:
        with gr.TabItem("Overview", id="overview"):
            gr.Markdown(f"# Welcome to {competition_info.competition_name}! 👋")
            gr.Markdown(f"{competition_info.competition_description}")
            gr.Markdown("## Dataset")
            gr.Markdown(f"{competition_info.dataset_description}")
        with gr.TabItem("Public Leaderboard", id="public_leaderboard") as public_leaderboard:
            output_text_public = gr.Markdown()
            output_df_public = gr.DataFrame(
                row_count=(50, "dynamic"), overflow_row_behaviour="paginate", visible=False
            )
        with gr.TabItem("Private Leaderboard", id="private_leaderboard") as private_leaderboard:
            output_text_private = gr.Markdown()
            output_df_private = gr.DataFrame(
                row_count=(50, "dynamic"), overflow_row_behaviour="paginate", visible=False
            )
        with gr.TabItem("New Submission", id="new_submission"):
            gr.Markdown(SUBMISSION_TEXT.format(competition_info.submission_limit))
            user_token = gr.Textbox(
                max_lines=1, value="", label="Please enter your Hugging Face token", type="password"
            )
            uploaded_file = gr.File()
            output_text = gr.Markdown(visible=True, show_label=False)
            new_sub_button = gr.Button("Upload Submission")
            new_sub_button.click(
                fn=_new_submission,
                inputs=[user_token, uploaded_file],
                outputs=[output_text],
            )
        with gr.TabItem("My Submissions", id="my_submissions"):
            gr.Markdown(SUBMISSION_SELECTION_TEXT.format(competition_info.selection_limit))
            user_token = gr.Textbox(
                max_lines=1, value="", label="Please enter your Hugging Face token", type="password"
            )
            output_text = gr.Markdown(visible=True, show_label=False)
            output_df = gr.DataFrame(visible=False)
            selected_submissions = gr.TextArea(
                visible=False,
                label="Selected Submissions (one submission id per line)",
                max_lines=competition_info.selection_limit,
                lines=competition_info.selection_limit,
            )
            update_selected_submissions = gr.Button("Update Selected Submissions", visible=False)
            my_subs_button = gr.Button("Fetch Submissions")
            my_subs_button.click(
                fn=_my_submissions,
                inputs=[user_token],
                outputs=[output_text, output_df, selected_submissions, update_selected_submissions],
            )
            update_selected_submissions.click(
                fn=_update_selected_submissions,
                inputs=[user_token, selected_submissions],
                outputs=[output_text, output_df, selected_submissions, update_selected_submissions],
            )

        fetch_lb_partial = partial(_fetch_leaderboard, private=False)
        public_leaderboard.select(fetch_lb_partial, inputs=[], outputs=[output_df_public, output_text_public])
        fetch_lb_partial_private = partial(_fetch_leaderboard, private=True)
        private_leaderboard.select(
            fetch_lb_partial_private, inputs=[], outputs=[output_df_private, output_text_private]
        )
