from datetime import datetime
from functools import partial

import gradio as gr

from . import AUTOTRAIN_BACKEND_API, AUTOTRAIN_TOKEN, AUTOTRAIN_USERNAME, COMPETITION_ID, competition_info
from .leaderboard import Leaderboard
from .submissions import Submissions
from .text import NO_SUBMISSIONS, SUBMISSION_SELECTION_TEXT, SUBMISSION_TEXT


leaderboard = Leaderboard(
    end_date=competition_info.end_date,
    eval_higher_is_better=competition_info.eval_higher_is_better,
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


def _my_submissions(user_token):
    df = submissions.my_submissions(user_token)
    if len(df) == 0:
        return [gr.Markdown.update(visible=True, value=NO_SUBMISSIONS), gr.DataFrame.update(visible=False)]
    return [gr.Markdown.update(visible=False), gr.DataFrame.update(visible=True, value=df)]


with gr.Blocks() as demo:
    with gr.Tabs() as tab_container:
        with gr.TabItem("Overview", id="overview"):
            gr.Markdown(f"# Welcome to {competition_info.competition_name}! 👋")
            gr.Markdown(f"{competition_info.competition_description}")
            gr.Markdown("## Dataset")
            gr.Markdown(f"{competition_info.dataset_description}")
        with gr.TabItem("Public Leaderboard", id="public_leaderboard") as public_leaderboard:
            output_df_public = gr.DataFrame()
        with gr.TabItem("Private Leaderboard", id="private_leaderboard") as private_leaderboard:
            current_date_time = datetime.now()
            if current_date_time > competition_info.end_date:
                output_df_private = gr.DataFrame()
            else:
                gr.Markdown("Private Leaderboard will be available after the competition ends")
        with gr.TabItem("New Submission", id="new_submission"):
            gr.Markdown(SUBMISSION_TEXT.format(competition_info.submission_limit))
            user_token = gr.Textbox(max_lines=1, value="hf_XXX", label="Please enter your Hugging Face token")
            uploaded_file = gr.File()
            output_text = gr.Markdown(visible=True, show_label=False)
            new_sub_button = gr.Button("Upload Submission")
            new_sub_button.click(
                fn=submissions.new_submission,
                inputs=[user_token, uploaded_file],
                outputs=[output_text],
            )
        with gr.TabItem("My Submissions", id="my_submissions"):
            gr.Markdown(SUBMISSION_SELECTION_TEXT.format(competition_info.selection_limit))
            user_token = gr.Textbox(max_lines=1, value="hf_XXX", label="Please enter your Hugging Face token")
            output_text = gr.Markdown(visible=True, show_label=False)
            output_df = gr.DataFrame(visible=False)
            my_subs_button = gr.Button("Fetch Submissions")
            my_subs_button.click(
                fn=_my_submissions,
                inputs=[user_token],
                outputs=[output_text, output_df],
            )

        fetch_lb_partial = partial(leaderboard.fetch, private=False)
        public_leaderboard.select(fetch_lb_partial, inputs=[], outputs=[output_df_public])
        if current_date_time > competition_info.end_date:
            fetch_lb_partial_private = partial(leaderboard.fetch, private=True)
            private_leaderboard.select(fetch_lb_partial_private, inputs=[], outputs=[output_df_private])
