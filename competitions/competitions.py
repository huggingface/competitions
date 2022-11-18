import gradio as gr

from . import competition_info
from .leaderboard import Leaderboard
from .submissions import Submissions


# with gr.Blocks() as demo:
#     with gr.Tab("Overview"):
#         gr.Markdown(f"# Welcome to {config.competition_info.competition_name}! 👋")

#         gr.Markdown(f"{config.competition_info.competition_description}")

#         gr.Markdown("## Dataset")
#         gr.Markdown(f"{config.competition_info.dataset_description}")

#     with gr.Tab("Public Leaderboard"):
#         output_markdown = gr.Markdown("")
#         fetch_lb = gr.Button("Fetch Leaderboard")
#         fetch_lb_partial = partial(utils.fetch_leaderboard, private=False)
#         fetch_lb.click(fn=fetch_lb_partial, outputs=[output_markdown])
#         # lb = utils.fetch_leaderboard(private=False)
#         # gr.Markdown(lb.to_markdown())
#     with gr.Tab("Private Leaderboard"):
#         current_date_time = datetime.now()
#         if current_date_time >= config.competition_info.end_date:
#             lb = utils.fetch_leaderboard(private=True)
#             gr.Markdown(lb)
#         else:
#             gr.Markdown("Private Leaderboard will be available after the competition ends")
#     with gr.Tab("New Submission"):
#         gr.Markdown(SUBMISSION_TEXT)
#         user_token = gr.Textbox(max_lines=1, value="hf_XXX", label="Please enter your Hugging Face token")
#         uploaded_file = gr.File()
#         output_text = gr.Markdown(visible=True, show_label=False)
#         new_sub_button = gr.Button("Upload Submission")
#         new_sub_button.click(fn=new_submission, inputs=[user_token, uploaded_file], outputs=[output_text])

#     with gr.Tab("My Submissions"):
#         gr.Markdown(SUBMISSION_LIMIT_TEXT)
#         user_token = gr.Textbox(max_lines=1, value="hf_XXX", label="Please enter your Hugging Face token")
#         output_text = gr.Markdown(visible=True, show_label=False)
#         output_df = gr.Dataframe(visible=False)
#         my_subs_button = gr.Button("Fetch Submissions")
#         my_subs_button.click(fn=my_submissions, inputs=[user_token], outputs=[output_text, output_df])

with gr.Blocks() as demo:
    with gr.Tabs() as tab_container:
        with gr.TabItem("Overview", id="overview"):
            gr.Markdown(f"# Welcome to {competition_info.competition_name}! 👋")
            gr.Markdown(f"{competition_info.competition_description}")
            gr.Markdown("## Dataset")
            gr.Markdown(f"{competition_info.dataset_description}")
        with gr.TabItem("Public Leaderboard", id="public_leaderboard"):
            pass
        with gr.TabItem("Private Leaderboard", id="private_leaderboard"):
            pass
        with gr.TabItem("New Submission", id="new_submission"):
            pass
        with gr.TabItem("My Submissions", id="my_submissions"):
            pass
