import streamlit as st
from huggingface_hub import HfApi, CommitOperationAdd
import uuid
import os
import pandas as pd
import utils
import config

SUBMISSION_TEXT = """You can select upto 2 submissions for private leaderboard.
"""


def app():
    st.set_page_config(page_title="Submission History", page_icon="🤗")
    st.write("## Your Submissions")
    st.markdown(SUBMISSION_TEXT)
    # user token
    user_token = st.text_input("Enter your token", value="", type="password")
    user_token = user_token.strip()
    if user_token != "":
        user_info = utils.user_authentication(token=user_token)
        if "error" in user_info:
            st.error("Invalid token")
            return

        if user_info["emailVerified"] is False:
            st.error("Please verify your email on Hugging Face Hub")
            return

        # get user submissions
        user_id = user_info["id"]
        user_submissions = utils.fetch_submissions(user_id)
        submissions_df = pd.DataFrame(user_submissions)
        st.write(submissions_df)


if __name__ == "__main__":
    app()
