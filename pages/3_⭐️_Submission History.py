from datetime import datetime

import pandas as pd
import streamlit as st
from huggingface_hub.utils._errors import EntryNotFoundError

import config
import utils

SUBMISSION_TEXT = f"""You can select upto {config.competition_info.selection_limit}
 submissions for private leaderboard."""


def get_subs(user_info, private=False):
    # get user submissions
    user_id = user_info["id"]
    try:
        user_submissions = utils.fetch_submissions(user_id)
    except EntryNotFoundError:
        st.error("No submissions found")
        return
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
    st.write(submissions_df)


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

        current_date_time = datetime.now()
        private = False
        if current_date_time >= config.END_DATE:
            private = True
        get_subs(user_info, private=private)


if __name__ == "__main__":
    app()
