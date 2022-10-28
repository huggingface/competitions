from datetime import datetime

import pandas as pd
import streamlit as st

import config
import utils

SUBMISSION_TEXT = """You can select upto 2 submissions for private leaderboard.
"""


def get_subs(user_info, private=False):
    # get user submissions
    user_id = user_info["id"]
    user_submissions = utils.fetch_submissions(user_id)
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
    return submissions_df


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

        public_lb, private_lb = st.tabs(["Public", "Private"])
        current_date_time = datetime.now()
        print(current_date_time)
        print(config.END_DATE)
        print(current_date_time >= config.END_DATE)
        with public_lb:
            st.header("Public Leaderboard")
            submission_df = get_subs(user_info, private=False)
            st.write(submission_df)

        with private_lb:
            if current_date_time >= config.END_DATE:
                st.header("Private Leaderboard")
                submission_df = get_subs(user_info, private=True)
                st.write(submission_df)
            else:
                st.error("Private Leaderboard will be available after the competition ends")


if __name__ == "__main__":
    app()
