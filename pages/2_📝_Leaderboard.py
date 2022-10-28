import glob
import json
import os
from datetime import datetime

import pandas as pd
import streamlit as st
from huggingface_hub import snapshot_download

import config


def fetch_leaderboard(private=False):
    submissions_folder = snapshot_download(
        repo_id=config.COMPETITION_ID,
        allow_patterns="*.json",
        use_auth_token=config.AUTOTRAIN_TOKEN,
        repo_type="dataset",
    )
    submissions = []
    for submission in glob.glob(os.path.join(submissions_folder, "*.json")):
        with open(submission, "r") as f:
            submission_info = json.load(f)
        submissions.append(submission_info)

    print(submissions)

    data_dict = {
        "Team Name": ["Team 1", "Team 2", "Team 3", "Team 4", "Team 5"],
        "Score": [0.9, 0.8, 0.7, 0.6, 0.5],
        "Rank": [1, 2, 3, 4, 5],
        "Submission Time": [
            "2021-01-01 00:00:00",
            "2021-01-01 00:00:00",
            "2021-01-01 00:00:00",
            "2021-01-01 00:00:00",
            "2021-01-01 00:00:00",
        ],
    }
    lb = pd.DataFrame(data_dict)
    st.table(lb)


def app():
    st.set_page_config(page_title="Leaderboard", page_icon="🤗")
    st.markdown("## Leaderboard")
    public_lb, private_lb = st.tabs(["Public", "Private"])
    current_date_time = datetime.now()

    with public_lb:
        st.header("Public Leaderboard")
        fetch_leaderboard(private=False)

    with private_lb:
        if current_date_time >= config.END_DATE:
            st.header("Private Leaderboard")
            fetch_leaderboard(private=True)
        else:
            st.error("Private Leaderboard will be available after the competition ends")


if __name__ == "__main__":
    app()
