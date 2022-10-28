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
        print(config.EVAL_HIGHER_IS_BETTER)
        if config.EVAL_HIGHER_IS_BETTER:
            submission_info["submissions"].sort(
                key=lambda x: x["private_score"] if private else x["public_score"], reverse=True
            )
        else:
            submission_info["submissions"].sort(key=lambda x: x["private_score"] if private else x["public_score"])
        # select only the best submission
        submission_info["submissions"] = submission_info["submissions"][0]
        temp_info = {
            "id": submission_info["id"],
            "name": submission_info["name"],
            "submission_id": submission_info["submissions"]["submission_id"],
            "submission_comment": submission_info["submissions"]["submission_comment"],
            "status": submission_info["submissions"]["status"],
            "selected": submission_info["submissions"]["selected"],
            "public_score": submission_info["submissions"]["public_score"],
            "private_score": submission_info["submissions"]["private_score"],
            "submission_date": submission_info["submissions"]["date"],
        }
        submissions.append(temp_info)
    print(submissions)

    df = pd.DataFrame(submissions)
    # sort by public score and then by submission_date
    df = df.sort_values(
        by=["public_score", "submission_date"],
        ascending=True if not config.EVAL_HIGHER_IS_BETTER else False,
    )
    df = df.reset_index(drop=True)
    df["rank"] = df.index + 1

    if private:
        columns = ["rank", "name", "private_score", "submission_date"]
    else:
        columns = ["rank", "name", "public_score", "submission_date"]
    st.table(df[columns])


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
