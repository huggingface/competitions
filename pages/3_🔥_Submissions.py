import streamlit as st
from huggingface_hub import HfApi, CommitOperationAdd
import uuid
import os
import pandas as pd

SUBMISSION_TEXT = """You can make upto 5 submissions per day.
The test data has been divided into public and private splits.
Your score on the public split will be shown on the leaderboard.
Your final score will be based on your private split performance.
The final rankings will be based on the private split performance.
"""

SUBMISSION_ERROR = """Submission is not in a proper format.
Please check evaluation instructions for more details."""

COMPETITION_ID = os.getenv("COMPETITION_ID", "sample_competition")
USER_ID = os.getenv("USER_ID", "sample_user")
DUMMY_DATA_PATH = os.getenv("DUMMY_DATA_PATH", "autoevaluator/benchmark-dummy-data")


def verify_submission(submission):
    # verify submission is valid
    return True


def fetch_submissions():
    submissions = [
        {
            "submission_id": "72836-23423",
            "score": 0.7,
            "created_at": "2021-01-01T00:00:00Z",
        },
        {
            "submission_id": "23-42332",
            "score": 0.5,
            "created_at": "2021-01-01T00:00:00Z",
        },
    ]
    df = pd.DataFrame(submissions)
    return df


def app():
    st.set_page_config(page_title="Submissions", page_icon="🤗")
    st.write("## Submissions")
    uploaded_file = st.sidebar.file_uploader("Choose a file")
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        # verify file is valid
        if not verify_submission(bytes_data):
            st.error("Invalid submission")
            st.write(SUBMISSION_ERROR)
            # write a horizontal html line
            st.markdown("<hr/>", unsafe_allow_html=True)
        else:
            # start progress bar
            progress_bar = st.progress(0)
            submission_id = str(uuid.uuid4())
            api = HfApi()
            operations = [
                CommitOperationAdd(
                    path_in_repo="submission.csv",
                    path_or_fileobj=bytes_data,
                ),
            ]
            # update progress bar
            progress_bar.progress(0.5)
            api.create_repo(
                repo_id=submission_id,
                private=True,
                repo_type="dataset",
            )
            api.create_commit(
                repo_id=f"{USER_ID}/{submission_id}",
                operations=operations,
                commit_message="add submission.csv",
                repo_type="dataset",
            )

    st.markdown(SUBMISSION_TEXT)
    # add submissions history table
    st.write("### Submissions History")
    submissions = fetch_submissions()
    if len(submissions) == 0:
        st.write("You have not made any submissions yet.")
    else:
        st.write(submissions)


if __name__ == "__main__":
    app()
