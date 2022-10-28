from datetime import datetime

import pandas as pd
import streamlit as st

import config


def fetch_leaderboard(private=False):
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
