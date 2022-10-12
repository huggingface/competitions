import streamlit as st
import pandas as pd


def fetch_leaderboard():
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
    return pd.DataFrame(data_dict)


def app():
    st.set_page_config(page_title="Leaderboard", page_icon="🤗")
    st.markdown("## Leaderboard")
    lb = fetch_leaderboard()
    st.table(lb)


if __name__ == "__main__":
    app()
