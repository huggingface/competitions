from datetime import datetime

import streamlit as st

import config
import utils


def app():
    st.set_page_config(page_title="Leaderboard", page_icon="🤗")
    st.markdown("## Leaderboard")
    public_lb, private_lb = st.tabs(["Public", "Private"])
    current_date_time = datetime.now()

    with public_lb:
        lb = utils.fetch_leaderboard(private=False)
        st.table(lb)

    with private_lb:
        if current_date_time >= config.competition_info.end_date:
            lb = utils.fetch_leaderboard(private=True)
            st.table(lb)
        else:
            st.error("Private Leaderboard will be available after the competition ends")


if __name__ == "__main__":
    app()
