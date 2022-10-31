from datetime import datetime

import streamlit as st

import config
import utils


def app():
    st.set_page_config(page_title="Private Leaderboard", page_icon="🤗")
    st.markdown("## Private Leaderboard")
    current_date_time = datetime.now()

    if current_date_time >= config.END_DATE:
        st.header("Private Leaderboard")
        lb = utils.fetch_leaderboard(private=True)
        st.table(lb)
    else:
        st.error("Private Leaderboard will be available after the competition ends")


if __name__ == "__main__":
    app()
