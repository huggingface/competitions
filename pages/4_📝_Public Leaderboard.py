import streamlit as st

import utils


def app():
    st.set_page_config(page_title="Public Leaderboard", page_icon="🤗")
    st.markdown("## Public Leaderboard")
    lb = utils.fetch_leaderboard(private=False)
    st.table(lb)


if __name__ == "__main__":
    app()
