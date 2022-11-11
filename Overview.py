import streamlit as st

import config

st.set_page_config(
    page_title="Overview",
    page_icon="🤗",
)

st.write(f"# Welcome to {config.competition_info.competition_name}! 👋")

st.markdown(
    f"""
    {config.competition_info.competition_description}
"""
)

st.markdown("## Dataset")
# st.sidebar.header("Dataset")
st.write(f"""{config.competition_info.dataset_description}""")
