import streamlit as st
import config


def app():
    st.set_page_config(page_title="Dataset", page_icon="📈")

    st.markdown("# Dataset")
    # st.sidebar.header("Dataset")
    st.write(f"""{config.competition_info.dataset_description}""")


if __name__ == "__main__":
    app()
