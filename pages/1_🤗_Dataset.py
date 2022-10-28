import streamlit as st


def app():
    st.set_page_config(page_title="Dataset", page_icon="📈")

    st.markdown("# Dataset")
    # st.sidebar.header("Dataset")
    st.write(
        """The dataset used for this competition can be found here: https://huggingface.co/datasets/cats_vs_dogs"""
    )


if __name__ == "__main__":
    app()
