import streamlit as st

st.set_page_config(
    page_title="Overview",
    page_icon="🤗",
)

st.write("# Welcome to Cats vs Dogs! 👋")

st.markdown(
    """
    In this competition, you will be creating an image classifier that can distinguish between cats and dogs.
    You can use any model you want, but we recommend using a pretrained model from the [🤗 Hub](https://huggingface.co/models).

    Rules:
    - You can make upto 5 submissions per day.
    - The test data has been divided into public and private splits.
    - Your score on the public split will be shown on the leaderboard.
    - Your final score will be based on your private split performance.
    - The final rankings will be based on the private split performance.
    - No cheating! You can only use the test data for inference. You cannot use it to train your model.
    - No private sharing of code or data. You can discuss the competition on the forums, but please do not share any code or data.
    - Have fun! 🤗

    For detailed rules, please check the [rules page](https://huggingface.co/competitions/transformers-cats-vs-dogs).
"""
)
