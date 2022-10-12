import streamlit as st


SUBMISSION_TEXT = """You can make upto 5 submissions per day.
The test data has been divided into public and private splits.
Your score on the public split will be shown on the leaderboard.
Your final score will be based on your private split performance.
The final rankings will be based on the private split performance.
"""


def app():
    st.set_page_config(page_title="Submissions", page_icon="🤗")
    st.write("## Submissions")
    st.markdown(SUBMISSION_TEXT)
    # add submissions history table
    st.write("### Submissions History")
    st.write("You have not made any submissions yet.")


if __name__ == "__main__":
    app()
