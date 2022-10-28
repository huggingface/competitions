import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

if Path(".env").is_file():
    load_dotenv(".env")


MOONLANDING_URL = os.getenv("MOONLANDING_URL")
COMPETITION_ID = os.getenv("COMPETITION_ID")
DUMMY_DATA_PATH = os.getenv("DUMMY_DATA_PATH")
AUTOTRAIN_USERNAME = os.getenv("AUTOTRAIN_USERNAME")
AUTOTRAIN_TOKEN = os.getenv("AUTOTRAIN_TOKEN")
HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")
AUTOTRAIN_BACKEND_API = os.getenv("AUTOTRAIN_BACKEND_API")
SUBMISSION_LIMIT = int(os.getenv("SUBMISSION_LIMIT"))
SELECTION_LIMIT = int(os.getenv("SELECTION_LIMIT"))
END_DATE = os.getenv("END_DATE")
END_DATE = datetime.strptime(END_DATE, "%Y-%m-%d")
EVAL_HIGHER_IS_BETTER = True if int(os.getenv("EVAL_HIGHER_IS_BETTER")) == 1 else False
