import os
from dotenv import load_dotenv
from pathlib import Path

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
