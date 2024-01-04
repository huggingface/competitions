import os


__version__ = "0.1.1"

MOONLANDING_URL = os.getenv("MOONLANDING_URL", "https://huggingface.co")
COMPETITION_ID = os.getenv("COMPETITION_ID")
AUTOTRAIN_USERNAME = os.getenv("AUTOTRAIN_USERNAME")
AUTOTRAIN_TOKEN = os.getenv("AUTOTRAIN_TOKEN")
