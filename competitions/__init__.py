import os

from .info import CompetitionInfo


__version__ = "0.1.1"

MOONLANDING_URL = os.getenv("MOONLANDING_URL", "https://huggingface.co")
COMPETITION_ID = os.getenv("COMPETITION_ID")
AUTOTRAIN_USERNAME = os.getenv("AUTOTRAIN_USERNAME")
AUTOTRAIN_TOKEN = os.getenv("AUTOTRAIN_TOKEN")
AUTOTRAIN_BACKEND_API = os.getenv("AUTOTRAIN_BACKEND_API", "https://api.autotrain.huggingface.co")
BOT_TOKEN = os.getenv("BOT_TOKEN")

competition_info = CompetitionInfo(competition_id=COMPETITION_ID, autotrain_token=AUTOTRAIN_TOKEN)
