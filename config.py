import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from huggingface_hub.utils._errors import EntryNotFoundError

if Path(".env").is_file():
    load_dotenv(".env")


MOONLANDING_URL = os.getenv("MOONLANDING_URL")
COMPETITION_ID = os.getenv("COMPETITION_ID")
AUTOTRAIN_USERNAME = os.getenv("AUTOTRAIN_USERNAME")
AUTOTRAIN_TOKEN = os.getenv("AUTOTRAIN_TOKEN")
AUTOTRAIN_BACKEND_API = os.getenv("AUTOTRAIN_BACKEND_API")


class CompetitionInfo:
    def __init__(self):
        try:
            config_fname = hf_hub_download(
                repo_id=COMPETITION_ID,
                filename="conf",
                use_auth_token=AUTOTRAIN_TOKEN,
                repo_type="dataset",
            )
        except EntryNotFoundError:
            raise Exception("Competition config not found. Please check the competition id.")
        except Exception as e:
            print(e)
            raise Exception("Hugging Face Hub is unreachable, please try again later.")

        self.config = self.load_config(config_fname)

    def load_config(self, config_path):
        with open(config_path) as f:
            config = json.load(f)
        return config

    @property
    def submission_limit(self):
        return self.config["SUBMISSION_LIMIT"]

    @property
    def selection_limit(self):
        return self.config["SELECTION_LIMIT"]

    @property
    def end_date(self):
        e_d = self.config["END_DATE"]
        return datetime.strptime(e_d, "%Y-%m-%d")

    @property
    def eval_higher_is_better(self):
        hb = self.config["EVAL_HIGHER_IS_BETTER"]
        return True if int(hb) == 1 else False

    @property
    def competition_dataset(self):
        return self.config["DATASET"]

    @property
    def competition_description(self):
        return self.config["COMPETITION_DESCRIPTION"]

    @property
    def competition_name(self):
        return self.config["COMPETITION_NAME"]

    @property
    def submission_columns(self):
        return self.config["SUBMISSION_COLUMNS"].split(",")

    @property
    def dataset_description(self):
        return self.config["DATASET_DESCRIPTION"]


competition_info = CompetitionInfo()
