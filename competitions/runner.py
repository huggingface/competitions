import glob
import json
import os
import time
from dataclasses import dataclass

import pandas as pd
from huggingface_hub import snapshot_download
from loguru import logger

from competitions.info import CompetitionInfo
from competitions.utils import run_evaluation


@dataclass
class JobRunner:
    competition_info: CompetitionInfo
    token: str
    output_path: str

    def __post_init__(self):
        self.competition_id = self.competition_info.competition_id
        self.competition_type = self.competition_info.competition_type
        self.metric = self.competition_info.metric
        self.submission_id_col = self.competition_info.submission_id_col
        self.submission_cols = self.competition_info.submission_cols
        self.submission_rows = self.competition_info.submission_rows

    def get_pending_subs(self):
        user_jsons = snapshot_download(
            repo_id=self.competition_id,
            allow_patterns="submission_info/*.json",
            token=self.token,
            repo_type="dataset",
        )
        user_jsons = glob.glob(os.path.join(user_jsons, "submission_info/*.json"))
        pending_submissions = []
        for _json in user_jsons:
            _json = json.load(open(_json, "r", encoding="utf-8"))
            user_id = _json["id"]
            for sub in _json["submissions"]:
                # if sub["status"] == "pending":
                pending_submissions.append(
                    {
                        "user_id": user_id,
                        "submission_id": sub["submission_id"],
                        "date": sub["date"],
                        "time": sub["time"],
                    }
                )
        if len(pending_submissions) == 0:
            logger.info("No pending submissions.")
            return None
        logger.info(f"Found {len(pending_submissions)} pending submissions.")
        pending_submissions = pd.DataFrame(pending_submissions)
        pending_submissions = pending_submissions.sort_values(by=["date", "time"])
        pending_submissions = pending_submissions.reset_index(drop=True)
        return pending_submissions

    def run_local(self, pending_submissions):
        for _, row in pending_submissions.iterrows():
            user_id = row["user_id"]
            submission_id = row["submission_id"]
            eval_params = {
                "competition_id": self.competition_id,
                "competition_type": self.competition_type,
                "metric": self.metric,
                "token": self.token,
                "user_id": user_id,
                "submission_id": submission_id,
                "submission_id_col": self.submission_id_col,
                "submission_cols": self.submission_cols,
                "submission_rows": self.submission_rows,
                "output_path": self.output_path,
            }
            eval_params = json.dumps(eval_params)
            eval_pid = run_evaluation(eval_params, local=True, wait=True)
            logger.info(f"New evaluation process started with pid {eval_pid}.")

    def run(self):
        while True:
            pending_submissions = self.get_pending_subs()
            if pending_submissions is None:
                time.sleep(5)
                continue
            if self.competition_type == "generic":
                self.run_local(pending_submissions)
            time.sleep(5)
