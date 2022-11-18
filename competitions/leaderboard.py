import glob
import json
import os
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from huggingface_hub import snapshot_download


@dataclass
class Leaderboard:
    end_date: datetime
    eval_higher_is_better: bool
    competition_id: str
    autotrain_token: str

    def __post_init__(self):
        self.private_columns = [
            "rank",
            "name",
            "private_score",
            "submission_datetime",
        ]
        self.public_columns = [
            "rank",
            "name",
            "public_score",
            "submission_datetime",
        ]

    def _download_submissions(self, private):
        submissions_folder = snapshot_download(
            repo_id=self.competition_id,
            allow_patterns="*.json",
            use_auth_token=self.autotrain_token,
            repo_type="dataset",
        )
        submissions = []
        for submission in glob.glob(os.path.join(submissions_folder, "*.json")):
            with open(submission, "r") as f:
                submission_info = json.load(f)
            if self.eval_higher_is_better:
                submission_info["submissions"].sort(
                    key=lambda x: x["private_score"] if private else x["public_score"],
                    reverse=True,
                )
            else:
                submission_info["submissions"].sort(key=lambda x: x["private_score"] if private else x["public_score"])

            # select only the best submission
            submission_info["submissions"] = submission_info["submissions"][0]
            temp_info = {
                "id": submission_info["id"],
                "name": submission_info["name"],
                "submission_id": submission_info["submissions"]["submission_id"],
                "submission_comment": submission_info["submissions"]["submission_comment"],
                "status": submission_info["submissions"]["status"],
                "selected": submission_info["submissions"]["selected"],
                "public_score": submission_info["submissions"]["public_score"],
                "private_score": submission_info["submissions"]["private_score"],
                "submission_date": submission_info["submissions"]["date"],
                "submission_time": submission_info["submissions"]["time"],
            }
            submissions.append(temp_info)
        return submissions

    def fetch(self, private=False):
        submissions = self._download_submissions(private)

        if len(submissions) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(submissions)
        # convert submission date and time to datetime
        df["submission_datetime"] = pd.to_datetime(
            df["submission_date"] + " " + df["submission_time"], format="%Y-%m-%d %H:%M:%S"
        )
        # convert datetime column to string
        df["submission_datetime"] = df["submission_datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

        # sort by submission datetime
        # sort by public score and submission datetime
        if self.eval_higher_is_better:
            df = df.sort_values(
                by=["public_score", "submission_datetime"],
                ascending=[False, True],
            )
        else:
            df = df.sort_values(
                by=["public_score", "submission_datetime"],
                ascending=[True, True],
            )

        # reset index
        df = df.reset_index(drop=True)
        df["rank"] = df.index + 1

        columns = self.public_columns if not private else self.private_columns
        return df[columns]
