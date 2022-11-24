import glob
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from loguru import logger

# from huggingface_hub import snapshot_download
from .download import snapshot_download


@dataclass
class Leaderboard:
    end_date: datetime
    eval_higher_is_better: bool
    max_selected_submissions: int
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

    def _process_public_lb(self):
        start_time = time.time()
        submissions_folder = snapshot_download(
            repo_id=self.competition_id,
            allow_patterns="*.json",
            use_auth_token=self.autotrain_token,
            repo_type="dataset",
        )
        logger.info(f"Downloaded submissions in {time.time() - start_time} seconds")
        start_time = time.time()
        submissions = []
        for submission in glob.glob(os.path.join(submissions_folder, "*.json")):
            with open(submission, "r") as f:
                submission_info = json.load(f)
            submission_info["submissions"].sort(
                key=lambda x: x["public_score"],
                reverse=True if self.eval_higher_is_better else False,
            )
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
                # "private_score": submission_info["submissions"]["private_score"],
                "submission_date": submission_info["submissions"]["date"],
                "submission_time": submission_info["submissions"]["time"],
            }
            submissions.append(temp_info)
        logger.info(f"Processed submissions in {time.time() - start_time} seconds")
        return submissions

    def _process_private_lb(self):
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
                # count the number of submissions which are selected
                selected_submissions = 0
                for sub in submission_info["submissions"]:
                    if sub["selected"]:
                        selected_submissions += 1
                if selected_submissions == 0:
                    # select submissions with best public score
                    submission_info["submissions"].sort(
                        key=lambda x: x["public_score"],
                        reverse=True if self.eval_higher_is_better else False,
                    )
                    # select only the best submission
                    submission_info["submissions"] = submission_info["submissions"][0]
                elif selected_submissions == self.max_selected_submissions:
                    # select only the selected submissions
                    submission_info["submissions"] = [sub for sub in submission_info["submissions"] if sub["selected"]]
                    # sort by private score
                    submission_info["submissions"].sort(
                        key=lambda x: x["private_score"],
                        reverse=True if self.eval_higher_is_better else False,
                    )
                    # select only the best submission
                    submission_info["submissions"] = submission_info["submissions"][0]
                else:
                    temp_selected_submissions = [sub for sub in submission_info["submissions"] if sub["selected"]]
                    temp_best_public_submissions = [
                        sub for sub in submission_info["submissions"] if not sub["selected"]
                    ]
                    temp_best_public_submissions.sort(
                        key=lambda x: x["public_score"],
                        reverse=True if self.eval_higher_is_better else False,
                    )
                    missing_candidates = self.max_selected_submissions - temp_selected_submissions
                    temp_best_public_submissions = temp_best_public_submissions[:missing_candidates]
                    submission_info["submissions"] = temp_selected_submissions + temp_best_public_submissions
                    submission_info["submissions"].sort(
                        key=lambda x: x["private_score"],
                        reverse=True if self.eval_higher_is_better else False,
                    )
                    submission_info["submissions"] = submission_info["submissions"][0]

                temp_info = {
                    "id": submission_info["id"],
                    "name": submission_info["name"],
                    "submission_id": submission_info["submissions"]["submission_id"],
                    "submission_comment": submission_info["submissions"]["submission_comment"],
                    "status": submission_info["submissions"]["status"],
                    "selected": submission_info["submissions"]["selected"],
                    "private_score": submission_info["submissions"]["private_score"],
                    "submission_date": submission_info["submissions"]["date"],
                    "submission_time": submission_info["submissions"]["time"],
                }
                submissions.append(temp_info)
        return submissions

    def fetch(self, private=False):
        if private:
            submissions = self._process_private_lb()
        else:
            submissions = self._process_public_lb()

        if len(submissions) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(submissions)
        # convert submission date and time to datetime
        df["submission_datetime"] = pd.to_datetime(
            df["submission_date"] + " " + df["submission_time"], format="%Y-%m-%d %H:%M:%S"
        )
        # only keep submissions before or on the end date
        df = df[df["submission_datetime"] <= self.end_date].reset_index(drop=True)

        # convert datetime column to string
        df["submission_datetime"] = df["submission_datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

        # sort by submission datetime
        # sort by public score and submission datetime
        if self.eval_higher_is_better:
            if private:
                df = df.sort_values(
                    by=["private_score", "submission_datetime"],
                    ascending=[False, True],
                )
            else:
                df = df.sort_values(
                    by=["public_score", "submission_datetime"],
                    ascending=[False, True],
                )
        else:
            if private:
                df = df.sort_values(
                    by=["private_score", "submission_datetime"],
                    ascending=[True, True],
                )
            else:
                df = df.sort_values(
                    by=["public_score", "submission_datetime"],
                    ascending=[True, True],
                )

        # only keep 4 significant digits in the score
        if private:
            df["private_score"] = df["private_score"].apply(lambda x: round(x, 4))
        else:
            df["public_score"] = df["public_score"].apply(lambda x: round(x, 4))

        # reset index
        df = df.reset_index(drop=True)
        df["rank"] = df.index + 1

        columns = self.public_columns if not private else self.private_columns
        return df[columns]
