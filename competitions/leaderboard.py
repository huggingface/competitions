import glob
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download
from loguru import logger


@dataclass
class Leaderboard:
    end_date: datetime
    eval_higher_is_better: bool
    max_selected_submissions: int
    competition_id: str
    token: str

    def __post_init__(self):
        self._refresh_columns()

    def _refresh_columns(self):
        self.private_columns = [
            "rank",
            "id",
            "private_score",
            "submission_datetime",
        ]
        self.public_columns = [
            "rank",
            "id",
            "public_score",
            "submission_datetime",
        ]

    def _process_public_lb(self):
        self._refresh_columns()
        start_time = time.time()
        submissions_folder = snapshot_download(
            repo_id=self.competition_id,
            allow_patterns="submission_info/*.json",
            use_auth_token=self.token,
            repo_type="dataset",
        )
        logger.info(f"Downloaded submissions in {time.time() - start_time} seconds")
        start_time = time.time()
        submissions = []
        for submission in glob.glob(os.path.join(submissions_folder, "submission_info", "*.json")):
            with open(submission, "r", encoding="utf-8") as f:
                submission_info = json.load(f)
            # only select submissions that are done
            submission_info["submissions"] = [
                sub for sub in submission_info["submissions"] if sub["status"] == "success"
            ]
            submission_info["submissions"] = [
                sub
                for sub in submission_info["submissions"]
                if datetime.strptime(sub["datetime"], "%Y-%m-%d %H:%M:%S") < self.end_date
            ]
            if len(submission_info["submissions"]) == 0:
                continue
            other_scores = []
            if isinstance(submission_info["submissions"][0]["public_score"], dict):
                # get keys of the dict
                score_keys = list(submission_info["submissions"][0]["public_score"].keys())
                # get the first key after sorting
                score_key = sorted(score_keys)[0]
                other_scores = [f"public_score_{k}" for k in score_keys if k != score_key]

                self.public_columns.extend(other_scores)
                for _sub in submission_info["submissions"]:
                    for skey in score_keys:
                        if skey != score_key:
                            _sub[f"public_score_{skey}"] = _sub["public_score"][skey]
                    _sub["public_score"] = _sub["public_score"][score_key]

            submission_info["submissions"].sort(key=lambda x: x["public_score"], reverse=self.eval_higher_is_better)
            # select only the best submission
            submission_info["submissions"] = submission_info["submissions"][0]
            temp_info = {
                "id": submission_info["id"],
                "submission_id": submission_info["submissions"]["submission_id"],
                "submission_comment": submission_info["submissions"]["submission_comment"],
                "status": submission_info["submissions"]["status"],
                "selected": submission_info["submissions"]["selected"],
                "public_score": submission_info["submissions"]["public_score"],
                # "private_score": submission_info["submissions"]["private_score"],
                "submission_datetime": submission_info["submissions"]["datetime"],
            }
            for score in other_scores:
                temp_info[score] = submission_info["submissions"][score]
            submissions.append(temp_info)
        logger.info(f"Processed submissions in {time.time() - start_time} seconds")
        return submissions

    def _process_private_lb(self):
        self._refresh_columns()
        start_time = time.time()
        submissions_folder = snapshot_download(
            repo_id=self.competition_id,
            allow_patterns="submission_info/*.json",
            use_auth_token=self.token,
            repo_type="dataset",
        )
        logger.info(f"Downloaded submissions in {time.time() - start_time} seconds")
        start_time = time.time()
        submissions = []
        for submission in glob.glob(os.path.join(submissions_folder, "submission_info", "*.json")):
            with open(submission, "r", encoding="utf-8") as f:
                submission_info = json.load(f)
                submission_info["submissions"] = [
                    sub for sub in submission_info["submissions"] if sub["status"] == "success"
                ]
                if len(submission_info["submissions"]) == 0:
                    continue
                other_scores = []
                if isinstance(submission_info["submissions"][0]["public_score"], dict):
                    # get keys of the dict
                    score_keys = list(submission_info["submissions"][0]["public_score"].keys())
                    # get the first key after sorting
                    score_key = sorted(score_keys)[0]
                    other_scores = [f"private_score_{k}" for k in score_keys if k != score_key]
                    self.private_columns.extend(other_scores)
                    for _sub in submission_info["submissions"]:
                        for skey in score_keys:
                            if skey != score_key:
                                _sub[f"public_score_{skey}"] = _sub["public_score"][skey]
                        _sub["public_score"] = _sub["public_score"][score_key]

                    for _sub in submission_info["submissions"]:
                        for skey in score_keys:
                            if skey != score_key:
                                _sub[f"private_score_{skey}"] = _sub["private_score"][skey]
                        _sub["private_score"] = _sub["private_score"][score_key]
                # count the number of submissions which are selected
                selected_submissions = 0
                for sub in submission_info["submissions"]:
                    if sub["selected"]:
                        selected_submissions += 1
                if selected_submissions == 0:
                    # select submissions with best public score
                    submission_info["submissions"].sort(
                        key=lambda x: x["public_score"], reverse=self.eval_higher_is_better
                    )
                    # select only the best submission
                    submission_info["submissions"] = submission_info["submissions"][0]
                elif selected_submissions == self.max_selected_submissions:
                    # select only the selected submissions
                    submission_info["submissions"] = [sub for sub in submission_info["submissions"] if sub["selected"]]
                    # sort by private score
                    submission_info["submissions"].sort(
                        key=lambda x: x["private_score"], reverse=self.eval_higher_is_better
                    )
                    # select only the best submission
                    submission_info["submissions"] = submission_info["submissions"][0]
                else:
                    temp_selected_submissions = [sub for sub in submission_info["submissions"] if sub["selected"]]
                    temp_best_public_submissions = [
                        sub for sub in submission_info["submissions"] if not sub["selected"]
                    ]
                    temp_best_public_submissions.sort(
                        key=lambda x: x["public_score"], reverse=self.eval_higher_is_better
                    )
                    missing_candidates = self.max_selected_submissions - len(temp_selected_submissions)
                    temp_best_public_submissions = temp_best_public_submissions[:missing_candidates]
                    submission_info["submissions"] = temp_selected_submissions + temp_best_public_submissions
                    submission_info["submissions"].sort(
                        key=lambda x: x["private_score"], reverse=self.eval_higher_is_better
                    )
                    submission_info["submissions"] = submission_info["submissions"][0]

                temp_info = {
                    "id": submission_info["id"],
                    "submission_id": submission_info["submissions"]["submission_id"],
                    "submission_comment": submission_info["submissions"]["submission_comment"],
                    "status": submission_info["submissions"]["status"],
                    "selected": submission_info["submissions"]["selected"],
                    "private_score": submission_info["submissions"]["private_score"],
                    "submission_datetime": submission_info["submissions"]["datetime"],
                }
                for score in other_scores:
                    temp_info[score] = submission_info["submissions"][score]
                submissions.append(temp_info)
        logger.info(f"Processed submissions in {time.time() - start_time} seconds")
        return submissions

    def fetch(self, private=False):
        if private:
            submissions = self._process_private_lb()
        else:
            submissions = self._process_public_lb()

        if len(submissions) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(submissions)

        # convert submission datetime to pandas datetime
        df["submission_datetime"] = pd.to_datetime(df["submission_datetime"], format="%Y-%m-%d %H:%M:%S")

        # only keep submissions before the end date
        df = df[df["submission_datetime"] < self.end_date].reset_index(drop=True)

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

        # convert datetime column to string
        df["submission_datetime"] = df["submission_datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(df)
        columns = self.public_columns if not private else self.private_columns
        logger.info(columns)
        # remove duplicate columns
        columns = list(dict.fromkeys(columns))

        # send submission_datetime to the end
        columns.remove("submission_datetime")
        columns.append("submission_datetime")

        team_metadata = hf_hub_download(
            repo_id=self.competition_id,
            filename="teams.json",
            token=self.token,
            repo_type="dataset",
        )

        with open(team_metadata, "r", encoding="utf-8") as f:
            team_metadata = json.load(f)

        df["id"] = df["id"].apply(lambda x: team_metadata[x]["name"])

        return df[columns]
