import glob
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download
from loguru import logger

from competitions.enums import SubmissionStatus


@dataclass
class Leaderboard:
    end_date: datetime
    eval_higher_is_better: bool
    max_selected_submissions: int
    competition_id: str
    token: str
    scoring_metric: str

    def __post_init__(self):
        self.non_score_columns = ["id", "submission_datetime"]

    def _process_public_lb(self):
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
                sub for sub in submission_info["submissions"] if sub["status"] == SubmissionStatus.SUCCESS.value
            ]
            submission_info["submissions"] = [
                sub
                for sub in submission_info["submissions"]
                if datetime.strptime(sub["datetime"], "%Y-%m-%d %H:%M:%S") < self.end_date
            ]
            if len(submission_info["submissions"]) == 0:
                continue

            user_id = submission_info["id"]
            user_submissions = []
            for sub in submission_info["submissions"]:
                _sub = {
                    "id": user_id,
                    # "submission_id": sub["submission_id"],
                    # "submission_comment": sub["submission_comment"],
                    # "status": sub["status"],
                    # "selected": sub["selected"],
                }
                for k, v in sub["public_score"].items():
                    _sub[k] = v
                _sub["submission_datetime"] = sub["datetime"]
                user_submissions.append(_sub)

            user_submissions.sort(key=lambda x: x[self.scoring_metric], reverse=self.eval_higher_is_better)
            best_user_submission = user_submissions[0]
            submissions.append(best_user_submission)
        logger.info(f"Processed submissions in {time.time() - start_time} seconds")
        return submissions

    def _process_private_lb(self):
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
                    sub for sub in submission_info["submissions"] if sub["status"] == SubmissionStatus.SUCCESS.value
                ]
                if len(submission_info["submissions"]) == 0:
                    continue

                user_id = submission_info["id"]
                user_submissions = []
                for sub in submission_info["submissions"]:
                    _sub = {
                        "id": user_id,
                        # "submission_id": sub["submission_id"],
                        # "submission_comment": sub["submission_comment"],
                        # "status": sub["status"],
                        "selected": sub["selected"],
                    }
                    for k, v in sub["public_score"].items():
                        _sub[f"public_{k}"] = v
                    for k, v in sub["private_score"].items():
                        _sub[f"private_{k}"] = v
                    _sub["submission_datetime"] = sub["datetime"]
                    user_submissions.append(_sub)

                # count the number of submissions which are selected
                selected_submissions = 0
                for sub in user_submissions:
                    if sub["selected"]:
                        selected_submissions += 1

                if selected_submissions == 0:
                    # select submissions with best public score
                    user_submissions.sort(
                        key=lambda x: x[f"public_{self.scoring_metric}"], reverse=self.eval_higher_is_better
                    )
                    # select only the best submission
                    best_user_submission = user_submissions[0]

                elif selected_submissions <= self.max_selected_submissions:
                    # select only the selected submissions
                    user_submissions = [sub for sub in user_submissions if sub["selected"]]
                    # sort by private score
                    user_submissions.sort(
                        key=lambda x: x[f"private_{self.scoring_metric}"], reverse=self.eval_higher_is_better
                    )
                    # select only the best submission
                    best_user_submission = user_submissions[0]
                else:
                    logger.warning(
                        f"User {user_id} has more than {self.max_selected_submissions} selected submissions. Skipping user..."
                    )
                    continue

                # remove all keys that start with "public_"
                best_user_submission = {k: v for k, v in best_user_submission.items() if not k.startswith("public_")}

                # remove private_ from the keys
                best_user_submission = {k.replace("private_", ""): v for k, v in best_user_submission.items()}

                # remove selected key
                best_user_submission.pop("selected")
                submissions.append(best_user_submission)
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
                    by=[self.scoring_metric, "submission_datetime"],
                    ascending=[False, True],
                )
            else:
                df = df.sort_values(
                    by=[self.scoring_metric, "submission_datetime"],
                    ascending=[False, True],
                )
        else:
            if private:
                df = df.sort_values(
                    by=[self.scoring_metric, "submission_datetime"],
                    ascending=[True, True],
                )
            else:
                df = df.sort_values(
                    by=[self.scoring_metric, "submission_datetime"],
                    ascending=[True, True],
                )

        # only keep 4 significant digits in the scores
        for col in df.columns:
            if col in self.non_score_columns:
                continue
            df[col] = df[col].round(4)

        # reset index
        df = df.reset_index(drop=True)
        df["rank"] = df.index + 1

        # convert datetime column to string
        df["submission_datetime"] = df["submission_datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

        # send submission_datetime to the end
        columns = df.columns.tolist()
        columns.remove("submission_datetime")
        columns.append("submission_datetime")
        df = df[columns]

        # send rank to first position
        columns = df.columns.tolist()
        columns.remove("rank")
        columns = ["rank"] + columns
        df = df[columns]

        team_metadata = hf_hub_download(
            repo_id=self.competition_id,
            filename="teams.json",
            token=self.token,
            repo_type="dataset",
        )
        with open(team_metadata, "r", encoding="utf-8") as f:
            team_metadata = json.load(f)

        df["id"] = df["id"].apply(lambda x: team_metadata[x]["name"])

        return df
