import glob
import io
import json
import os
import random
import string
import time
from dataclasses import dataclass

import pandas as pd
from huggingface_hub import HfApi, snapshot_download
from loguru import logger

from competitions.info import CompetitionInfo
from competitions.utils import run_evaluation


_DOCKERFILE = """
FROM huggingface/competitions:latest

CMD uvicorn competitions.app:app --port 7860 --host 0.0.0.0
"""

# format _DOCKERFILE
_DOCKERFILE = _DOCKERFILE.replace("\n", " ").replace("  ", "\n").strip()


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
        submission_jsons = snapshot_download(
            repo_id=self.competition_id,
            allow_patterns="submission_info/*.json",
            token=self.token,
            repo_type="dataset",
        )
        submission_jsons = glob.glob(os.path.join(submission_jsons, "submission_info/*.json"))
        pending_submissions = []
        for _json in submission_jsons:
            _json = json.load(open(_json, "r", encoding="utf-8"))
            team_id = _json["id"]
            for sub in _json["submissions"]:
                if sub["status"] == "pending":
                    pending_submissions.append(
                        {
                            "team_id": team_id,
                            "submission_id": sub["submission_id"],
                            "datetime": sub["datetime"],
                            "submission_repo": sub["submission_repo"],
                        }
                    )
        if len(pending_submissions) == 0:
            logger.info("No pending submissions.")
            return None
        logger.info(f"Found {len(pending_submissions)} pending submissions.")
        pending_submissions = pd.DataFrame(pending_submissions)
        pending_submissions["datetime"] = pd.to_datetime(pending_submissions["datetime"])
        pending_submissions = pending_submissions.sort_values("datetime")
        pending_submissions = pending_submissions.reset_index(drop=True)
        return pending_submissions

    def run_local(self, pending_submissions):
        for _, row in pending_submissions.iterrows():
            team_id = row["team_id"]
            submission_id = row["submission_id"]
            eval_params = {
                "competition_id": self.competition_id,
                "competition_type": self.competition_type,
                "metric": self.metric,
                "token": self.token,
                "team_id": team_id,
                "submission_id": submission_id,
                "submission_id_col": self.submission_id_col,
                "submission_cols": self.submission_cols,
                "submission_rows": self.submission_rows,
                "output_path": self.output_path,
                "submission_repo": row["submission_repo"],
            }
            eval_params = json.dumps(eval_params)
            eval_pid = run_evaluation(eval_params, local=True, wait=True)
            logger.info(f"New evaluation process started with pid {eval_pid}.")

    def _create_readme(self, project_name):
        _readme = "---\n"
        _readme += f"title: {project_name}\n"
        _readme += "emoji: 🚀\n"
        _readme += "colorFrom: green\n"
        _readme += "colorTo: indigo\n"
        _readme += "sdk: docker\n"
        _readme += "pinned: false\n"
        _readme += "duplicated_from: autotrain-projects/autotrain-advanced\n"
        _readme += "---\n"
        _readme = io.BytesIO(_readme.encode())
        return _readme

    def create_space(self, team_id, submission_id, submission_repo):
        project_name = "".join(
            random.choices(
                string.ascii_lowercase + string.digits,
                k=10,
            )
        )
        api = HfApi(token=self.token)
        username = self.competition_id.split("/")[0]
        repo_id = f"{username}/competitions-{project_name}"
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            space_hardware="cpu-basic",
            private=True,
        )
        params = {
            "competition_id": self.competition_id,
            "competition_type": self.competition_type,
            "metric": self.metric,
            "token": self.token,
            "team_id": team_id,
            "submission_id": submission_id,
            "submission_id_col": self.submission_id_col,
            "submission_cols": self.submission_cols,
            "submission_rows": self.submission_rows,
            "output_path": self.output_path,
            "submission_repo": submission_repo,
        }

        api.add_space_secret(repo_id=repo_id, key="PARAMS", value=json.dumps(params))

        readme = self._create_readme(project_name)
        api.upload_file(
            path_or_fileobj=readme,
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="space",
        )

        _dockerfile = io.BytesIO(_DOCKERFILE.encode())
        api.upload_file(
            path_or_fileobj=_dockerfile,
            path_in_repo="Dockerfile",
            repo_id=repo_id,
            repo_type="space",
        )
        return repo_id

    def run(self):
        while True:
            pending_submissions = self.get_pending_subs()
            if pending_submissions is None:
                time.sleep(5)
                continue
            if self.competition_type == "generic":
                self.run_local(pending_submissions)
            elif self.competition_type == "code":
                for _, row in pending_submissions.iterrows():
                    team_id = row["team_id"]
                    submission_id = row["submission_id"]
                    submission_repo = row["submission_repo"]
                    self.create_space(team_id, submission_id, submission_repo)
            time.sleep(5)
