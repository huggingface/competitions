import glob
import io
import json
import os
import time
from dataclasses import dataclass

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from loguru import logger

from competitions.enums import SubmissionStatus
from competitions.info import CompetitionInfo
from competitions.utils import run_evaluation


_DOCKERFILE = """
FROM huggingface/competitions:latest

CMD uvicorn competitions.api:api --port 7860 --host 0.0.0.0
"""

# format _DOCKERFILE
_DOCKERFILE = _DOCKERFILE.replace("\n", " ").replace("  ", "\n").strip()


@dataclass
class JobRunner:
    competition_id: str
    token: str
    output_path: str

    def __post_init__(self):
        self.competition_info = CompetitionInfo(competition_id=self.competition_id, autotrain_token=self.token)
        self.competition_id = self.competition_info.competition_id
        self.competition_type = self.competition_info.competition_type
        self.metric = self.competition_info.metric
        self.submission_id_col = self.competition_info.submission_id_col
        self.submission_cols = self.competition_info.submission_cols
        self.submission_rows = self.competition_info.submission_rows
        self.time_limit = self.competition_info.time_limit
        self.dataset = self.competition_info.dataset
        self.submission_filenames = self.competition_info.submission_filenames

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
                if sub["status"] == SubmissionStatus.PENDING.value:
                    pending_submissions.append(
                        {
                            "team_id": team_id,
                            "submission_id": sub["submission_id"],
                            "datetime": sub["datetime"],
                            "submission_repo": sub["submission_repo"],
                            "space_id": sub["space_id"],
                        }
                    )
        if len(pending_submissions) == 0:
            return None
        logger.info(f"Found {len(pending_submissions)} pending submissions.")
        pending_submissions = pd.DataFrame(pending_submissions)
        pending_submissions["datetime"] = pd.to_datetime(pending_submissions["datetime"])
        pending_submissions = pending_submissions.sort_values("datetime")
        pending_submissions = pending_submissions.reset_index(drop=True)
        return pending_submissions

    def _queue_submission(self, team_id, submission_id):
        team_fname = hf_hub_download(
            repo_id=self.competition_id,
            filename=f"submission_info/{team_id}.json",
            token=self.token,
            repo_type="dataset",
        )
        with open(team_fname, "r", encoding="utf-8") as f:
            team_submission_info = json.load(f)

        for submission in team_submission_info["submissions"]:
            if submission["submission_id"] == submission_id:
                submission["status"] = SubmissionStatus.QUEUED.value
                break

        team_submission_info_json = json.dumps(team_submission_info, indent=4)
        team_submission_info_json_bytes = team_submission_info_json.encode("utf-8")
        team_submission_info_json_buffer = io.BytesIO(team_submission_info_json_bytes)
        api = HfApi(token=self.token)
        api.upload_file(
            path_or_fileobj=team_submission_info_json_buffer,
            path_in_repo=f"submission_info/{team_id}.json",
            repo_id=self.competition_id,
            repo_type="dataset",
        )

    def run_local(self, team_id, submission_id, submission_repo):
        self._queue_submission(team_id, submission_id)
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
            "submission_repo": submission_repo,
            "time_limit": self.time_limit,
            "dataset": self.dataset,
            "submission_filenames": self.submission_filenames,
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

    def create_space(self, team_id, submission_id, submission_repo, space_id):
        api = HfApi(token=self.token)
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
            "time_limit": self.time_limit,
            "dataset": self.dataset,
            "submission_filenames": self.submission_filenames,
        }

        api.add_space_secret(repo_id=space_id, key="PARAMS", value=json.dumps(params))

        readme = self._create_readme(space_id.split("/")[-1])
        api.upload_file(
            path_or_fileobj=readme,
            path_in_repo="README.md",
            repo_id=space_id,
            repo_type="space",
        )

        _dockerfile = io.BytesIO(_DOCKERFILE.encode())
        api.upload_file(
            path_or_fileobj=_dockerfile,
            path_in_repo="Dockerfile",
            repo_id=space_id,
            repo_type="space",
        )
        self._queue_submission(team_id, submission_id)

    def run(self):
        while True:
            pending_submissions = self.get_pending_subs()
            if pending_submissions is None:
                time.sleep(5)
                continue
            if self.competition_type == "generic":
                for _, row in pending_submissions.iterrows():
                    team_id = row["team_id"]
                    submission_id = row["submission_id"]
                    submission_repo = row["submission_repo"]
                    self.run_local(team_id, submission_id, submission_repo)
            elif self.competition_type == "script":
                for _, row in pending_submissions.iterrows():
                    team_id = row["team_id"]
                    submission_id = row["submission_id"]
                    submission_repo = row["submission_repo"]
                    space_id = row["space_id"]
                    self.create_space(team_id, submission_id, submission_repo, space_id)
            time.sleep(5)
