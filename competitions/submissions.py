import io
import json
import uuid
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils._errors import EntryNotFoundError
from loguru import logger

from .errors import AuthenticationError, PastDeadlineError, SubmissionError, SubmissionLimitError
from .utils import user_authentication


@dataclass
class Submissions:
    competition_id: str
    competition_type: str
    submission_limit: str
    end_date: datetime
    token: str

    def __post_init__(self):
        self.public_sub_columns = [
            "datetime",
            "submission_id",
            "public_score",
            "submission_comment",
            "selected",
            "status",
        ]
        self.private_sub_columns = [
            "datetime",
            "submission_id",
            "public_score",
            "private_score",
            "submission_comment",
            "selected",
            "status",
        ]

    def _verify_submission(self, bytes_data):
        return True

    def _add_new_team(self, team_id):
        api = HfApi(token=self.token)
        team_submission_info = {}
        team_submission_info["id"] = team_id
        team_submission_info["submissions"] = []
        team_submission_info_json = json.dumps(team_submission_info, indent=4)
        team_submission_info_json_bytes = team_submission_info_json.encode("utf-8")
        team_submission_info_json_buffer = io.BytesIO(team_submission_info_json_bytes)

        api.upload_file(
            path_or_fileobj=team_submission_info_json_buffer,
            path_in_repo=f"submission_info/{team_id}.json",
            repo_id=self.competition_id,
            repo_type="dataset",
        )

    def _check_team_submission_limit(self, team_id):
        try:
            team_fname = hf_hub_download(
                repo_id=self.competition_id,
                filename=f"submission_info/{team_id}.json",
                token=self.token,
                repo_type="dataset",
            )
        except EntryNotFoundError:
            self._add_new_team(team_id)
            team_fname = hf_hub_download(
                repo_id=self.competition_id,
                filename=f"submission_info/{team_id}.json",
                token=self.token,
                repo_type="dataset",
            )
        except Exception as e:
            logger.error(e)
            raise Exception("Hugging Face Hub is unreachable, please try again later.")

        with open(team_fname, "r", encoding="utf-8") as f:
            team_submission_info = json.load(f)

        todays_date = datetime.utcnow().strftime("%Y-%m-%d")
        if len(team_submission_info["submissions"]) == 0:
            team_submission_info["submissions"] = []

        # count the number of times user has submitted today
        todays_submissions = 0
        for sub in team_submission_info["submissions"]:
            submission_datetime = sub["datetime"]
            submission_date = submission_datetime.split(" ")[0]
            if submission_date == todays_date:
                todays_submissions += 1
        if todays_submissions >= self.submission_limit:
            return False
        return True

    def _submissions_today(self, team_id):
        try:
            team_fname = hf_hub_download(
                repo_id=self.competition_id,
                filename=f"submission_info/{team_id}.json",
                token=self.token,
                repo_type="dataset",
            )
        except EntryNotFoundError:
            self._add_new_team(team_id)
            team_fname = hf_hub_download(
                repo_id=self.competition_id,
                filename=f"submission_info/{team_id}.json",
                token=self.token,
                repo_type="dataset",
            )
        except Exception as e:
            logger.error(e)
            raise Exception("Hugging Face Hub is unreachable, please try again later.")

        with open(team_fname, "r", encoding="utf-8") as f:
            team_submission_info = json.load(f)

        todays_date = datetime.utcnow().strftime("%Y-%m-%d")
        if len(team_submission_info["submissions"]) == 0:
            team_submission_info["submissions"] = []

        # count the number of times user has submitted today
        todays_submissions = 0
        for sub in team_submission_info["submissions"]:
            submission_datetime = sub["datetime"]
            submission_date = submission_datetime.split(" ")[0]
            if submission_date == todays_date:
                todays_submissions += 1
        return todays_submissions

    def _increment_submissions(
        self,
        team_id,
        user_id,
        submission_id,
        submission_comment,
        submission_repo=None,
        space_id=None,
        space_status=0,
    ):
        if submission_repo is None:
            submission_repo = ""
        if space_id is None:
            space_id = ""
        team_fname = hf_hub_download(
            repo_id=self.competition_id,
            filename=f"submission_info/{team_id}.json",
            token=self.token,
            repo_type="dataset",
        )
        with open(team_fname, "r", encoding="utf-8") as f:
            team_submission_info = json.load(f)
        datetime_now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        # here goes all the default stuff for submission
        team_submission_info["submissions"].append(
            {
                "datetime": datetime_now,
                "submission_id": submission_id,
                "submission_comment": submission_comment,
                "submission_repo": submission_repo,
                "space_id": space_id,
                "submitted_by": user_id,
                "status": "pending",
                "selected": False,
                "public_score": -1,
                "private_score": -1,
                "space_status": space_status,
            }
        )
        # count the number of times user has submitted today
        todays_submissions = 0
        todays_date = datetime.utcnow().strftime("%Y-%m-%d")
        for sub in team_submission_info["submissions"]:
            submission_datetime = sub["datetime"]
            submission_date = submission_datetime.split(" ")[0]
            if submission_date == todays_date:
                todays_submissions += 1

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
        return todays_submissions

    def _download_team_subs(self, team_id):
        team_fname = hf_hub_download(
            repo_id=self.competition_id,
            filename=f"submission_info/{team_id}.json",
            token=self.token,
            repo_type="dataset",
        )
        with open(team_fname, "r", encoding="utf-8") as f:
            team_submission_info = json.load(f)
        return team_submission_info["submissions"]

    def update_selected_submissions(self, user_token, selected_submission_ids):
        current_datetime = datetime.utcnow()
        if current_datetime > self.end_date:
            raise PastDeadlineError("Competition has ended.")

        user_info = self._get_user_info(user_token)
        team_id = self._get_team_id(user_info)

        team_fname = hf_hub_download(
            repo_id=self.competition_id,
            filename=f"submission_info/{team_id}.json",
            token=self.token,
            repo_type="dataset",
        )
        with open(team_fname, "r", encoding="utf-8") as f:
            team_submission_info = json.load(f)

        for sub in team_submission_info["submissions"]:
            if sub["submission_id"] in selected_submission_ids:
                sub["selected"] = True
            else:
                sub["selected"] = False

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

    def _get_team_subs(self, team_id, private=False):
        try:
            team_submissions = self._download_team_subs(team_id)
        except EntryNotFoundError:
            logger.warning("No submissions found for user")
            return pd.DataFrame(), pd.DataFrame()

        submissions_df = pd.DataFrame(team_submissions)

        if not private:
            submissions_df = submissions_df.drop(columns=["private_score"])
            submissions_df = submissions_df[self.public_sub_columns]
        else:
            submissions_df = submissions_df[self.private_sub_columns]
        if not private:
            failed_submissions = submissions_df[
                (submissions_df["status"].isin(["failed", "error"])) | (submissions_df["public_score"] == -1)
            ]
            successful_submissions = submissions_df[
                ~submissions_df["status"].isin(["failed", "error"]) & (submissions_df["public_score"] != -1)
            ]
        else:
            failed_submissions = submissions_df[
                (submissions_df["status"].isin(["failed", "error"]))
                | (submissions_df["private_score"] == -1)
                | (submissions_df["public_score"] == -1)
            ]
            successful_submissions = submissions_df[
                ~submissions_df["status"].isin(["failed", "error"])
                & (submissions_df["private_score"] != -1)
                & (submissions_df["public_score"] != -1)
            ]
        failed_submissions = failed_submissions.reset_index(drop=True)
        successful_submissions = successful_submissions.reset_index(drop=True)

        if not private:
            first_submission = successful_submissions.iloc[0]
            if isinstance(first_submission["public_score"], dict):
                # split the public score dict into columns
                temp_scores_df = successful_submissions["public_score"].apply(pd.Series)
                temp_scores_df = temp_scores_df.rename(columns=lambda x: "public_" + str(x))
                successful_submissions = pd.concat(
                    [
                        successful_submissions.drop(["public_score"], axis=1),
                        temp_scores_df,
                    ],
                    axis=1,
                )
        else:
            first_submission = successful_submissions.iloc[0]
            if isinstance(first_submission["private_score"], dict):
                # split the public score dict into columns
                temp_scores_df = successful_submissions["private_score"].apply(pd.Series)
                temp_scores_df = temp_scores_df.rename(columns=lambda x: "private_" + str(x))
                successful_submissions = pd.concat(
                    [
                        successful_submissions.drop(["private_score"], axis=1),
                        temp_scores_df,
                    ],
                    axis=1,
                )

            if isinstance(first_submission["public_score"], dict):
                # split the public score dict into columns
                temp_scores_df = successful_submissions["public_score"].apply(pd.Series)
                temp_scores_df = temp_scores_df.rename(columns=lambda x: "public_" + str(x))
                successful_submissions = pd.concat(
                    [
                        successful_submissions.drop(["public_score"], axis=1),
                        temp_scores_df,
                    ],
                    axis=1,
                )
        return successful_submissions, failed_submissions

    def _get_user_info(self, user_token):
        user_info = user_authentication(token=user_token)
        if "error" in user_info:
            raise AuthenticationError("Invalid token")

        if user_info["emailVerified"] is False:
            raise AuthenticationError("Please verify your email on Hugging Face Hub")
        return user_info

    def my_submissions(self, user_token):
        user_info = self._get_user_info(user_token)
        current_date_time = datetime.utcnow()
        private = False
        if current_date_time >= self.end_date:
            private = True
        team_id = self._get_team_id(user_info)
        success_subs, failed_subs = self._get_team_subs(team_id, private=private)
        return success_subs, failed_subs

    def _get_team_id(self, user_info):
        user_id = user_info["id"]
        user_name = user_info["name"]
        user_team = hf_hub_download(
            repo_id=self.competition_id,
            filename="user_team.json",
            token=self.token,
            repo_type="dataset",
        )
        with open(user_team, "r", encoding="utf-8") as f:
            user_team = json.load(f)

        if user_id in user_team:
            return user_team[user_id]

        team_metadata = hf_hub_download(
            repo_id=self.competition_id,
            filename="teams.json",
            token=self.token,
            repo_type="dataset",
        )

        with open(team_metadata, "r", encoding="utf-8") as f:
            team_metadata = json.load(f)

        # create a new team, if user is not in any team
        team_id = str(uuid.uuid4())
        user_team[user_id] = team_id

        team_metadata[team_id] = {
            "id": team_id,
            "name": user_name,
            "members": [user_id],
            "leader": user_id,
        }

        user_team_json = json.dumps(user_team, indent=4)
        user_team_json_bytes = user_team_json.encode("utf-8")
        user_team_json_buffer = io.BytesIO(user_team_json_bytes)

        team_metadata_json = json.dumps(team_metadata, indent=4)
        team_metadata_json_bytes = team_metadata_json.encode("utf-8")
        team_metadata_json_buffer = io.BytesIO(team_metadata_json_bytes)

        api = HfApi(token=self.token)
        api.upload_file(
            path_or_fileobj=user_team_json_buffer,
            path_in_repo="user_team.json",
            repo_id=self.competition_id,
            repo_type="dataset",
        )
        api.upload_file(
            path_or_fileobj=team_metadata_json_buffer,
            path_in_repo="teams.json",
            repo_id=self.competition_id,
            repo_type="dataset",
        )

        return team_id

    def new_submission(self, user_token, uploaded_file, submission_comment):
        # verify token
        user_info = self._get_user_info(user_token)
        submission_id = str(uuid.uuid4())
        user_id = user_info["id"]
        team_id = self._get_team_id(user_info)

        # check if team can submit to the competition
        if self._check_team_submission_limit(team_id) is False:
            raise SubmissionLimitError("Submission limit reached")

        if self.competition_type == "generic":
            bytes_data = uploaded_file.file.read()
            # verify file is valid
            if not self._verify_submission(bytes_data):
                raise SubmissionError("Invalid submission file")

            file_extension = uploaded_file.filename.split(".")[-1]
            # upload file to hf hub
            api = HfApi(token=self.token)
            api.upload_file(
                path_or_fileobj=bytes_data,
                path_in_repo=f"submissions/{team_id}-{submission_id}.{file_extension}",
                repo_id=self.competition_id,
                repo_type="dataset",
            )
            submissions_made = self._increment_submissions(
                team_id=team_id,
                user_id=user_id,
                submission_id=submission_id,
                submission_comment=submission_comment,
            )
        else:
            # Download the submission repo and upload it to the competition repo
            # submission_repo = snapshot_download(
            #     repo_id=uploaded_file,
            #     local_dir=submission_id,
            #     token=user_token,
            #     repo_type="model",
            # )
            # api = HfApi(token=self.token)
            # competition_user = self.competition_id.split("/")[0]
            # api.create_repo(
            #     repo_id=f"{competition_user}/{submission_id}",
            #     repo_type="model",
            #     private=True,
            # )
            # api.upload_folder(
            #     folder_path=submission_repo,
            #     repo_id=f"{competition_user}/{submission_id}",
            #     repo_type="model",
            # )
            # create barebones submission runner space
            competition_organizer = self.competition_id.split("/")[0]
            space_id = f"{competition_organizer}/comp-{submission_id}"
            api = HfApi(token=self.token)
            api.create_repo(
                repo_id=space_id,
                repo_type="space",
                space_sdk="docker",
                space_hardware=self.hardware,
                private=True,
            )

            api.add_space_secret(repo_id=space_id, key="USER_TOKEN", value=user_token)
            submissions_made = self._increment_submissions(
                team_id=team_id,
                user_id=user_id,
                submission_id=submission_id,
                submission_comment=submission_comment,
                submission_repo=uploaded_file,
                space_id=space_id,
                space_status=0,
            )
        remaining_submissions = self.submission_limit - submissions_made
        return remaining_submissions
