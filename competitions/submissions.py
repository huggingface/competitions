import io
import json
import uuid
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from competitions.enums import SubmissionStatus
from competitions.errors import AuthenticationError, PastDeadlineError, SubmissionError, SubmissionLimitError
from competitions.utils import user_authentication


@dataclass
class Submissions:
    competition_id: str
    competition_type: str
    submission_limit: str
    hardware: str
    end_date: datetime
    token: str

    def _verify_submission(self, bytes_data):
        return True

    def _num_subs_today(self, todays_date, team_submission_info):
        todays_submissions = 0
        for sub in team_submission_info["submissions"]:
            submission_datetime = sub["datetime"]
            submission_date = submission_datetime.split(" ")[0]
            if submission_date == todays_date:
                todays_submissions += 1
        return todays_submissions

    def _is_submission_allowed(self, team_id):
        todays_date = datetime.utcnow()
        if todays_date > self.end_date:
            raise PastDeadlineError("Competition has ended.")

        todays_date = todays_date.strftime("%Y-%m-%d")
        team_submission_info = self._download_team_submissions(team_id)

        if len(team_submission_info["submissions"]) == 0:
            team_submission_info["submissions"] = []

        todays_submissions = self._num_subs_today(todays_date, team_submission_info)
        if todays_submissions >= self.submission_limit:
            return False
        return True

    def _increment_submissions(
        self,
        team_id,
        user_id,
        submission_id,
        submission_comment,
        submission_repo=None,
        space_id=None,
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
                "status": SubmissionStatus.PENDING.value,
                "selected": False,
                "public_score": {},
                "private_score": {},
            }
        )
        # count the number of times user has submitted today
        todays_date = datetime.utcnow().strftime("%Y-%m-%d")
        todays_submissions = self._num_subs_today(todays_date, team_submission_info)
        self._upload_team_submissions(team_id, team_submission_info)
        return todays_submissions

    def _upload_team_submissions(self, team_id, team_submission_info):
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

    def _download_team_submissions(self, team_id):
        team_fname = hf_hub_download(
            repo_id=self.competition_id,
            filename=f"submission_info/{team_id}.json",
            token=self.token,
            repo_type="dataset",
        )
        with open(team_fname, "r", encoding="utf-8") as f:
            team_submission_info = json.load(f)
        return team_submission_info

    def update_selected_submissions(self, user_token, selected_submission_ids):
        current_datetime = datetime.utcnow()
        if current_datetime > self.end_date:
            raise PastDeadlineError("Competition has ended.")

        user_info = self._get_user_info(user_token)
        team_id = self._get_team_id(user_info, create_team=False)
        team_submission_info = self._download_team_submissions(team_id)

        for sub in team_submission_info["submissions"]:
            if sub["submission_id"] in selected_submission_ids:
                sub["selected"] = True
            else:
                sub["selected"] = False

        self._upload_team_submissions(team_id, team_submission_info)

    def _get_team_subs(self, team_id, private=False):
        team_submissions_info = self._download_team_submissions(team_id)
        submissions_df = pd.DataFrame(team_submissions_info["submissions"])

        if len(submissions_df) == 0:
            return pd.DataFrame(), pd.DataFrame()

        if not private:
            submissions_df = submissions_df.drop(columns=["private_score"])

        submissions_df = submissions_df.sort_values(by="datetime", ascending=False)
        submissions_df = submissions_df.reset_index(drop=True)

        # stringify public_score column
        submissions_df["public_score"] = submissions_df["public_score"].apply(json.dumps)

        if private:
            submissions_df["private_score"] = submissions_df["private_score"].apply(json.dumps)

        submissions_df["status"] = submissions_df["status"].apply(lambda x: SubmissionStatus(x).name)

        return submissions_df

    def _get_user_info(self, user_token):
        user_info = user_authentication(token=user_token)
        if "error" in user_info:
            raise AuthenticationError("Invalid token")

        # if user_info["emailVerified"] is False:
        #     raise AuthenticationError("Please verify your email on Hugging Face Hub")
        return user_info

    def my_submissions(self, user_token):
        user_info = self._get_user_info(user_token)
        current_date_time = datetime.utcnow()
        private = False
        if current_date_time >= self.end_date:
            private = True
        team_id = self._get_team_id(user_info, create_team=False)
        if not team_id:
            return pd.DataFrame()
        return self._get_team_subs(team_id, private=private)

    def _create_team(self, user_team, user_id, user_name):
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

        team_submission_info = {}
        team_submission_info["id"] = team_id
        team_submission_info["submissions"] = []
        team_submission_info_json = json.dumps(team_submission_info, indent=4)
        team_submission_info_json_bytes = team_submission_info_json.encode("utf-8")
        team_submission_info_json_buffer = io.BytesIO(team_submission_info_json_bytes)

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
        api.upload_file(
            path_or_fileobj=team_submission_info_json_buffer,
            path_in_repo=f"submission_info/{team_id}.json",
            repo_id=self.competition_id,
            repo_type="dataset",
        )
        return team_id

    def _get_team_id(self, user_info, create_team):
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

        if create_team is False:
            return None

        # if user_id is not there in user_team, create a new team
        team_id = self._create_team(user_team, user_id, user_name)
        return team_id

    def new_submission(self, user_token, uploaded_file, submission_comment):
        # verify token
        user_info = self._get_user_info(user_token)
        submission_id = str(uuid.uuid4())
        user_id = user_info["id"]
        team_id = self._get_team_id(user_info, create_team=True)

        # check if team can submit to the competition
        if self._is_submission_allowed(team_id) is False:
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
            )
        remaining_submissions = self.submission_limit - submissions_made
        return remaining_submissions
