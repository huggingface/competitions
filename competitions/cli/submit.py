import os
from argparse import ArgumentParser

import requests

from . import BaseCompetitionsCommand


def submit_commands_factory(args):
    return SubmitCompetitionAppCommand(args)


class SubmitCompetitionAppCommand(BaseCompetitionsCommand):
    def __init__(self, args):
        self.args = args

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        submit_competition_parser = parser.add_parser("submit", description="Submit to a competition")
        submit_competition_parser.add_argument(
            "--competition_id", type=str, help="ID of the competition, e.g. huggingface/cool-competition"
        )
        submit_competition_parser.add_argument(
            "--submission", type=str, help="Path to submission file or HuggingFace hub repo"
        )
        submit_competition_parser.add_argument("--comment", type=str, help="Submission comment", default="")
        submit_competition_parser.add_argument("--token", type=str, help="User token, read-only", default="")
        submit_competition_parser.set_defaults(func=submit_commands_factory)

    def run(self):
        if os.path.isfile(self.args.submission):
            files = {"submission_file": open(self.args.submission, "rb")}
            data = {
                "hub_model": "",
                "submission_comment": self.args.comment,
            }
        else:
            files = {"submission_file": None}
            data = {
                "hub_model": self.args.submission,
                "submission_comment": self.args.comment,
            }

        headers = {"Authorization": f"Bearer {self.args.token}"}

        api_url = "https://" + self.args.competition_id.replace("/", "-") + ".hf.space/new_submission"

        response = requests.post(api_url, data=data, files=files, headers=headers)
        print(response.json())
