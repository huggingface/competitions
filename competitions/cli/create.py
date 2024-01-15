import io
import json
from argparse import ArgumentParser

import click
from huggingface_hub import HfApi, get_token
from loguru import logger
from sklearn.metrics import get_scorer_names

from . import BaseCompetitionsCommand


COMPETITION_DESC = """Sample competition description"""
DATASET_DESC = """Sample dataset description"""
SUBMISSION_DESC = """Sample submission description"""
SOLUTION_CSV = """
id,pred,split
0,1,public
1,0,private
2,0,private
3,1,private
4,0,public
5,1,private
6,1,public
7,1,private
8,0,public
9,0,private
10,0,private
11,0,private
12,1,private
13,0,private
14,1,public
15,1,private
16,1,private
17,0,private
18,0,private
19,0,public
20,0,private
21,0,private
22,1,private
23,1,public
24,0,private
25,0,private
26,0,public
27,1,private
28,1,private
29,0,private
30,0,public
"""
SOLUTION_CSV = SOLUTION_CSV.strip()

DOCKERFILE = """
FROM huggingface/competitions:latest

CMD uvicorn competitions.app:app --host 0.0.0.0 --port 7860 --workers 1
"""
DOCKERFILE = DOCKERFILE.replace("\n", " ").replace("  ", "\n").strip()

README = """
---
title: My Competition
emoji: 🤗
colorFrom: indigo
colorTo: gray
sdk: docker
pinned: false
---
"""
README = README.strip()


def create_command_factory(args):
    return CreateCompetitionAppCommand()


class CreateCompetitionAppCommand(BaseCompetitionsCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        create_project_parser = parser.add_parser("create", description="✨ Create a new competition")
        create_project_parser.set_defaults(func=create_command_factory)

    def _create_readme(self, competition_name):
        _readme = "---\n"
        _readme += f"title: {competition_name}\n"
        _readme += "emoji: 🚀\n"
        _readme += "colorFrom: green\n"
        _readme += "colorTo: indigo\n"
        _readme += "sdk: docker\n"
        _readme += "pinned: false\n"
        _readme += "duplicated_from: autotrain-projects/autotrain-advanced\n"
        _readme += "---\n"
        _readme = io.BytesIO(_readme.encode())
        return _readme

    def run(self):
        competition_name_text = "Competition name. Must be unqiue and contain only letters, numbers & hypens."
        competition_name = click.prompt(competition_name_text, type=str)
        competition_name = competition_name.lower().replace(" ", "-")
        competition_name = competition_name.replace("_", "-")
        competition_name = competition_name.replace(".", "-")
        competition_name = competition_name.replace("/", "-")
        competition_name = competition_name.replace("\\", "-")
        competition_name = competition_name.replace(":", "-")
        competition_name = competition_name.replace(";", "-")
        competition_name = competition_name.replace(",", "-")
        competition_name = competition_name.replace("!", "-")
        competition_name = competition_name.replace("?", "-")
        competition_name = competition_name.replace("'", "-")
        competition_name = competition_name.replace('"', "-")
        competition_name = competition_name.replace("`", "-")
        competition_name = competition_name.replace("~", "-")
        competition_name = competition_name.replace("@", "-")
        competition_name = competition_name.replace("#", "-")

        competition_org_text = "Competition organization. Choose one of the organizations you are a part of."
        competition_org = click.prompt(competition_org_text, type=str)

        competition_type_text = "Competition type. Choose one of 'generic', 'script'"
        competition_type = click.prompt(competition_type_text, type=str)
        if competition_type not in ["generic", "script"]:
            raise ValueError(f"Competition type {competition_type} not found in ['generic', 'script']")
        if competition_type == "script":
            time_limit = click.prompt("Time limit in seconds", type=int)
        else:
            time_limit = 10

        hardware_choices = [
            "cpu-basic",
            "cpu-upgrade",
            "t4-small",
            "t4-medium",
            "zero-a10g",
            "a10g-small",
            "a10g-large",
            "a10g-largex2",
            "a10g-largex4",
            "a100-large",
        ]
        hardware_text = f"Hardware. Choose one of {hardware_choices}"
        hardware = click.prompt(hardware_text, type=str)
        if hardware not in hardware_choices:
            raise ValueError(f"Hardware {hardware} not found in {hardware_choices}")

        metric_choices = get_scorer_names()
        metric_text = f"Metric. Choose one of {metric_choices}"
        metric = click.prompt(metric_text, type=str)
        if metric not in metric_choices:
            raise ValueError(f"Metric {metric} not found in {metric_choices}")

        eval_higher_text = "Is higher metric better? Enter 1, if yes"
        eval_higher = click.prompt(eval_higher_text, type=int)
        if eval_higher not in [0, 1]:
            raise ValueError("Invalid value for eval_higher. Must be 0 or 1")

        submission_limit_text = "Daily submission limit"
        submission_limit = click.prompt(submission_limit_text, type=int)
        if submission_limit < 1:
            raise ValueError("Submission limit must be positive integer, greater than 0")

        end_date_text = "End date. Format: YYYY-MM-DD. Private leaderboard will be available on this date."
        end_date = click.prompt(end_date_text, type=str)

        submission_id_col_text = "Submission ID column name. This column will be used to identify submissions."
        submission_id_col = click.prompt(submission_id_col_text, type=str)

        submission_cols_text = "Submission columns. Enter comma separated column names, including id column."
        submission_cols = click.prompt(submission_cols_text, type=str)

        submission_rows_text = "Submission rows. How many rows are allowed in a submission, exluding header?"
        submission_rows = click.prompt(submission_rows_text, type=int)

        competition_logo_text = "Competition logo. Enter URL to logo."
        competition_logo = click.prompt(competition_logo_text, type=str)

        conf_json = {
            "COMPETITION_TYPE": competition_type,
            "SUBMISSION_LIMIT": submission_limit,
            "TIME_LIMIT": time_limit,
            "SELECTION_LIMIT": 2,
            "END_DATE": end_date,
            "EVAL_HIGHER_IS_BETTER": eval_higher,
            "SUBMISSION_ID_COLUMN": submission_id_col,
            "SUBMISSION_COLUMNS": submission_cols,
            "SUBMISSION_ROWS": submission_rows,
            "EVAL_METRIC": metric,
            "LOGO": competition_logo,
        }

        teams_json = {}
        user_team_json = {}

        logger.info(f"Creating competition: {competition_name}")

        api = HfApi()
        api.create_repo(
            repo_id=f"{competition_org}/{competition_name}",
            repo_type="dataset",
            private=True,
        )

        conf_json = json.dumps(conf_json, indent=4)
        conf_json_bytes = conf_json.encode("utf-8")
        conf_json_buffer = io.BytesIO(conf_json_bytes)
        api.upload_file(
            path_or_fileobj=conf_json_buffer,
            path_in_repo="conf.json",
            repo_id=f"{competition_org}/{competition_name}",
            repo_type="dataset",
        )

        teams_json = json.dumps(teams_json, indent=4)
        teams_json_bytes = teams_json.encode("utf-8")
        teams_json_buffer = io.BytesIO(teams_json_bytes)
        api.upload_file(
            path_or_fileobj=teams_json_buffer,
            path_in_repo="teams.json",
            repo_id=f"{competition_org}/{competition_name}",
            repo_type="dataset",
        )

        user_team_json = json.dumps(user_team_json, indent=4)
        user_team_json_bytes = user_team_json.encode("utf-8")
        user_team_json_buffer = io.BytesIO(user_team_json_bytes)
        api.upload_file(
            path_or_fileobj=user_team_json_buffer,
            path_in_repo="user_team.json",
            repo_id=f"{competition_org}/{competition_name}",
            repo_type="dataset",
        )

        # create competition space
        api.create_repo(
            repo_id=f"{competition_org}/{competition_name}",
            repo_type="space",
            space_sdk="docker",
            space_hardware="cpu-basic" if competition_type == "generic" else hardware,
            private=True,
        )
        api.add_space_secret(repo_id=f"{competition_org}/{competition_name}", key="HF_TOKEN", value=get_token())
        api.add_space_secret(
            repo_id=f"{competition_org}/{competition_name}",
            key="COMPETITION_ID",
            value=f"{competition_org}/{competition_name}",
        )
        readme = self._create_readme(competition_name)
        api.upload_file(
            path_or_fileobj=readme,
            path_in_repo="README.md",
            repo_id=f"{competition_org}/{competition_name}",
            repo_type="space",
        )

        _dockerfile = io.BytesIO(DOCKERFILE.encode())
        api.upload_file(
            path_or_fileobj=_dockerfile,
            path_in_repo="Dockerfile",
            repo_id=f"{competition_org}/{competition_name}",
            repo_type="space",
        )

        logger.info(
            "Created private dataset and competition space. To make competition public, you should make the space private. Please note that the dataset should always be kept private."
        )
