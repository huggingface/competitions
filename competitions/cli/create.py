from argparse import ArgumentParser

from . import BaseCompetitionsCommand
import click
from loguru import logger
from huggingface_hub import HfApi


def create_command_factory(args):
    return CreateCompetitionAppCommand()


class CreateCompetitionAppCommand(BaseCompetitionsCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        create_project_parser = parser.add_parser("create", description="✨ Create a new competition")
        create_project_parser.set_defaults(func=create_command_factory)

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
        logger.info(f"Creating competition: {competition_name}")

        org_choices = HfApi().organization_list()
        competition_org_text = "Competition organization. Choose one of {org_choices}"
        competition_org = click.prompt(competition_org_text, type=str)
        if competition_org not in org_choices:
            raise ValueError(f"Organization {competition_org} not found in {org_choices}")

        competition_type_text = "Competition type. Choose one of 'generic', 'script'"
        competition_type = click.prompt(competition_type_text, type=str)
        if competition_type not in ["generic", "script"]:
            raise ValueError(f"Competition type {competition_type} not found in ['generic', 'script']")
        if competition_type == "script":
            time_limit = click.prompt("Time limit in seconds", type=int)
        else:
            time_limit = 10
        
        hardware_choices = ["cpu-free", "gpu-free", "tpu-free"]
        hardware_text = "Hardware. Choose one of {hardware_choices}"