from argparse import ArgumentParser

from . import BaseCompetitionsCommand


def create_command_factory(args):
    return CreateCompetitionAppCommand()


class CreateCompetitionAppCommand(BaseCompetitionsCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        create_project_parser = parser.add_parser("create", description="✨ Start UI to create a new competition")
        create_project_parser.set_defaults(func=create_command_factory)

    def run(self):
        from competitions.create import main

        demo = main()
        demo.launch()
