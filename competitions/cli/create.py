from argparse import ArgumentParser

from . import BaseCompetitionsCommand


def create_command_factory(args):
    return CreateCompetitionAppCommand()


class CreateCompetitionAppCommand(BaseCompetitionsCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        create_project_parser = parser.add_parser("create", description="✨ Create a new competition")
        create_project_parser.set_defaults(func=create_command_factory)

    def run(self):
        from ..create import demo

        demo.queue().launch()
