from argparse import ArgumentParser

from . import BaseCompetitionsCommand


def run_app_command_factory(args):
    return RunCompetitionsAppCommand()


class RunCompetitionsAppCommand(BaseCompetitionsCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        create_project_parser = parser.add_parser("run", description="✨ Run competitions app")
        create_project_parser.set_defaults(func=run_app_command_factory)

    def run(self):
        from ..competitions import demo

        demo.queue().launch()
