from argparse import ArgumentParser

from . import BaseCompetitionsCommand


def run_app_command_factory(args):
    return RunCompetitionsAppCommand()


class RunCompetitionsAppCommand(BaseCompetitionsCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        create_project_parser = parser.add_parser("run", description="✨ Run competitions app")
        # create_project_parser.add_argument("--name", type=str, default=None, required=True, help="The project's name")
        create_project_parser.set_defaults(func=run_app_command_factory)

    # def __init__(self):
    #     self._name = name
    #     self._task = task
    #     self._lang = language
    #     self._max_models = max_models
    #     self._hub_model = hub_model

    def run(self):
        from ..competitions import demo

        demo.launch()
