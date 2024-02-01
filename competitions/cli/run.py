from argparse import ArgumentParser

from . import BaseCompetitionsCommand


def run_app_command_factory(args):
    return RunCompetitionsAppCommand(args)


class RunCompetitionsAppCommand(BaseCompetitionsCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        create_project_parser = parser.add_parser("run", description="✨ Run competitions app")
        create_project_parser.add_argument("--host", default="0.0.0.0", help="Host to run app on")
        create_project_parser.add_argument("--port", default=7860, help="Port to run app on")
        create_project_parser.set_defaults(func=run_app_command_factory)

    def __init__(self, args):
        self.host = args.host
        self.port = args.port

    def run(self):
        import uvicorn

        from competitions.app import app

        uvicorn.run(app, host=self.host, port=self.port)
