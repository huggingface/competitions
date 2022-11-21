import argparse

from .. import __version__
from .run import RunCompetitionsAppCommand


def main():
    parser = argparse.ArgumentParser(
        "Competitions CLI",
        usage="competitions <command> [<args>]",
        epilog="For more information about a command, run: `competitions <command> --help`",
    )
    parser.add_argument("--version", "-v", help="Display competitions version", action="store_true")
    commands_parser = parser.add_subparsers(help="commands")

    # Register commands
    RunCompetitionsAppCommand.register_subcommand(commands_parser)

    args = parser.parse_args()

    if args.version:
        print(__version__)
        exit(0)

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    command = args.func(args)

    # try:
    command.run()
    # except Exception as e:
    #    logger.error(e)
    #    sys.exit(1)


if __name__ == "__main__":
    main()
