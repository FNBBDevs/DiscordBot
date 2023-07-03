from invoke import task
from bruhcolor import bruhcolored

bot_dir = "BOT"
start_file = "run"


@task
def lint(c, reports=False):
    reports = "--reports yes" if reports else ""
    print(bruhcolored("RUNNING PYLINTER: ", color=118) + "\n\n")
    c.run(
        f"pylint {reports} {bot_dir}"
    )
    print(bruhcolored("RUNNING RUFF LINTER: ", color=118) + "\n\n")
    c.run(
        "ruff . check"
    )


@task
def format(c, check_only=False):
    check = "--check" if check_only else ""
    print(bruhcolored("RUNNING BLACK FORMATTER: ", color=118) + "\n\n")
    c.run(
        f"black {bot_dir}{check}"
    )
    print(bruhcolored("RUNNING ISORT IMPORT SORTER: ", color=118) + "\n\n")
    c.run(
        f"isort --py auto {bot_dir}{check}"
    )

