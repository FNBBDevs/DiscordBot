import bruhcolor
from bruhcolor import bruhcolored
from invoke import task


bot_dir = "BOT"
start_file = "run"


@task
def lint(c, silent=False):
    if silent:
        print(
            f"\n{bruhcolor.bruhcolored('Running linter silently... ', 'white', attrs='bold')}"
            f"{bruhcolor.bruhcolored('This may take a minute!', color=116, attrs='bold')}"
        )
        c.run(f"python tasks/silint.py")
    else:
        c.run(f"python tasks/linter.py")


@task
def format(c):
    print()
    print(bruhcolored("RUNNING BLACK FORMATTER", color=118) + "\n")
    c.run(f"black .")
    print(bruhcolored("RUNNING ISORT IMPORT SORTER", color=118) + "\n")
    c.run(f"isort . --profile=black --lines-after-imports=2 --py 311 > NUL")
