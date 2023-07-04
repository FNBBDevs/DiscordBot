import bruhcolor
from invoke import task


bot_dir = "BOT"
start_file = "run"


@task
def lint(c):
    print(
        f"\n{bruhcolor.bruhcolored('Running linter silently... ', 'white', attrs='bold')}"
        f"{bruhcolor.bruhcolored('This may take a minute!', color=116, attrs='bold')}"
    )
    c.run(f"python tasks/linting.py")
