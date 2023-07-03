from invoke import task

bot_dir = "BOT"

@task
def lint(c, reports=False):
    reports = "--reports yes" if reports else ""
    c.run(
        f"pylint {reports}tasks {bot_dir}"
    )
    c.run(
        "ruff . check"
    )


@task
def format(c, check_only=False):
    check = "--check" if check_only else ""
    c.run(
        f"black {bot_dir}{check}"
    )
    c.run(
        f"isort --py auto {bot_dir}{check}"
    )
