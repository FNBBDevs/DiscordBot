import os


def main():
    os.system("pylint BOT --reports yes")
    os.system("black BOT --check --quiet")
    os.system("flake8 --ignore=W503,E501 BOT")
    os.system("ruff BOT check")
    os.system("mypy BOT --ignore-missing-imports")
    os.system("isort . --profile=black --lines-after-imports=2 --check")
    os.system("bandit -r BOT -s B608")


if __name__ == "__main__":
    main()
