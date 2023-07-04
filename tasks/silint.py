import os

import bruhcolor


def show_message(*args) -> None:
    print()
    score = 0
    for arg in args:
        if arg[0] == 0:
            score += 1
            print(
                f"{bruhcolor.bruhcolored(arg[1], color=255, attrs='bold')}:\t{'.' * 50}"
                f"{bruhcolor.bruhcolored('SUCCESS', color=118)}"
            )
        else:
            print(
                f"{bruhcolor.bruhcolored(arg[1], color=255, attrs='bold')}:\t"
                + "." * 50
                + f"{bruhcolor.bruhcolored('FAILED', color='light_red')}"
            )
    print()
    if score < 3:
        print(
            f"{bruhcolor.bruhcolored('WARNING:', color='light_red', attrs='bold')} "
            f"{bruhcolor.bruhcolored(f'passed {score}/6!', color='white')} "
        )
        print()
        print(
            f"{bruhcolor.bruhcolored('SUGGESTIONS:', color='116', attrs='bold')} "
            f"{bruhcolor.bruhcolored('Try running `', color='white')}"
            f"{bruhcolor.bruhcolored('invoke', color='220')} "
            f"{bruhcolor.bruhcolored('code.format` to autoformat your code.', color='white')}"
        )
    elif 3 <= score <= 5:
        print(
            f"{bruhcolor.bruhcolored('WARNING:', color='220', attrs='bold')} "
            f"{bruhcolor.bruhcolored(f'passed {score}/6!', color='white')} "
        )
        print()
        print(
            f"{bruhcolor.bruhcolored('SUGGESTIONS:', color='116', attrs='bold')}"
            f"{bruhcolor.bruhcolored('Try running `', color='white')}"
            f"{bruhcolor.bruhcolored('invoke', color='220')} "
            f"{bruhcolor.bruhcolored('code.format` to autoformat your code.', color='white')}"
        )
    else:
        print(
            f"{bruhcolor.bruhcolored('SUCCESS:', color='118', attrs='bold')} "
            f"{bruhcolor.bruhcolored(f'passed {score}/6!', color='white')} "
        )
    print()


def lint_clean():
    pylint = os.system("pylint BOT --reports yes > NUL")
    black = os.system("black BOT --check --quiet > NUL")
    flake = os.system("flake8 --ignore=W503,E501 BOT > NUL")
    ruff = os.system("ruff check BOT > NUL")
    mypy = os.system("mypy BOT --ignore-missing-imports > NUL")
    isort = os.system(
        "isort --profile=black --lines-after-imports=2 --check-only BOT > NUL"
    )

    show_message(
        [pylint, "PYLINT"],
        [black, "BLACK"],
        [flake, "FLAKE"],
        [ruff, "RUFF"],
        [mypy, "MYPY"],
        [isort, "ISORT"],
    )


if __name__ == "__main__":
    lint_clean()
