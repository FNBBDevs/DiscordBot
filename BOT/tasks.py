from invoke import task
import pylint.lint

@task
def lint(c):
    pylint.lint.Run(["*"])


# @task(help={'lint': "Run the linter on the code."})
# def lint(c, lint):
#     """
#     Say hi to someone.
#     """
#     print("Hi {}!".format(lint))