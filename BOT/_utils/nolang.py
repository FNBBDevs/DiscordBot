import contextlib
import multiprocessing
import re
import subprocess
import sys
import time
from io import StringIO

from .logerr import Logerr

logerr = Logerr()


@contextlib.contextmanager
def stdoutIO(stdout=None):
    """
    Function to route stdout to a new stdout and
    capture the output
    """
    old = sys.stdout
    if stdout is None:
        stdout = StringIO()
    sys.stdout = stdout
    yield stdout
    sys.stdout = old


def execute_processed_command(program, results, debug, pvn):
    """
    Function for executing the code and capturing any output to
    stdout
    :param program: the code to execute
    :param results: multiprocessing Manager Dict to store the stdout to
    :param debug  : to show debug messages
    """
    if debug:
        print(f"\nEPC called with\n{program}\n")
    with stdoutIO():
        try:
            proc = subprocess.Popen(
                ["nolang", "./BOT/_utils/_tmp/tmp.nl"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            out, err = proc.communicate()
            if out.decode("utf-8") != "":
                results[pvn] = ("OUTPUT", str(out.decode("utf-8")))
            elif err:
                err = err.decode("utf-8")
                if x := re.search(r"(.*) '.*':(\d+)", err):
                    error, line = x.groups()
                    results[pvn] = ("ERROR", f"line {line}: '{error}'")
                else:
                    results[pvn] = ("ERROR", err)
            else:
                results[pvn] = ("OUTPUT", "No output produced")
        except Exception as exception:
            # should not happen as the error is caught above . . .
            logerr.log(str(exception))
            results[pvn] = ("ERROR", "An error was encountered. Check the logs.")


class Nolang:
    def __init__(self, debug=False, post_val_name="POST"):
        self.debug = debug
        self.responses = []
        self.manager = multiprocessing.Manager()
        self.results = self.manager.dict()
        self.post_val_name = post_val_name

    def run(self, arg, argvs, user):
        if arg == "-s":
            pre_process = (
                f"{' '.join(argvs) if argvs else ''}".replace("\\t", "\t")
                .replace("“", '"')
                .replace("”", '"')
                .replace("\\\\", "\\")
            )
            self.responses.append(("NL", f"# your_code.nl\n{pre_process}"))
        else:
            pre_process = (
                f"{arg + ' ' + (' '.join(argvs) if argvs else '')}".replace("\\t", "\t")
                .replace("“", '"')
                .replace("”", '"')
                .replace("\\\\", "\\")
            )

        with open("./BOT/_utils/_tmp/tmp.nl", "w", encoding="utf-8") as f:
            for line in pre_process:
                f.write(line)

        process = multiprocessing.Process(
            target=execute_processed_command,
            args=(pre_process, self.results, self.debug, self.post_val_name),
        )
        process.start()

        time.sleep(3)

        if process.is_alive():
            process.terminate()
            self.responses.append(("ERROR", "Valid runtime exceeded!"))
        else:
            self.responses.append(self.results[self.post_val_name])

        return self.responses
