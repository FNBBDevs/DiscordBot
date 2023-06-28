import re
import sys
import time
import contextlib
import multiprocessing
import subprocess
from io import StringIO


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
                ["nolang", "./BOT/_utils/_tmp/tmp.nl"], stdout=subprocess.PIPE
            )
            out, err = proc.communicate()
            # os.system('nolang ./BOT/_utils/_gif/tmp.nl')
            if out != "":
                results[pvn] = ("NORMAL", "[OUTPUT]\n" + out.decode("utf-8"))
            else:
                results[pvn] = ("INFO", "[0;45;37mno output produced[0;0m")
        except Exception as exception:
            error_response = ""
            line_num = None
            exception = str(exception)
            try:
                line_num = int(re.search("line (\d+)\)", exception).groups()[0])
            except Exception:
                pass

            error_response += f"[ERROR]: {exception}\n"
            if line_num:
                for i, line in enumerate(program.split("\n")):
                    if i == line_num - 2:
                        error_response += f"line {line_num}: '{line}'"

            results[pvn] = ("ERROR", error_response)
    if debug:
        print(f"leaving EPC. . .\nwith {pvn} val of {results[pvn]}")


class Nolang:
    def __init__(self, debug=False, post_val_name="POST"):
        self._debug = debug
        self._responses = []
        self._manager = multiprocessing.Manager()
        self._results = self._manager.dict()
        self._post_val_name = post_val_name

    def run(self, arg, argvs, user):
        if arg == "-s":
            pre_process = (
                f"{' '.join(argvs) if argvs else ''}".replace("^", "\n")
                .replace("&", "    ")
                .replace("“", '"')
                .replace("”", '"')
                .replace("\\\\", "\\")
            )
            self._responses.append(("PY", pre_process))
        else:
            pre_process = (
                f"{arg + ' ' + (' '.join(argvs) if argvs else '')}".replace("^", "\n")
                .replace("&", "    ")
                .replace("“", '"')
                .replace("”", '"')
                .replace("\\\\", "\\")
            )

        with open("./BOT/_utils/_tmp/tmp.nl", "w", encoding="utf-8") as f:
            for line in pre_process:
                f.write(line)

        process = multiprocessing.Process(
            target=execute_processed_command,
            args=(pre_process, self._results, self._debug, self._post_val_name),
        )
        process.start()

        time.sleep(3)

        if process.is_alive():
            process.terminate()
            self._responses.append(
                ("ERROR", "[0;41;37m[ERROR]: valid runtime exceeded![0;0m")
            )
        else:
            self._responses.append(self._results[self._post_val_name])

        return self._responses
