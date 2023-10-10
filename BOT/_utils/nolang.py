import multiprocessing
import re
import subprocess
import time

from _utils.capstdout import stdoutIO

from .logerr import Logerr


logerr = Logerr()


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
            if out.decode("utf-8") != "":
                results[pvn] = ("OUTPUT", str(out.decode("utf-8")))
            elif err:
                err = err.decode("utf-8")
                if error_search := re.search(r"(.*) '.*':(\d+)", err):
                    error, line = error_search.groups()
                    results[pvn] = ("ERROR", f"line {line}: '{error}'")
                else:
                    results[pvn] = ("ERROR", err)
                    results[pvn] = ("ERROR", err)
            else:
                results[pvn] = ("OUTPUT", "No output produced")
                results[pvn] = ("OUTPUT", "No output produced")
        except Exception as exception:
            # should not happen as the error is caught above . . .
            logerr.log(str(exception))
            results[pvn] = ("ERROR", "An error was encountered. Check the logs.")

            results[pvn] = ("ERROR", "An error was encountered. Check the logs.")


class Nolang:
    """
    Class to execute nolang code
    """

    def __init__(self, debug=False, post_val_name="POST"):
        self.debug = debug
        self.responses = []
        self.manager = multiprocessing.Manager()
        self.results = self.manager.dict()
        self.post_val_name = post_val_name

    def run(self, arg, argvs):
        """
        Function to process and run the nolang code
        """
        if arg == "-s":
            nolang_pre_process = (
                f"{' '.join(argvs) if argvs else ''}".replace("\\t", "\t")
                .replace("“", '"')
                .replace("”", '"')
                .replace("\\\\", "\\")
                .replace("`", "")
            )
            self.responses.append(("NL", f"# your_code.nl\n{nolang_pre_process}"))
        else:
            nolang_pre_process = (
                f"{arg + ' ' + (' '.join(argvs) if argvs else '')}".replace("\\t", "\t")
                .replace("“", '"')
                .replace("”", '"')
                .replace("\\\\", "\\")
                .replace("`", "")
            )

        with open("./BOT/_utils/_tmp/tmp.nl", "w", encoding="utf-8") as tmp_nl:
            for line in nolang_pre_process:
                tmp_nl.write(line)

        nolang_process = multiprocessing.Process(
            target=execute_processed_command,
            args=(nolang_pre_process, self.results, self.debug, self.post_val_name),
        )
        nolang_process.start()

        time.sleep(3)

        if nolang_process.is_alive():
            nolang_process.terminate()
            self.responses.append(("ERROR", "Valid runtime exceeded!"))
        else:
            self.responses.append(self.results[self.post_val_name])

        return self.responses
