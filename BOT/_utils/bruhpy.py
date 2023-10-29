""" File to process python code """
import multiprocessing
import re
import time


try:
    from bruhanimate import images as images
except:
    pass

from _utils.capstdout import stdoutIO
from _utils.marcus import Marcus
from _utils.restrictions import BRUHPY_RESTRICTIONS


def execute_processed_command(program, results, debug, pvn):
    """
    Function for executing the code and capturing any output to
    stdout
    :param program: the code to execute
    :param results: multiprocessing Manager Dict to store the stdout to
    :param debug  : to show debug messages
    """

    with stdoutIO() as sout:
        try:
            exec(f"""\n{program}\n""")
            if sout.getvalue() != "":
                results[pvn] = ("OUTPUT", sout.getvalue())
            else:
                results[pvn] = ("OUTPUT", "No output produced")
        except Exception as exception:
            error_response = ""
            line_num = None
            exception = str(exception)
            try:
                line_num = int(re.search(r"line (\d+)\)", exception).groups()[0])
            except Exception:
                pass

            error_response += f"{exception}\n"
            if "bruhpy" in program:
                error_response += (
                    "it looks like 'bruhpy' was found in the program, did you type it"
                    " twice?\n"
                )
            if line_num:
                for i, line in enumerate(program.split("\n")):
                    if i == line_num - 2:
                        error_response += f"line {line_num}: '{line}'"

            results[pvn] = ("ERROR", error_response)


class BruhPy:
    """
    Class responsible for processing of incoming python code from the /bruh command
    """

    def __init__(self, debug=False, post_val_name="POST"):
        self.debug = debug
        self.responses = []
        self.manager = multiprocessing.Manager()
        self.results = self.manager.dict()
        self.restictions = BRUHPY_RESTRICTIONS + ["exec", "eval"]
        self.post_val_name = post_val_name
        self.marcus = Marcus()

    def run(self, arg, argvs, user):
        """
        Parse, prepare and execute the code passed in
        :param arg  : second word in the command
        :param argvs: every other word in the command
        """
        if arg == "-s":
            bruhpy_pre_process = (
                f"{' '.join(argvs) if argvs else ''}".replace("\\t", "\t")
                .replace("“", '"')
                .replace("”", '"')
                .replace("\\\\", "\\")
                .replace("```py", "")
                .replace("`", "")
            )
            self.responses.append(("PY", f"# your_code.py\n{bruhpy_pre_process}"))
        else:
            bruhpy_pre_process = (
                f"{arg + ' ' + (' '.join(argvs) if argvs else '')}".replace("\\t", "\t")
                .replace("“", '"')
                .replace("”", '"')
                .replace("\\\\", "\\")
                .replace("```py", "")
                .replace("`", "")
            )

        code_check = self.marcus.erm__hey_marcus__can_you_check_this_code_out(
            bruhpy_pre_process, user
        )

        code_check = False

        if not code_check:
            self.responses += [
                ("ERROR", "FNBB is not executing python commands for the time being."),
                ("INFO", "Code did not execute, no output produced"),
            ]
            return self.responses

        # Execute the code
        bruhpy_process = multiprocessing.Process(
            target=execute_processed_command,
            args=(bruhpy_pre_process, self.results, self.debug, self.post_val_name),
        )
        bruhpy_process.start()

        # sleep while code is running
        time.sleep(5)

        # timeout after the two seconds
        if bruhpy_process.is_alive():
            bruhpy_process.terminate()
            self.responses.append(("ERROR", "Valid runtime exceeded!"))
        else:
            self.responses.append(self.results[self.post_val_name])

        return self.responses
