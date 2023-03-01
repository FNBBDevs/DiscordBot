import re
import sys
import time
import random
import contextlib
import python_weather
import multiprocessing
from io import StringIO
from _utils.restrictions import BRUHPY_RESTRICTIONS


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
    with stdoutIO() as s:
        try:
            exec(f"""\n{program}\n""")
            if s.getvalue() != '':
                results['POST'] = ('NORMAL', '[OUTPUT]\n'+s.getvalue())
            else:
                results[pvn] = ('INFO', '[INFO]: no output produced')
        except Exception as exception:
            error_response = ''
            line_num = None
            exception = str(exception)
            try:
                line_num = int(re.search('line (\d+)\)',
                               exception).groups()[0])
            except Exception as e:
                pass

            error_response += f"-[ERROR]: {exception}\n"
            if "bruhpy" in program:
                error_response += "it looks like 'bruhpy' was found in the program, did you type it twice?\n"
            if line_num:
                for i, line in enumerate(program.split("\n")):
                    if i == line_num - 2:
                        error_response += f"line {line_num}: '{line}'"

            results[pvn] = ('ERROR', error_response)
    if debug:
        print(f'leaving EPC. . .\nwith {pvn} val of {results[pvn]}')


class BruhPy:
    """
    Class responsible for processing of incoming python code from the /bruh command
    """

    def __init__(self, debug=False, post_val_name='POST'):
        self._debug = debug
        self._responses = []
        self._manager = multiprocessing.Manager()
        self._results = self._manager.dict()
        self._restictions = BRUHPY_RESTRICTIONS + ['exec', 'eval']
        self._post_val_name = post_val_name

    def run(self, arg, argvs):
        """
        Parse, prepare and execute the code passed in
        :param arg  : second word in the command
        :param argvs: every other word in the command
        """
        print(arg, argvs)
        if arg == '-s':
            pre_process = f"{' '.join(argvs) if argvs else ''}".replace('#', '\n').replace(
                '\\t', '\t').replace("“", "\"").replace("”", "\"").replace("\\\\", "\\")
        else:
            pre_process = f"{arg + ' ' + (' '.join(argvs) if argvs else '')}".replace(
                '#', '\n').replace('\\t', '\t').replace("“", "\"").replace("”", "\"").replace("\\\\", "\\")

        code_check = self._check(pre_process)

        if not code_check:
            self._responses.append(
                ("ERROR", "-[ERROR]: code did not pass preliminary inspection"))
            self._responses.append(
                ("INFO", "[INFO]: code did not execute, no output produced"))
            return self._responses

        # Execute the code
        process = multiprocessing.Process(target=execute_processed_command, args=(
            pre_process, self._results, self._debug, self._post_val_name))
        process.start()

        # sleep while code is running
        time.sleep(3)

        # timeout after the two seconds
        if process.is_alive():
            process.terminate()
            self._responses.append(
                ('ERROR', '-[ERROR]: valid runtime exceeded!'))
        else:
            self._responses.append(self._results[self._post_val_name])

        return self._responses

    def _check(self, program):
        checks = [
            r""".*(eval[(].*[)]).*""",
            r""".*(exec[(].*[)]).*""",
            r""".*(import .*).*""",
            r""".*(__import__[(].*[)]).*""",
            r""".*(globals[(][)]).*""",
            r""".*(getattr[(].*[)]).*""",
            r""".*(open[(].*[)]).*""",
            r""".*(while True:).*""",
        ]
        anti_checks = [
            r"""(.*(".*eval.*[(].*[)].*").*)|(.*('.*eval.*[(].*[)].*').*)""",
            r"""(.*(".*exec.*[(].*[)].*").*)|(.*('.*exec.*[(].*[)].*').*)""",
            r"""(.*(".*import .*").*)|(.*('.*import .*').*)""",
            r"""(.*(".*__import__[(].*[)].*").*)|(.*('__import__[(].*[)].*').*)""",
            r"""(.*(".*globals[(][)].*").*)|(.*('.*globals[(][)].*').*)""",
            r"""(.*(".*getattr[(].*[)].*").*)|(.*('.*getattr[(].*[)].*').*)""",
            r"""(.*(".*open[(].*[)].*").*)|(.*('.*open[(].*[)].*').*)""",
            r"""(.*(".*while True:.*").*)|(.*('.*while True:.*').*)""",
        ]
        no_antis = [
            r"""(.*(f".*{.*eval.*[(].*[)].*}.*").*)|(.*(f'.*{.*eval.*[(].*[)].*}.*').*)""",
            r"""(.*(f".*{.*exec.*[(].*[)].*}.*").*)|(.*(f'.*{.*exec.*[(].*[)].*}.*').*)""",
            r"""(.*(f".*{.*import .*}.*").*)|(.*(f'.*{.*import .*}.*').*)""",
            r"""(.*(f".*{.*__import__[(].*[)].*}.*").*)|(.*(f'.*{.*__import__[(].*[)].*}.*').*)""",
            r"""(.*(f".*{.*globals[(][)].*}.*").*)|(.*(f'.*{.*globals[(][)].*}.*').*)""",
            r"""(.*(f".*{.*getattr[(].*[)].*}.*").*)|(.*(f'.*{.*getattr[(].*[)].*}.*').*)""",
            r"""(.*(f".*{.*open[(].*[)].*}.*").*)|(.*(f'.*{.*open[(].*[)].*}.*').*)""",
            r"""(.*(f".*{.*while True:.*}.*").*)|(.*(f'.*{.*while True:.*}.*').*)""",
        ]
        no_antis_antis = [
            r"""(.*(f".*{.*'.*eval.*[(].*[)].*'.*}.*").*)|(.*(f'.*{.*".*eval.*[(].*[)].*".*}.*').*)""",
            r"""(.*(f".*{.*'.*exec.*[(].*[)].*'.*}.*").*)|(.*(f'.*{.*".*exec.*[(].*[)].*".*}.*').*)""",
            r"""(.*(f".*{.*'.*import .*'.*}.*").*)|(.*(f'.*{.*".*import .*".*}.*').*)""",
            r"""(.*(f".*{.*'.*__import__[(].*[)].*'.*}.*").*)|(.*(f'.*{.*".*__import__[(].*[)].*".*}.*').*)""",
            r"""(.*(f".*{.*'.*globals[(][)].*'.*}.*").*)|(.*(f'.*{.*".*globals[(][)].*".*}.*').*)""",
            r"""(.*(f".*{.*'.*getattr[(].*[)].*'.*}.*").*)|(.*(f'.*{.*".*getattr[(].*[)].*".*}.*').*)""",
            r"""(.*(f".*{.*'.*open[(].*[)].*'.*}.*").*)|(.*(f'.*{.*".*open[(].*[)].*".*}.*').*)""",
            r"""(.*(f".*{.*'.*while True:.*'.*}.*").*)|(.*(f'.*{.*".*while True:.*".*}.*').*)""",
        ]
            
        hits = []
        flag = False
        lines = program.split("\n")
        for line in lines:
            if flag:break
            line = line.split(";")
            for hidden_line in line:
                hidden_line = hidden_line.replace('"""', '"')
                for check, anti_check in list(zip(checks, anti_checks)):
                    if (s1 := re.search(check, hidden_line)) and not (s2 := re.search(anti_check, hidden_line)):
                        hits.append(s1)
                        hits.append(s2)
                        flag = True
                        break
                for restriction in self._restictions:
                    check_1 = r"""(.*=.*"""+restriction+r""".*)"""
                    check_2 = r"""(.*[(]"""+restriction+r"""[)].*)"""
                    anti_check_1 = r"""(.*=.*\""""+restriction+r""".*")"""
                    anti_check_1_2 = r"""(.*=.*'"""+restriction+r""".*')"""
                    anti_check_2 = r"""(.*[(]\""""+restriction+r"""[)].*")"""
                    anti_check_2_2 = r"""(.*[(]'"""+restriction+r"""[)].*')"""
                    if re.search(check_1, hidden_line) or re.search(check_2, hidden_line):
                        if not (re.search(anti_check_1, hidden_line) or re.search(anti_check_1_2, hidden_line) or re.search(anti_check_2, hidden_line) or re.search(anti_check_2_2, hidden_line)):
                            break
                        else:
                            hits.append(check_1)
                            hits.append(check_2)
                            flag = True
                            break
                for check, anti_check in list(zip(no_antis, no_antis_antis)):
                    if re.search(check, hidden_line) and not re.search(anti_check, hidden_line):
                        hits.append(check)
                        hits.append(anti_check)
                        flag = True
                        break
        print(hits)
        #return not flag
        return True if input("allow?: ").strip().lower() == "y" else False
