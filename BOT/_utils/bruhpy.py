import re
import sys
import time
import random
import contextlib
import python_weather
import multiprocessing
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
    if debug: print(f"\nEPC called with\n{program}\n")
    with stdoutIO() as s:
        try:
            exec(f"""\n{program}\n""")
            if s.getvalue() != '': results['POST'] = ('NORMAL', '[OUTPUT]\n'+s.getvalue())
            else:results[pvn] =  ('INFO', '[INFO]: no output produced')
        except Exception as exception:
            error_response = ''
            line_num = None
            exception = str(exception)
            try:
                line_num = int(re.search('line (\d+)\)', exception).groups()[0])
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
    if debug: print(f'leaving EPC. . .\nwith {pvn} val of {results[pvn]}')

class BruhPy:
    """
    Class responsible for processing of incoming python code from the /bruh command
    """
    def __init__(self, debug=False, post_val_name='POST'):
        self._debug         = debug
        self._responses     = [] 
        self._manager       = multiprocessing.Manager()
        self._results       = self._manager.dict()
        self._restictions   = ['webbrowser']
        self._post_val_name = post_val_name
    
    def run(self, arg, argvs):
        """
        Parse, prepare and execute the code passed in
        :param arg  : second word in the command
        :param argvs: every other word in the command
        """
        print(arg, argvs)
        if arg == '-s':
            pre_process = f"{' '.join(argvs) if argvs else ''}".replace('#', '\n').replace('\\t', '\t').replace("“", "\"").replace("”", "\"").replace("\\\\", "\\")
            self._responses.append(('PY', pre_process))
        else: pre_process = f"{arg + ' ' + (' '.join(argvs) if argvs else '')}".replace('#', '\n').replace('\\t', '\t').replace("“", "\"").replace("”", "\"").replace("\\\\", "\\")

        code_check = self._check(pre_process)

        if not code_check:
            self._responses.append(("ERROR", "-[ERROR]: code did not pass preliminary inspection"))
            self._responses.append(("INFO", "[INFO]: code did not execute, not output produced"))
            return self._responses

        # Execute the code 
        process = multiprocessing.Process(target=execute_processed_command, args=(pre_process, self._results, self._debug, self._post_val_name))
        process.start()

        # sleep while code is running
        time.sleep(1)

        # timeout after the two seconds
        if process.is_alive():
            process.terminate()
            self._responses.append(('ERROR', '-[ERROR]: valid runtime exceeded!'))
        else: self._responses.append(self._results[self._post_val_name])

        return self._responses

    def _check(self, program):
        for restriction in self._restictions:
            if re.search(restriction, program):
                return False
        return True