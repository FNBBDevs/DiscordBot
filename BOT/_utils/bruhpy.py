import sys
import time
import contextlib
import multiprocessing
from io import StringIO

@contextlib.contextmanager
def stdoutIO(stdout=None):
    old = sys.stdout
    if stdout is None:
        stdout = StringIO()
    sys.stdout = stdout
    yield stdout
    sys.stdout = old

def execute_processed_command(program, results, debug):
    if debug: print(f"\nEPC called with\n{program}\n")
    with stdoutIO() as s:
        try:
            exec(f"""\n{program}\n""")
            results['POST'] = [True, s.getvalue()]
        except Exception as exception:
            results['POST'] = [True, str(exception)]

class BruhPy:
    def __init__(self, debug=False):
        self._debug=debug
        self._response = ''
        self._manager = multiprocessing.Manager()
        self._results = self._manager.dict()
    
    def run(self, arg, argvs):
        pre_process = f"{arg + ' ' + (' '.join(argvs) if argvs else '')}".replace('#', '\n').replace('\\t', '\t')
        process = multiprocessing.Process(target=execute_processed_command, args=(pre_process, self._results, self._debug))
        process.start()
        time.sleep(2)

        if process.is_alive():
            process.terminate()
            self._response += 'frick you bro!'
        else:
            print(self._results)
            self._response += self._results['POST'][1]
        return self._response
