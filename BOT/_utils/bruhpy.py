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
            if s.getvalue() != '': results['POST'] = ('NORMAL', '[OUTPUT]\n'+s.getvalue())
            else:results['POST'] =  ('INFO', '[INFO]: no output produced')
        except Exception as exception:
            results['POST'] = ('ERROR', "-[ERROR]: " + str(exception))

class BruhPy:
    def __init__(self, debug=False):
        self._debug     = debug
        self._responses = [] 
        self._manager   = multiprocessing.Manager()
        self._results   = self._manager.dict()
    
    def run(self, arg, argvs):
        if arg == '-s':
            pre_process = f"{' '.join(argvs) if argvs else ''}".replace('#', '\n').replace('\\t', '\t').replace("“", "\"").replace("”", "\"")
            self._responses.append(('PY', pre_process))
        else: pre_process = f"{arg + ' ' + (' '.join(argvs) if argvs else '')}".replace('#', '\n').replace('\\t', '\t').replace("“", "\"").replace("”", "\"")
        process = multiprocessing.Process(target=execute_processed_command, args=(pre_process, self._results, self._debug))
        process.start()
        time.sleep(2)

        if process.is_alive():
            process.terminate()
            self._responses.append(('ERROR', '-[ERROR]: valid runtime exceeded!'))
        else: self._responses.append(self._results['POST'])

        return self._responses
