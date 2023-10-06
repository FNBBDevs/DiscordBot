
from .types import *
from .parser.expressions import *
from .parser.statements import *
from .exception import *

import time
import random
import math
from bruhcolor import bruhcolored as colored

# Implementations of runtime library objects

class Nolout(NolangCallable):
    def arity(self) -> int:
        return 1

    def __call__(self, _, args: list[NolangType], *__):
        print(args[0])
        return NOL

class Nolin(NolangCallable):
    def arity(self) -> int:
        return 1

    def __call__(self, _, args: list[NolangType], *__):
        return NolangString(input(args[0]))

class Time(NolangCallable):
    def arity(self) -> int:
        return 0

    def __call__(self, *_):
        return NolangInt(round(time.time() * 1000))

class Random(NolangCallable):
    def arity(self) -> int:
        return 0

    def __call__(self, *_):
        return NolangFloat(random.random())

class Int(NolangCallable):
    def arity(self) -> int:
        return 1

    def __call__(self, _, args: list[NolangType], line: int, file_name: str):
        try:
            return NolangInt(args[0].value)

        except ValueError:
            raise RuntimeException(line, file_name, message=f'Cannot convert {args[0]} to int')

class ColoredOut(NolangCallable):
    def arity(self) -> int:
        return 3
    
    def __call__(self, _, args: list[NolangType], line: int, file_name: str):
        try:
            print(colored(args[0], color=args[1], on_color=args[2]))
            return NOL

        except ValueError:
            raise RuntimeException(line, file_name, message=f'YUP')
        
class Colored(NolangCallable):
    def arity(self) -> int:
        return 3
    
    def __call__(self, _, args: list[NolangType], line: int, file_name: str):
        try:
            return colored(args[0], color=args[1], on_color=args[2])
        
        except ValueError:
            raise RuntimeException(line, file_name, message=f'YUP')
 
class Sleep(NolangCallable):
    def arity(self) -> int:
        return 1
    
    def __call__(self, _, args: list[NolangType], line: int, file_name: str):
        try:
            time.sleep(args[0].value)
            return NOL
        
        except TypeError:
            raise RuntimeException(line, file_name, message=f'Invalid type {args[0].type_name()}')

class Float(NolangCallable):
    def arity(self) -> int:
        return 1

    def __call__(self, _, args: list[NolangType], line: int, file_name: str):
        try:
            return NolangFloat(args[0].value)

        except ValueError:
            raise RuntimeException(line, file_name, message=f'Cannot convert {args[0]} to float')

class RoundDown(NolangCallable):
    def arity(self) -> int:
        return 1

    def __call__(self, _, args: list[NolangType], line: int, file_name: str):
        try:
            return NolangInt(math.floor(args[0].value))

        except TypeError:
            raise RuntimeException(line, file_name, message=f'Invalid type {args[0].type_name()}')

class RoundUp(NolangCallable):
    def arity(self) -> int:
        return 1

    def __call__(self, _, args: list[NolangType], line: int, file_name: str):
        try:
            return NolangInt(math.ceil(args[0].value))

        except TypeError:
            raise RuntimeException(line, file_name, message=f'Invalid type {args[0].type_name()}')

# Global runtime, this should be immutable!

RUNTIME_GLOBALS: dict[str, NolangType] = \
{
    'nolout':     Nolout(),
    'nolin':      Nolin(),
    'time':       Time(),
    'random':     Random(),
    'int':        Int(),
    'float':      Float(),
    'roundup':    RoundUp(),
    'rounddown':  RoundDown(),
    'coloredout': ColoredOut(),
    'colored':    Colored(),
    'sleep':      Sleep(),
}
