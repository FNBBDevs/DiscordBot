
from .parser.statements import FunDeclaration
from .exception import *

class NolangType:
    def __init__(self) -> None:
        self.value = self

    def type_name(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return str(self)

class NolangInt(NolangType):
    def __init__(self, value: int) -> None:
        super().__init__()
        self.value = int(value) # Force to int

    def type_name(self) -> str:
        return 'int'

    def __str__(self) -> str:
        try:
            return str(self.value)

        except ValueError:
            return 'Too big to be stringified!'

class NolangFloat(NolangType):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = float(value) # Force to float

    def type_name(self) -> str:
        return 'float'

    def __str__(self) -> str:
        return str(self.value)

class NolangBool(NolangType):
    def __init__(self, value: bool) -> None:
        super().__init__()
        self.value = bool(value) # Force to bool

    def type_name(self) -> str:
        return 'bool'

    def __str__(self) -> str:
        return 'True' if self.value else 'False'

class NolangString(NolangType):
    def __init__(self, value: str) -> None:
        super().__init__()
        self.value = str(value) # Force to string

    def type_name(self) -> str:
        return 'string'

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f'\'{self}\''

class NolangArray(NolangType):
    def __init__(self, value: list) -> None:
        super().__init__()
        self.value = list(value) # Force to list

    def type_name(self) -> str:
        return 'array'

    def __len__(self) -> int:
        return len(self.value)

    def __getitem__(self, i):
        return self.value[i]

    def append(self, v):
        self.value.append(v)


class NolangNol(NolangType):
    def __str__(self) -> str:
        return 'nol'

# nol type can be treated as immutable
NOL = NolangNol()

class Interpreter: pass

class NolangCallable(NolangType):
    def arity(self) -> int:
        raise NotImplementedError()

    def __call__(self, interpreter: Interpreter, args: list[NolangType], line: int, file_name: str):
        raise NotImplementedError()

    def __str__(self) -> str:
        return f'<function built-in {self.__class__.__name__} {self.arity()}-ary>'

class NolangFunction(NolangCallable):
    def __init__(self, fun: FunDeclaration, env) -> None:
        super().__init__()
        self.fun = fun
        self.env = env

    def arity(self) -> int:
        return len(self.fun.params)

    def __call__(self, interpreter: Interpreter, args: list[NolangType], *_):
        from .astvisitors.interpreter import Environment
        env = Environment(self.env)

        # Binding arguments to parameters
        for param, arg in zip(self.fun.params, args):
            env.define(param, arg)

        try:
            interpreter._execute_body(self.fun.body, env)

        except Return as ret:
            return ret.value

    def __str__(self) -> str:
        return f'<function {self.fun.id.value} {self.arity()}-ary>'
