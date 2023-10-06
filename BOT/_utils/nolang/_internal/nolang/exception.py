
from .parser.expressions import Expression
from .lexer.token import Token

class NolangException(Exception):
    def __init__(self, line: int, file_name: str, *args: object) -> None:
        super().__init__(*args)
        self.line = line
        self.file_name = file_name

    def __str__(self) -> str:
        return f'A nolang exception has occured! {self._loc_to_str()}'

    def _loc_to_str(self) -> str:
        """Returns a string representation indicating location in source where exception was called"""
        return f'\'{self.file_name}\':{self.line}'

# Syntax Exceptions

class SyntaxError(NolangException):
    def __init__(self, *args: object, message: str = 'Syntax error') -> None:
        super().__init__(*args)
        self.message = message

    def __str__(self) -> str:
        return f'{self.message} in {self._loc_to_str()}'

class CharacterUnexpectedException(SyntaxError):
    def __init__(self, char: str, *args: object) -> None:
        super().__init__(*args)
        self.char = char

    def __str__(self) -> str:
        return f'Unexpected character: \'{self.char}\' in {self._loc_to_str()}'

class InconsistentIndentationException(SyntaxError):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

    def __str__(self) -> str:
        return f'Inconsistent indentation in {self._loc_to_str()}'

class UnterminatedStringException(SyntaxError):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

    def __str__(self) -> str:
        return f'Unterminated string literal in {self._loc_to_str()}'

class UnknownEscapeSequenceException(SyntaxError):
    def __init__(self, char: str, *args: object) -> None:
        super().__init__(*args)
        self.char = char

    def __str__(self) -> str:
        return f'Unknown escape sequence character: \'{self.char}\' in {self._loc_to_str()}'

class TokenUnexpectedException(SyntaxError):
    def __init__(self, token: Token, *args: object) -> None:
        super().__init__(token.line, token.file_name, *args)
        self.token = token

    def __str__(self) -> str:
        return f'Syntax error: \'{self.token}\' in {self._loc_to_str()}'

class EOFUnexpectedException(SyntaxError):
    def __init__(self, *args: object) -> None:
        super().__init__(-1, *args)

    def __str__(self) -> str:
        return f'Reached EOF unexpectedly while parsing {self.file_name}'

class InvalidBindingException(SyntaxError):
    def __init__(self, expr: Expression, *args: object) -> None:
        super().__init__(*args)
        self.expr = expr

    def __str__(self) -> str:
        return f'Cannot bind to non-lvalue expression \'{self.expr}\' {self._loc_to_str()}'

# Semantic Exceptions

class SemanticError(NolangException):
    def __init__(self, *args: object, message: str = 'Semantic error') -> None:
        super().__init__(*args)
        self.message = message

    def __str__(self) -> str:
        return f'{self.message} in {self._loc_to_str()}'

class UndefinedVariableUsage(SemanticError):
    def __init__(self, name: str, *args: object) -> None:
        super().__init__(*args)
        self.name = name

    def __str__(self) -> str:
        return f'Using variable \'{self.name}\' before initialization {self._loc_to_str()}'

class VariableRedefinitionException(SemanticError):
    def __init__(self, name: str, *args: object) -> None:
        super().__init__(*args)
        self.name = name

    def __str__(self) -> str:
        return f'{self.name} has already been defined in this scope {self._loc_to_str()}'

class UnexpectedReturnException(SemanticError):
    def __str__(self) -> str:
        return f'\'pay\' must be in function body {self._loc_to_str()}'

# Runtime Exceptions

class RuntimeException(NolangException):
    def __init__(self, *args: object, message: str = 'A nolang runtime exception has occured!') -> None:
        super().__init__(*args)
        self.message = message

    def __str__(self) -> str:
        return f'{self.message} {self._loc_to_str()}'

class InvalidTypeException(RuntimeException):
    def __init__(self, op: Token, operand, *args: object) -> None:
        super().__init__(op.line, op.file_name, *args)
        self.op = op
        self.operand = operand

    # TODO: Operator/Operands might not always apply for this type of exception
    def __str__(self) -> str:
        return f'Invalid operand \'{self.operand.type_name()}\' for operator \'{self.op}\' {self._loc_to_str()}'

class IncompatibleTypesException(RuntimeException):
    def __init__(self, op: Token, operand1, operand2, *args: object) -> None:
        super().__init__(op.line, op.file_name, *args)
        self.op = op
        self.operand1 = operand1
        self.operand2 = operand2

    def __str__(self) -> str:
        return f'Operator \'{self.op}\' on incompatible types {self._operands_str()} {self._loc_to_str()}'

    def _operands_str(self) -> str:
        return f'{self.operand1.type_name()} and {self.operand2.type_name()}'

class DivideByZeroException(RuntimeException):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

    def __str__(self) -> str:
        return f'Divide by zero {self._loc_to_str()}'

class VariableNotDefinedException(RuntimeException):
    def __init__(self, name: str, *args: object) -> None:
        super().__init__(*args)
        self.name = name

    def __str__(self) -> str:
        return f'{self.name} has not been defined in this scope {self._loc_to_str()}'

class NotCallableException(RuntimeException):
    def __init__(self, expr: Expression, *args: object) -> None:
        super().__init__(*args)
        self.expr = expr

    def __str__(self) -> str:
        return f'{self.expr} is not a callable object {self._loc_to_str()}'

class NotIndexableException(RuntimeException):
    def __init__(self, expr: Expression, *args: object) -> None:
        super().__init__(*args)
        self.expr = expr

    def __str__(self) -> str:
        return f'{self.expr} is not an indexable object {self._loc_to_str()}'

class InvalidArgumentsException(RuntimeException):
    def __init__(self, callee: Expression, arity: int, given: int, *args: object) -> None:
        super().__init__(*args)
        self.callee = callee
        self.arity = arity
        self.given = given

    def __str__(self) -> str:
        return f'{self.callee} requires {self.arity} arguments but {self.given} were provided {self._loc_to_str()}'

class OutOfBoundsException(RuntimeException):
    def __init__(self, indexable: Expression, index: int, size: int, *args: object) -> None:
        super().__init__(*args)
        self.indexable = indexable
        self.index = index
        self.size = size

    def __str__(self) -> str:
        return f'Index {self.index} out of bounds for {self.indexable} of size {self.size} {self._loc_to_str()}'

# Non-exceptional exceptions

class Return(Exception):
    def __init__(self, value) -> None:
        self.value = value
