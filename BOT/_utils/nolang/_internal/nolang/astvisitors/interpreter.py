
from typing_extensions import override
from .astvisitor import ASTVisitor
from .resolver import Resolver

from ..parser.expressions import *
from ..parser.statements import *
from ..lexer.token import Tokens
from ..types import *

from ..runtime import RUNTIME_GLOBALS
from ..exception import *

class Environment: pass
class Environment:
    """
    The environment consists of a mapping or binding of names to values.
    It also contains a reference to the parent or "enclosing" environment.

    We could have an intermediary known as the 'store' which acts as main
    memory and stores values at addresses while the environment stores
    name and address bindings.

    Or we could have the environment directly map to the values using a
    hashmap structure.
    """

    def __init__(self, enclosing: Environment = None) -> None:
        self.enclosing: Environment = enclosing
        self.values: dict[str, NolangType] = {}

    def define(self, id: Token, value: NolangType):
        # Insert new entry in the dictionary in the most recent scope always!
        self.values[id.lexeme] = value

    def get_at(self, id: Token, distance: int) -> NolangType:
        return self._env_at(distance).values[id.lexeme]

    def get(self, id: Token) -> NolangType:
        name = id.lexeme

        if name not in self.values:
            raise VariableNotDefinedException(name, id.line, id.file_name)

        return self.values[name]

    def assign_at(self, id: Token, new_value: NolangType, distance: int) -> None:
        self._env_at(distance).values[id.lexeme] = new_value

    def assign(self, id: Token, new_value) -> None:
        name = id.lexeme

        if name not in self.values:
            raise VariableNotDefinedException(name, id.line, id.file_name)

        self.values[name] = new_value

    ### Utilities ###

    def _env_at(self, distance: int) -> Environment:
        environment = self
        for _ in range(distance):
            environment = environment.enclosing

        return environment

class Interpreter(ASTVisitor):
    def __init__(self):
        self.globals = Environment()
        self.resolver = Resolver()

        # Initialize globals with runtime library
        self.globals.values.update(RUNTIME_GLOBALS)

        self.environment = self.globals

    def explore(self, program: list[Statement]):
        try:
            self.bindings = self.resolver.explore(program)

            for stmt in program:
                stmt.visit(self)

        except (RuntimeException, SemanticError) as e:
            raise e

    def visit_vardecl(self, stmt: VarDeclaration):
        value = None
        if stmt.has_initializer():
            value = stmt.init.visit(self)

        self.environment.define(stmt.id, value)

    def visit_fundecl(self, stmt: FunDeclaration):
        # We give the function the current environment when DECLARED.
        # That way it can use variables outside of its scope
        fun = NolangFunction(stmt, self.environment)

        self.environment.define(stmt.id, fun)

    def visit_ifstmt(self, stmt: IfStatement):
        cond = stmt.cond.visit(self)
        if self._to_truthy(cond):
            self._execute_body(stmt.if_body)
            return

        for cond, body in stmt.erm_bodies:
            if self._to_truthy(cond.visit(self)):
                self._execute_body(body)
                return

        if stmt.has_hermph():
            self._execute_body(stmt.hermph_body)

    def visit_whileloop(self, stmt: WhileStatement):
        while self._to_truthy(stmt.cond.visit(self)):
            self._execute_body(stmt.while_body)

        else:
            if stmt.has_hermph():
                self._execute_body(stmt.hermph_body)

    def visit_bounceloop(self, stmt: BounceStatement):
        self._execute_body(stmt.bounce_body)
        while self._to_truthy(stmt.cond.visit(self)):
            self._execute_body(stmt.bounce_body)

    def visit_exprstmt(self, stmt: ExprStatement):
        stmt.expr.visit(self)

    def visit_return(self, stmt: ReturnStatement):
        value = NOL # NOTE: Return NOL if there is no value!
        if stmt.has_value():
            value = stmt.value.visit(self)

        raise Return(value)

    def visit_identifier_assign(self, expr: IDAssignExpression):
        value = expr.assign.visit(self)
        distance = self.bindings.get(expr)

        # We know which environment local variables are in
        if distance is not None:
            self.environment.assign_at(expr.id, value, distance)

        # Distance is None, this might be a global variable
        else:
            self.globals.assign(expr.id, value)

        return value

    def visit_identifier_access(self, expr: IDAccessorExpression):
        distance = self.bindings.get(expr)

        # We know which environment local variables are in
        if distance is not None:
            return self.environment.get_at(expr.id, distance)

        # Distance is None, this might be a global variable
        else:
            return self.globals.get(expr.id)

    def visit_index_access(self, expr: IndexAccessorExpression):
        indexable, index = self._try_get_indexable_and_index(expr)

        return indexable[index]

    def visit_index_assign(self, expr: IndexAssignExpression):
        indexable, index = self._try_get_indexable_and_index(expr.accessor)

        indexable[index] = expr.assign.visit(self)

        return indexable[index]

    def visit_call(self, expr: CallExpression):
        callee = expr.callee.visit(self)

        # Static initialization of arguments
        args = [ arg.visit(self) for arg in expr.args ]

        if not Interpreter._is_type(callee, NolangCallable):
            raise NotCallableException(expr.callee, expr.paren.line, expr.paren.file_name)

        callee: NolangCallable
        arity = callee.arity()
        given = len(args)

        if arity != given:
            raise InvalidArgumentsException(expr.callee, arity, given, expr.paren.line, expr.paren.file_name)

        result = callee(self, args, expr.paren.line, expr.paren.file_name)
        return result if result else NOL

    def visit_binexpr(self, expr: BinaryExpression):
        val1: NolangType = expr.left.visit(self)

        # Make OR and AND operators short-circuited, we DO NOT evaluate RHS unless we have to
        match expr.op.type_id:
            case Tokens.OR:
                return NolangBool(Interpreter._to_truthy(val1) \
                    or Interpreter._to_truthy(expr.right.visit(self)))

            case Tokens.AND:
                return NolangBool(Interpreter._to_truthy(val1) \
                   and Interpreter._to_truthy(expr.right.visit(self)))

        # All other operators will need RHS evaluated to work
        val2: NolangType = expr.right.visit(self)

        match expr.op.type_id:
            case Tokens.EQUAL: return NolangBool(val1.value == val2.value)
            case Tokens.NEQUAL: return NolangBool(val1.value != val2.value)
            case Tokens.LESS_THAN:
                Interpreter._check_ordering(val1, val2, expr.op)
                return NolangBool(val1.value < val2.value)

            case Tokens.GREATER_THAN:
                Interpreter._check_ordering(val1, val2, expr.op)
                return NolangBool(val1.value > val2.value)

            case Tokens.LESS_THAN_EQ:
                Interpreter._check_ordering(val1, val2, expr.op)
                return NolangBool(val1.value <= val2.value)

            case Tokens.GREATER_THAN_EQ:
                Interpreter._check_ordering(val1, val2, expr.op)
                return NolangBool(val1.value >= val2.value)

            case Tokens.PLUS:
                # NOTE: We use the 'safe' to-string functions which will catch any python exceptions that may be thrown
                if Interpreter._is_type(val1, NolangString) \
                or Interpreter._is_type(val2, NolangString):
                    return NolangString(str(val1) + str(val2))

                typ = Interpreter._check_numerics(val1, val2, expr.op)
                return typ(val1.value + val2.value)

            case Tokens.MINUS:
                typ = Interpreter._check_numerics(val1, val2, expr.op)
                return typ(val1.value - val2.value)

            case Tokens.STAR:
                typ = Interpreter._check_numerics(val1, val2, expr.op)
                return typ(val1.value * val2.value)

            case Tokens.SLASH:
                typ = Interpreter._check_numerics(val1, val2, expr.op)
                if val2.value == 0:
                    raise DivideByZeroException(expr.op.line, expr.op.file_name)
                return typ(val1.value / val2.value)

            case Tokens.PERCENT:
                Interpreter._check_types(val1, val2, expr.op, NolangInt)
                return NolangInt(val1.value % val2.value)

            case Tokens.EXP:
                typ = Interpreter._check_numerics(val1, val2, expr.op)
                return typ(val1.value ** val2.value)

        # This should never happen in a completed implementation, do it for debugging purposes
        raise Exception(f'Failed to interpret expression: {expr}')

    def visit_unexpr(self, expr: UnaryExpression):
        value: NolangType = expr.operand.visit(self)

        match expr.op.type_id:
            case Tokens.NOT:
                return NolangBool(not Interpreter._to_truthy(value))
            case Tokens.MINUS:
                typ = self._check_numeric(value, expr.op)
                return typ(-value.value)
            case Tokens.PLUS:
                typ = self._check_numeric(value, expr.op)
                return typ(+value.value)

        # This should never happen in a completed implementation, do it for debugging purposes
        raise Exception(f'Failed to interpret expression: {expr}')

    def visit_literal(self, expr: Literal):
        return expr.value()

    def visit_array_init(self, expr: ArrayInitializer):
        return NolangArray([ element.visit(self) for element in expr.values ])

    ### Utilities ###

    def _execute_body(self, body: Body, new_env: Environment = None):
        previous_env = self.environment

        # Create a new environment
        self.environment = Environment(previous_env) if new_env is None else new_env

        try:
            # Execute all the statements
            for stmt in body.stmts:
                stmt.visit(self)

        finally:
            # Always restore the previous environment
            self.environment = previous_env

    def _try_get_indexable_and_index(self, expr: IndexAccessorExpression):
        indexable = expr.indexable.visit(self)

        if not Interpreter._is_type(indexable, NolangArray):
            raise NotIndexableException(expr.indexable, expr.bracket.line, expr.bracket.file_name)

        indexable: NolangArray
        index = expr.index.visit(self)

        Interpreter._check_type(index, expr.bracket, NolangInt)
        index: NolangInt

        if index.value < 0 or index.value > len(indexable.value) - 1:
            raise OutOfBoundsException(expr.indexable, index.value, len(indexable.value), expr.bracket.line, expr.bracket.file_name)

        return (indexable.value, index.value)

    @staticmethod
    def _is_type(val, *types: type):
        return Interpreter._which_type(val, types) is not None

    @staticmethod
    def _which_type(val, *types: type):
        for t in types:
            if isinstance(val, t):
                return t

    @staticmethod
    def _to_truthy(val: NolangType):
        """ In Nolang, nol is False, False (nolang) is False (python), 0 and 0.0 are False, and everything else is True"""
        if Interpreter._is_type(val, NolangNol): return False
        if Interpreter._is_type(val, NolangBool): return val.value
        if Interpreter._is_type(val, NolangInt, NolangFloat): return val.value != 0
        return True

    @staticmethod
    def _check_numeric(val, op: Token):
        """Checks if the value is of numeric type"""
        return Interpreter._check_type(val, op, NolangInt, NolangFloat)

    @staticmethod
    def _check_numerics(val1, val2, op: Token):
        """Checks if both values are of numeric type"""
        typ1, typ2 = Interpreter._check_types(val1, val2, op, NolangInt, NolangFloat)

        # Integers are promoted to float if second operand is float
        if typ1 is NolangFloat or typ2 is NolangFloat:
            return NolangFloat

        return NolangInt

    @staticmethod
    def _check_ordering(val1, val2, op: Token):
        """Checks if there is an ordering between the two input values"""
        typ1, typ2 = Interpreter._check_types(val1, val2, op, NolangInt, NolangFloat, NolangString)

        if Interpreter._is_type(val1, NolangString) and not Interpreter._is_type(val2, NolangString) \
        or Interpreter._is_type(val2, NolangString) and not Interpreter._is_type(val1, NolangString):
            raise IncompatibleTypesException(op, val1, val2)

        return typ1, typ2

    @staticmethod
    def _check_type(val, op, *types: type):
        """Checks if value is any of the provided types and returns it"""
        typ = Interpreter._which_type(val, *types)

        if typ is None:
            raise InvalidTypeException(op, val)

        return typ

    @staticmethod
    def _check_types(val1, val2, op, *types: type):
        """Checks if both values are any of the provided types"""
        typ1 = Interpreter._check_type(val1, op, *types)
        typ2 = Interpreter._check_type(val2, op, *types)
        return typ1, typ2
