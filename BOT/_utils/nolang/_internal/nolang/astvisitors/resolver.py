
from .astvisitor import ASTVisitor
from ..parser.expressions import *
from ..parser.statements import *
from ..exception import *

class Resolver(ASTVisitor):
    """
    The variable resolver pass bakes the number of hops in the environment stack
    needed for any variable usage. This allows for a mutable environment that
    still respects static scoping.
    """

    def __init__(self):
        # Extend all variable expression nodes with an extra property indicating the distance
        # from the current environment in which it appears to the enclosing environment
        # referencing the variable.
        #
        # This applies only to local variables since globals are treated differently. This
        # allows global-level functions to have cyclic dependencies. Unfortunately this means
        # we can't check for undefined variables at compile time. :(
        #
        self.locals: dict[Expression, int] = dict()

        # A scope acts like a quasi-environment which indicates only whether variables
        # are initialized or not, we do not care about the actual values of these variables
        # during static analysis.
        #
        # Keep a stack of all the scopes that is pushed when entering a scope and popped
        # when exiting, local variables are declared/defined in the current stack.
        #
        self.scopes: list[dict[str, bool]] = [{}]

    def explore(self, program: list[Statement]) -> dict[Expression, int]:

        # We keep track of function and loop nesting to ensure some statements are used correctly!
        self.function_counter = 0
        self.loop_counter = 0

        try:
            for stmt in program:
                stmt.visit(self)

        except SemanticError as e:
            raise e

        return self.locals

    def visit_vardecl(self, stmt: VarDeclaration):
        # We declare it first but don't initialize
        self._declare(stmt.id)

        # We can now detect if user is trying to initialize with same variable name
        if stmt.has_initializer():
            stmt.init.visit(self)

        # Mark the variable as ready to use
        self._define(stmt.id)

    def visit_fundecl(self, stmt: FunDeclaration):
        # We eagerly define a function to allow recursion
        self._define(stmt.id)

        # Parameters appear in the same scope as the body
        self.scopes.append({})

        # Parameters are not initialized by the user, so they're safe to eagerly define too
        for param in stmt.params:
            self._define(param)

        # We are entering a function body, increment the counter
        self.function_counter += 1
        for stmt in stmt.body.stmts:
            stmt.visit(self)
        self.function_counter -= 1

        self.scopes.pop()

    def visit_ifstmt(self, stmt: IfStatement):
        stmt.cond.visit(self)
        self._visit_body(stmt.if_body)

        for cond, erm in stmt.erm_bodies:
            cond.visit(self)
            self._visit_body(erm)

        if stmt.has_hermph():
            self._visit_body(stmt.hermph_body)

    def visit_whileloop(self, stmt: WhileStatement):
        stmt.cond.visit(self)
        self._visit_body(stmt.while_body)

        if stmt.has_hermph():
            self._visit_body(stmt.hermph_body)

    def visit_bounceloop(self, stmt: BounceStatement):
        stmt.cond.visit(self)
        self._visit_body(stmt.bounce_body)

    def visit_exprstmt(self, stmt: ExprStatement):
        stmt.expr.visit(self)

    def visit_return(self, stmt: ReturnStatement):
        # This needs to be in a function!
        if self.function_counter == 0:
            raise UnexpectedReturnException(stmt.token.line, stmt.token.file_name)

        if stmt.has_value():
            stmt.value.visit(self)

    def visit_identifier_assign(self, expr: IDAssignExpression):
        self._resolve(expr, expr.id)
        expr.assign.visit(self)

    def visit_identifier_access(self, expr: IDAccessorExpression):
        scope = self.scopes[-1]

        # Catch user using unitialized variable in expression
        if len(scope) > 0 and expr.name() in scope and not scope[expr.name()]:
            raise UndefinedVariableUsage(expr.name(), expr.id.line, expr.id.file_name)

        self._resolve(expr, expr.id)

    def visit_index_access(self, expr: IndexAccessorExpression):
        expr.indexable.visit(self)
        expr.index.visit(self)

    def visit_index_assign(self, expr: IndexAssignExpression):
        expr.accessor.visit(self)
        expr.assign.visit(self)

    def visit_call(self, expr: CallExpression):
        expr.callee.visit(self)
        for arg in expr.args:
            arg.visit(self)

    def visit_binexpr(self, expr: BinaryExpression):
        expr.left.visit(self)
        expr.right.visit(self)

    def visit_unexpr(self, expr: UnaryExpression):
        expr.operand.visit(self)

    def visit_literal(self, _):
        return # Do nothing, literals are end-of-the-line.

    def visit_array_init(self, expr: ArrayInitializer):
        for element in expr.values:
            element.visit(self)

    ### Utilities ###

    def _visit_body(self, body: Body):
        self.scopes.append({})
        for stmt in body.stmts:
            stmt.visit(self)
        self.scopes.pop()

    def _declare(self, name: Token):
        if len(self.scopes) == 0:
            return

        scope = self.scopes[-1]
        scope[name.lexeme] = False

    def _define(self, name: Token):
        if len(self.scopes) == 0:
            return

        scope = self.scopes[-1]
        if name.lexeme in scope and scope[name.lexeme]:
            raise VariableRedefinitionException(name.lexeme, name.line, name.file_name)

        scope[name.lexeme] = True

    def _resolve(self, expr: Expression, name: Token):
        for i in range(len(self.scopes) - 1, -1, -1):
            if name.lexeme in self.scopes[i]:
                self.locals[expr] = len(self.scopes) - 1 - i
                return
