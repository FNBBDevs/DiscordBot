
from ..parser.expressions import *
from ..parser.statements import *

class ASTVisitor:
    """Interface that defines computations for all possible objects in a parse tree"""

    def explore(self, program: list[Statement]):
        raise NotImplementedError()

    def visit_vardecl(self, stmt: VarDeclaration):
        raise NotImplementedError()

    def visit_fundecl(self, stmt: FunDeclaration):
        raise NotImplementedError()

    def visit_ifstmt(self, stmt: IfStatement):
        raise NotImplementedError()

    def visit_whileloop(self, stmt: WhileStatement):
        raise NotImplementedError()

    def visit_bounceloop(self, stmt: BounceStatement):
        raise NotImplementedError()

    def visit_exprstmt(self, stmt: ExprStatement):
        raise NotImplementedError()

    def visit_return(self, stmt: ReturnStatement):
        raise NotImplementedError()

    def visit_identifier_access(self, expr: IDAccessorExpression):
        raise NotImplementedError()

    def visit_identifier_assign(self, expr: IDAssignExpression):
        raise NotImplementedError()

    def visit_index_access(self, expr: IndexAccessorExpression):
        raise NotImplementedError()

    def visit_index_assign(self, expr: IndexAssignExpression):
        raise NotImplementedError()

    def visit_call(self, expr: CallExpression):
        raise NotImplementedError()

    def visit_binexpr(self, expr: BinaryExpression):
        raise NotImplementedError()

    def visit_unexpr(self, expr: UnaryExpression):
        raise NotImplementedError()

    def visit_literal(self, expr: Literal):
        raise NotImplementedError()

    def visit_array_init(self, expr: ArrayInitializer):
        raise NotImplementedError()


