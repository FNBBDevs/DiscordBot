
from .expressions import Expression
from ..lexer.lexer import Token

# Forward declare the visitor interface
class ASTVisitor: pass

class Statement:
    def visit(self, _: ASTVisitor):
        """Pure virtual function that calls respective handler for this type in the visitor"""
        raise NotImplementedError

class Body:
    """
    Bodies/Blocks are not statements in Nolang but rather extensions for compound
    statements such as 'if', 'while', 'for' etc. It is still useful to separately
    parse them, as they show up frequently (maybe even more than once) in different rules.
    """
    def __init__(self, stmts: list[Statement]) -> None:
        self.stmts = stmts

class VarDeclaration(Statement):
    def __init__(self, id: Token, init: Expression = None) -> None:
        super().__init__()
        self.id = id
        self.init = init

    def visit(self, visitor: ASTVisitor):
        return visitor.visit_vardecl(self)

    def has_initializer(self) -> bool:
        return self.init is not None

    def __repr__(self) -> str:
        return f'no {self.id.value}{f" = {self.init}" if self.has_initializer() else ""}'

class FunDeclaration(Statement):
    def __init__(self, id: Token, params: list[Token], body: Body) -> None:
        super().__init__()
        self.id = id
        self.params = params
        self.body = body

    def visit(self, visitor: ASTVisitor):
        return visitor.visit_fundecl(self)

    def __repr__(self) -> str:
        return f'greg {self.id.value}({self.params})'

class IfStatement(Statement):
    def __init__(self, cond: Expression, if_body: Body, erm_bodies: list[tuple[Expression, Body]], hermph_body: Body) -> None:
        super().__init__()
        self.cond = cond
        self.if_body = if_body
        self.erm_bodies = erm_bodies
        self.hermph_body = hermph_body

    def visit(self, visitor: ASTVisitor):
        return visitor.visit_ifstmt(self)

    def has_hermph(self) -> bool:
        return self.hermph_body is not None

class WhileStatement(Statement):
    def __init__(self, cond: Expression, while_body: Body, hermph_body: Body) -> None:
        super().__init__()
        self.cond = cond
        self.while_body = while_body
        self.hermph_body = hermph_body

    def visit(self, visitor: ASTVisitor):
        return visitor.visit_whileloop(self)

    def has_hermph(self) -> bool:
        return self.hermph_body is not None

class BounceStatement(Statement):
    def __init__(self, bounce_body: Body, cond: Expression) -> None:
        super().__init__()
        self.cond = cond
        self.bounce_body = bounce_body

    def visit(self, visitor: ASTVisitor):
        return visitor.visit_bounceloop(self)

class ExprStatement(Statement):
    """Statement that may cause a side-effect and evaluate"""

    def __init__(self, expr: Expression) -> None:
        super().__init__()
        self.expr = expr

    def visit(self, visitor: ASTVisitor):
        return visitor.visit_exprstmt(self)

    def __repr__(self) -> str:
        return repr(self.expr)

class ReturnStatement(Statement):
    def __init__(self, token: Token, value: Expression) -> None:
        super().__init__()
        self.token = token
        self.value = value

    def visit(self, visitor: ASTVisitor):
        return visitor.visit_return(self)

    def has_value(self) -> bool:
        return self.value is not None

    def __repr__(self) -> str:
        return f'pay {f"{self.value}" if self.has_value() else ""}'
