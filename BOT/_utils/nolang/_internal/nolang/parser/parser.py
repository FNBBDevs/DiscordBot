
from ..lexer.token import Token
from ..lexer.token import Tokens
from .expressions import *
from .statements import *

from ..exception import *

MAX_PARAMETERS = 255

class Parser:
    def parse(self, tokens: list[Token], filename: str) -> list[Statement]:
        self.tokens = tokens
        self.filename = filename
        self.exceptions = []

        """Current token to be examined in the current production rule"""
        self.current: int = 0
        statements: list[Statement] = []

        while not self._next_is(Tokens.EOF) and not self._at_end():
            statements.append(self.statement())

        if len(self.exceptions) > 0:
            raise ExceptionGroup('Parser exceptions', self.exceptions)

        return statements

    def statement(self) -> Statement:
        try:
            if self._next_is(Tokens.NO): return self.var_decl()
            if self._next_is(Tokens.GREG): return self.fun_decl()

            return self.cmpd_stmt()

        except NolangException as e:
            self.exceptions.append(e)
            self._next_statment()

    def var_decl(self) -> Statement:
        id = self._consume(Tokens.IDENTIFIER)

        init = None
        if self._next_is(Tokens.ASSIGN):
            init = self.expression()

        self._consume(Tokens.NEWLINE)
        return VarDeclaration(id, init)

    def fun_decl(self) -> Statement:
        id = self._consume(Tokens.IDENTIFIER)
        self._consume(Tokens.L_PARENTHESIS)

        params: list[Token] = []
        if self._peek().type_id != Tokens.R_PARENTHESIS:
            params.append(self._consume(Tokens.IDENTIFIER))

            while self._next_is(Tokens.COMMA):
                token = self._consume(Tokens.IDENTIFIER)

                if len(params) > MAX_PARAMETERS:
                    self.exceptions.append(SyntaxError(token.line, token.file_name, message=f'Too many parameters for function, MAX: {MAX_PARAMETERS}!'))
                    continue

                params.append(token)

        self._consume(Tokens.R_PARENTHESIS)
        self._consume(Tokens.NEWLINE)
        return FunDeclaration(id, params, self._body())

    def cmpd_stmt(self) -> Statement:
        if self._next_is(Tokens.IF): return self.if_stmt()
        if self._next_is(Tokens.WHILE): return self.while_loop()
        if self._next_is(Tokens.BOUNCE): return self.bounce_loop()

        return self.std_stmt()

    def if_stmt(self) -> Statement:
        cond = self.expression()
        self._consume(Tokens.NEWLINE)
        if_body = self._body()
        elif_bodies = []
        else_body = None

        while self._next_is(Tokens.ERM):
            elif_cond = self.expression()
            self._consume(Tokens.NEWLINE)
            elif_bodies.append(( elif_cond, self._body() ))

        if self._next_is(Tokens.HERMPH):
            self._consume(Tokens.NEWLINE)
            else_body = self._body()

        return IfStatement(cond, if_body, elif_bodies, else_body)

    def while_loop(self) -> Statement:
        cond = self.expression()
        self._consume(Tokens.NEWLINE)
        while_body = self._body()
        else_body = None

        if self._next_is(Tokens.HERMPH):
            self._consume(Tokens.NEWLINE)
            else_body = self._body()

        return WhileStatement(cond, while_body, else_body)

    def bounce_loop(self) -> Statement:
        self._consume(Tokens.NEWLINE)
        bounce_body = self._body()
        self._consume(Tokens.WHILE)
        cond = self.expression()
        self._consume(Tokens.NEWLINE)

        return BounceStatement(bounce_body, cond)

    def std_stmt(self) -> Statement:
        if self._next_is(Tokens.RETURN):
            stmt = self.return_stmt()

        else:
            stmt = self.expr_stmt()

        self._consume(Tokens.NEWLINE)
        return stmt

    def expr_stmt(self) -> Statement:
        return ExprStatement(self.expression())

    def return_stmt(self) -> Statement:
        value = None
        token = self._previous()
        if self._peek().type_id != Tokens.NEWLINE:
            value = self.expression()

        return ReturnStatement(token, value)

    def expression(self) -> Expression:
        return self.assign_expr()

    def assign_expr(self) -> Expression:
        lhs = self.or_expr()

        if self._next_is(Tokens.ASSIGN):

            if isinstance(lhs, IDAccessorExpression):
                # We know that we have an identifier now
                lhs: IDAccessorExpression
                return IDAssignExpression(lhs.id, self.assign_expr())

            if isinstance(lhs, IndexAccessorExpression):
                # We know that we have an index accessor now
                lhs: IndexAccessorExpression
                return IndexAssignExpression(lhs, self.assign_expr())

            # Must be a valid lvalue target!
            assign: Token = self._previous()
            raise InvalidBindingException(lhs, assign.line, assign.file_name)

        return lhs

    def or_expr(self) -> Expression:
        expr: Expression = self.and_expr()

        while self._next_is(Tokens.OR):
            op: Token = self._previous()
            right: Expression = self.and_expr()
            expr = BinaryExpression(expr, right, op)

        return expr

    def and_expr(self) -> Expression:
        expr: Expression = self.not_expr()

        while self._next_is(Tokens.AND):
            op: Token = self._previous()
            right: Expression = self.not_expr()
            expr = BinaryExpression(expr, right, op)

        return expr

    def not_expr(self) -> Expression:
        if self._next_is(Tokens.NOT):
            op: Token = self._previous()
            right: Expression = self.not_expr()
            return UnaryExpression(right, op)

        return self.eq_expr()

    def eq_expr(self) -> Expression:
        expr: Expression = self.rel_expr()

        while self._next_is(Tokens.EQUAL, Tokens.NEQUAL):
            op: Token = self._previous()
            right: Expression = self.not_expr()
            expr = BinaryExpression(expr, right, op)

        return expr

    def rel_expr(self) -> Expression:
        expr: Expression = self.add_expr()

        while self._next_is(Tokens.LESS_THAN, Tokens.GREATER_THAN, Tokens.LESS_THAN_EQ, Tokens.GREATER_THAN_EQ):
            op: Token = self._previous()
            right: Expression = self.add_expr()
            expr = BinaryExpression(expr, right, op)

        return expr

    def add_expr(self) -> Expression:
        expr: Expression = self.mul_expr()

        while self._next_is(Tokens.PLUS, Tokens.MINUS):
            op: Token = self._previous()
            right: Expression = self.mul_expr()
            expr = BinaryExpression(expr, right, op)

        return expr

    def mul_expr(self) -> Expression:
        expr: Expression = self.sign_expr()

        while self._next_is(Tokens.STAR, Tokens.SLASH, Tokens.PERCENT):
            op: Token = self._previous()
            right: Expression = self.sign_expr()
            expr = BinaryExpression(expr, right, op)

        return expr

    def sign_expr(self) -> Expression:
        if self._next_is(Tokens.PLUS, Tokens.MINUS):
            op: Token = self._previous()
            right: Expression = self.sign_expr()
            return UnaryExpression(right, op)

        return self.exp_expr()

    def exp_expr(self) -> Expression:
        expr: Expression = self.primary()

        if self._next_is(Tokens.EXP):
            op: Token = self._previous()
            right: Expression = self.sign_expr()
            expr = BinaryExpression(expr, right, op)

        return expr

    def primary(self) -> Expression:
        expr: Expression = self.atom()

        while True:
            if self._next_is(Tokens.L_PARENTHESIS):
                expr = self._finish_call(expr)

            elif self._next_is(Tokens.L_BRACKET):
                index: Expression = self.expression()
                bracket = self._consume(Tokens.R_BRACKET)
                expr = IndexAccessorExpression(expr, bracket, index)

            else: break

        return expr

    def atom(self) -> Expression:
        if self._next_is(
            Tokens.INT_LITERAL,
            Tokens.FLOAT_LITERAL,
            Tokens.STR_LITERAL,
            Tokens.TRUE,
            Tokens.FALSE,
            Tokens.NOL):
            return Literal(self._previous())

        if self._next_is(Tokens.IDENTIFIER):
            return IDAccessorExpression(self._previous())

        if self._next_is(Tokens.L_BRACKET):
            return self._finish_array()

        self._consume(Tokens.L_PARENTHESIS)
        expr: Expression = self.expression()
        self._consume(Tokens.R_PARENTHESIS)

        return expr

    ### Helpers ###

    def _body(self) -> Body:
        stmts: list[Statement] = []

        self._consume(Tokens.INDENT)

        # We require atleast one statement
        stmts.append(self.statement())

        while not self._at_end() and not self._next_is(Tokens.DEDENT):
            stmts.append(self.statement())

        return Body(stmts)

    def _finish_array(self) -> list[Expression]:
        values: list[Expression] = []

        if self._peek().type_id != Tokens.R_BRACKET:
            values.append(self.expression())

            while self._next_is(Tokens.COMMA):
                expr = self.expression()
                values.append(expr)

        self._consume(Tokens.R_BRACKET)
        return ArrayInitializer(values)

    def _finish_call(self, callee: Expression) -> CallExpression:
        args: list[Expression] = []

        if self._peek().type_id != Tokens.R_PARENTHESIS:
            args.append(self.expression())

            while self._next_is(Tokens.COMMA):
                token = self._previous()
                expr = self.expression()

                if len(args) > MAX_PARAMETERS:
                    self.exceptions.append(SyntaxError(token.line, token.file_name, message=f'Too many arguments for function call, MAX: {MAX_PARAMETERS}!'))
                    continue

                args.append(expr)

        paren = self._consume(Tokens.R_PARENTHESIS)
        return CallExpression(callee, paren, args)

    ### Utilities ###

    def _next_is(self, *args: Tokens) -> bool:
        """Checks and consumes the next token if it is any of 'args', otherwise the token stream is unaffected"""
        for type_id in args:
            if self._current_is(type_id):
                self._advance()
                return True

        return False

    def _consume(self, *types: Tokens) -> Token:
        """Consume the next token if it is any of types, raise an exception otherwise"""

        for type_id in types:
            if self._current_is(type_id):
                return self._advance()

        if self._at_end() or self._peek().type_id == Tokens.EOF:
            raise EOFUnexpectedException(self.filename)

        raise TokenUnexpectedException(self._peek())

    def _current_is(self, type_id: Tokens) -> bool:
        """Checks if the current token is of type_id without consuming"""
        if self._at_end():
            return False

        return self._peek().type_id == type_id

    def _advance(self) -> Token:
        """Consumes and returns the current token to be examined. Always returns the last token if at end"""
        if not self._at_end():
            self.current += 1

        # Advance will incrememnt if we have reached the end otherwise it will just
        # always return whatever the last value was.
        return self._previous()

    def _next_statment(self) -> None:
        """Consumes all tokens from the stream until it reaches what appears to be the next statement boundary"""
        while not self._at_end() and not self._next_is(Tokens.NEWLINE):
            self._advance()

        return

    def _previous(self) -> Token:
        """Returns most recently examined token in the stream"""
        return self.tokens[self.current - 1]

    def _peek(self) -> Token:
        """Returns current token to be examined"""
        return self.tokens[self.current]

    def _at_end(self) -> bool:
        return self.current == len(self.tokens)
