
from .token import Token
from .token import Tokens
from .token import RESERVED_IDENTIFIERS

from ..types import *
from ..util import *
from ..exception import *

class Lexer:
    """
    The lexer has two main states: NEWLINE and GENERIC.

    In the NEWLINE state (which is the initial state), the lexer will consume all
    newline characters and acitvely search for code. Upon encountering whitespace that
    isn't a newline, it will increment an indentation level counter appropriately to
    keep track. If it reaches a newline or a comment all previous characters will be consumed
    and discarded and no tokens will be generated

    If it encounters a character that suggests the start of an actual token it will first
    generate either one INDENT token or zero or more DEDENT tokens based on an indent stack.
    Then it will switch into the GENERIC state.

    The indentation stack starts with a single 0, indicating indentation level 0
    an INDENT token is generated when the new indentation level is larger than the topmost value
    of the stack, the new indentation level to be pushed to the stack.

    One or more DEDENT tokens are generated if the new indentation level is less than the topmost
    value of the stack, each value will be popped of the stack (and likewise a DEDENT generated) until
    the topmost value of the stack equals the new indentation level. If it does not find an equivalent
    value in the stack then this is an error and the indentation level is reset to 0 for the next tokens

    If the new indentation level is equal to the topmost value of the stack, then no INDENT or DEDENT
    token is generated.

    The GENERIC state of the lexer acts normally and attempts to capture entire tokens for
    the language. Whitespace and comments are discarded.

    Upon reaching a newline character, the lexer will generate a NEWLINE token and switch to the
    NEWLINE state.
    """

    def scan(self, source: str, file_name: str = None) -> list[Tokens]:
        """Scan source string and generate stream of tokens"""
        self.source = source
        self.file_name = file_name
        self.exceptions = []

        # Start of current lexeme
        self.start: int = 0

        # Current character in current lexeme
        self.current: int = 0

        # Current line in the source, we start at 1
        self.line: int = 1

        # List of tokens extracted
        self.tokens: list[Token] = []

        # Stack keeping track of indentation levels
        self.indent_stack: list[int] = [0]

        while not self._at_end():
            self._scan_nextline()

        # Make sure to reset back to indentation level 0!
        while self.indent_stack[-1] != 0:
            self.indent_stack.pop()
            self._gen_token(Tokens.DEDENT)

        self._gen_token(Tokens.EOF)

        if len(self.exceptions) > 0:
            raise ExceptionGroup('Lexer exceptions', self.exceptions)

        return self.tokens

    ### Mainstates ###

    def _scan_nextline(self) -> None:
        """
        Scans the next 'line' of tokens. This may span more than one actual line from the source code.
        'Line' refers to actual code present up to the next newline '\n' character. Any whitespace, newlines
        and comments present before this do not count.
        """

        # We first consume as many consecutive newlines as we see
        while self._next_is('\n'):
            self.line += 1

        # Check for indentation level
        indentation = self._consume_indentation()

        # This was just an empty line, throw away everything...
        if self._at_end() or self._peek() == '\n': return

        # Comments in this state are ignored all the way to the next newline
        if self._next_is('#'):
            self._goto_next('\n')
            return

        # Now we know we are moving to the GENERIC state we may need to generate
        # an indent or a dedent token before proceeding.
        top = self.indent_stack[-1]

        # If we increased indentation level since last time, we generate an indent
        if top < indentation:
            self.indent_stack.append(indentation)
            self._gen_token(Tokens.INDENT)

        # If we have equal or fewer spaces we generate zero or more dedents
        else:
            while top != indentation and top > 0:
                self.indent_stack.pop()
                self._gen_token(Tokens.DEDENT)
                top = self.indent_stack[-1]

            # Something went wrong...
            if top != indentation:
                self._error(InconsistentIndentationException(self.line, self.file_name))

        # Process the actual code
        while not self._at_end() and self._peek() != '\n':
            self.start = self.current
            self._scan_next_token()

        self._gen_token(Tokens.NEWLINE)

    def _scan_next_token(self) -> None:

        match self._advance():
            case '(': self._gen_token(Tokens.L_PARENTHESIS)
            case ')': self._gen_token(Tokens.R_PARENTHESIS)
            case '[': self._gen_token(Tokens.L_BRACKET)
            case ']': self._gen_token(Tokens.R_BRACKET)
            case ',': self._gen_token(Tokens.COMMA)
            case '+': self._gen_token(Tokens.PLUS)
            case '-': self._gen_token(Tokens.MINUS)
            case '/': self._gen_token(Tokens.SLASH)
            case '%': self._gen_token(Tokens.PERCENT)
            case '*': self._gen_token(Tokens.EXP if self._next_is('*') else Tokens.STAR)
            case '<': self._gen_token(Tokens.LESS_THAN_EQ if self._next_is('=') else Tokens.LESS_THAN)
            case '>': self._gen_token(Tokens.GREATER_THAN_EQ if self._next_is('=') else Tokens.GREATER_THAN)
            case '=': self._gen_token(Tokens.EQUAL if self._next_is('=') else Tokens.ASSIGN)
            case '!':
                if self._next_is('='):
                    self._gen_token(Tokens.NEQUAL)

                else:
                    self._error(CharacterUnexpectedException('!', self.line, self.file_name))

            # We ignore white space here
            case ' ':  return
            case '\t':  return

            # If we encounter comment, consume upto the next newline
            case '#': self._goto_next('\n')

            case '"': self._process_string_literal('"')
            case "'": self._process_string_literal('\'')
            case '\\': self._process_escape()

            case c:
                if is_digit(c):
                    self._process_number_literal()
                elif is_alpha(c) or c == '_':
                    self._process_identifier()
                else:
                    self._error(CharacterUnexpectedException(c, self.line, self.file_name))

    ### Substates ###

    def _process_string_literal(self, end) -> None:

        # Start with empty char buffer
        val : list[str] = []

        while self._peek() != end:
            if self._at_end():
                self._error(UnterminatedStringException(self.line, self.file_name))
                return

            next_char = self._advance()

            # Multiline string
            if next_char == '\n':
                self.line += 1

            # Encountered an escape sequence
            elif next_char == '\\':
                match self._advance():
                    case '\\': next_char = '\\'
                    case '"':  next_char = '"'
                    case "'":  next_char = "'"
                    case 'n':  next_char = '\n'
                    case 'r':  next_char = '\r'
                    case 'b':  next_char = '\b'
                    case 'v':  next_char = '\v'
                    case 't':  next_char = '\t'
                    case 'a':  next_char = '\a'
                    case '0':  next_char = '\0'
                    case   c:  self._error(UnknownEscapeSequenceException(c, self.line, self.file_name))

            val.append(next_char)

        # Consume closing quote
        self._advance()
        self._gen_token(Tokens.STR_LITERAL, NolangString(''.join(val)))

    def _process_number_literal(self) -> None:
        # Consume all next digits
        while is_digit(self._peek()):
            self._advance()

        # If the next character is a decimal point this may be a floating point literal
        if self._peek() == '.':
            self._advance()

            while is_digit(self._peek()):
                self._advance()

            self._gen_token(Tokens.FLOAT_LITERAL, NolangFloat(float(self._current_lexeme())))
            return

        # Otherwise we have an integer literal
        self._gen_token(Tokens.INT_LITERAL, NolangInt(int(self._current_lexeme())))

    def _process_identifier(self) -> None:
        while is_alpha_numeric(self._peek()) or self._peek() == '_':
            self._advance()

        val = self._current_lexeme()
        token_type = RESERVED_IDENTIFIERS.get(val)

        if token_type:
            match token_type:
                case Tokens.TRUE: val = NolangBool(True)
                case Tokens.FALSE: val = NolangBool(False)
                case Tokens.NOL: val = NOL

        # If it's not an existing token then it's a user specified identifier
        else:
            token_type = Tokens.IDENTIFIER

        self._gen_token(token_type, val)

    def _process_escape(self) -> None:
        while self._peek() == ' ' or self._peek() == '\t':
            self._advance()

        if self._next_is('\n'):
            self.line += 1
        else:
            self._error(CharacterUnexpectedException('\\', self.line, self.file_name))

    ### Utilities ###

    def _advance(self) -> str:
        """Advance a character in the source string. Returns None and does not advance if at end."""
        if self._at_end():
            return

        cur = self.source[self.current]
        self.current += 1
        return cur

    def _peek(self, look_ahead: int = 1) -> str:
        """Peek the next character without advancing. Returns None if at end or look_ahead is at end."""
        index = self.current + look_ahead - 1

        if self._at_end() or index >= len(self.source):
            return

        return self.source[index]

    def _next_is(self, c: str) -> bool:
        """Returns whether or not the next value is 'c' and consumes the character if so. Returns False if at end."""
        if self._at_end():
            return False

        if self.source[self.current] != c:
            return False

        self.current += 1
        return True

    def _goto_next(self, c: str) -> None:
        """Will attempt to find the given character or, if not present, goes to end"""
        while not self._at_end() and self._peek() != c:
            self._advance()

    def _at_end(self) -> bool:
        """Returns true if there are no characters left to process, else otherwise"""
        return self.current >= len(self.source)

    def _gen_token(self, type_id, value: NolangType = None) -> None:
        """Generates a token of type_id with optional value at the current lexeme"""
        self.tokens.append(Token(type_id, '\0' if type_id == Tokens.EOF else self._current_lexeme(), self.line, self.file_name, value))

    def _current_lexeme(self, start_offset: int = 0, current_offset: int = 0) -> str:
        """Returns the current lexeme given the processing window"""
        return self.source[self.start + start_offset:self.current + current_offset]

    def _consume_indentation(self) -> int:
        """Calculates and returns any indentation level at the current processing location"""
        indentation = 0

        while self._peek() == ' ' or self._peek() == '\t':
            c = self._advance()

            # A space increments indentation level by one
            # A tab increases the indentation to the next multiple of 4
            indentation += int(c == ' ') + int(c == '\t') * (4 - (indentation % 4))

        return indentation

    def _error(self, exception: NolangException) -> None:
        self.exceptions.append(exception)
