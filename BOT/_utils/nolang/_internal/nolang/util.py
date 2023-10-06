
def is_digit(c: str):
    return ord('0') <= ord(c) <= ord('9') if c else False

def is_alpha(c: str):
    return ord('A') <= ord(c) <= ord('Z') or ord('a') <= ord(c) <= ord('z') if c else False

def is_alpha_numeric(c: str):
    return is_digit(c) or is_alpha(c) if c else False
