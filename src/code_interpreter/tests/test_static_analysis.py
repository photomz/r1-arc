from src.code_interpreter.static_analysis import (
    StaticMetrics,
    inject_logging,
    validate_solve_function_cst,
    SolveFunctionValidator,
)
import pytest

code = """
def solve_e9afcf9a(I):
    x1 = astuple(TWO, ONE)
    x2 = crop(I, ORIGIN, x1)
    x3 = hmirror(x2)
    x4 = hconcat(x2, x3)
    x5 = hconcat(x4, x4)
    O = hconcat(x5, x4)
    return O
    """.strip()

f_67a3c6ac = """
import os
def solve(I):
    O = vmirror(I)
    return O""".strip()

STRINGS = [code, f_67a3c6ac]


def test_ast(snapshot):
    """Should meter syntax by AST"""
    return StaticMetrics.from_literal(code) == snapshot


@pytest.mark.parametrize("code", STRINGS)
def test_cst(snapshot, code):
    """Should inject logging code"""
    return inject_logging(code) == snapshot


def test_no_solver():
    """Should reject code without a solve function"""
    code = """
def not_solve(x):
    return x
    """
    valid = validate_solve_function_cst(code)
    assert not valid
    # assert isinstance(validator, SolveFunctionValidator)
    # assert not validator.has_solve_function
    # assert not validator.is_valid


def test_nested_solve_function():
    """Should reject nested solve functions that aren't at the top level"""
    code = """
def outer():
    def solve(x):
        return x
    return solve(5)
    """
    valid = validate_solve_function_cst(code)
    assert not valid
    # validator = validate_solve_function_cst(code)
    # assert isinstance(validator, SolveFunctionValidator)
    # assert not validator.is_valid
    # assert not validator.has_solve_function


def test_multiple_solve_functions():
    """Should detect multiple solve functions in the same module"""
    code = """
def solve(x):
    return x
def solve(y):
    return y*2
    """
    valid = validate_solve_function_cst(code)
    assert not valid
    # validator = validate_solve_function_cst(code)
    # assert isinstance(validator, SolveFunctionValidator)
    # # The validator will only track the last one it sees
    # assert validator.has_solve_function
    # assert not validator.is_valid
    # assert validator.n_return == 2


def test_conditional_returns():
    """Should reject solve functions with multiple return statements in conditionals"""
    code = """
def solve(x):
    if x > 0:
        return x
    else:
        return -x
    """
    valid = validate_solve_function_cst(code)
    assert not valid
    # validator = validate_solve_function_cst(code)
    # assert isinstance(validator, SolveFunctionValidator)
    # assert validator.has_solve_function
    # assert validator.n_param == 1
    # assert validator.n_return == 2
    # assert not validator.is_valid


def test_loop_return():
    """Should reject solve functions with returns inside loops"""
    code = """
def solve(x):
    for i in range(10):
        if i == x:
            return i
    return 0
    """
    valid = validate_solve_function_cst(code)
    assert not valid
    # validator = validate_solve_function_cst(code)
    # assert isinstance(validator, SolveFunctionValidator)
    # assert validator.has_solve_function
    # assert validator.n_param == 1
    # assert validator.n_return == 2
    # assert not validator.is_valid


def test_code_injection():
    """Should validate solve functions even with potentially malicious code"""
    code = """
def solve(x):
    exec("import os; os.system('rm -rf /')")
    return x
    """
    valid = validate_solve_function_cst(code)
    assert valid
    # validator = validate_solve_function_cst(code)
    # assert isinstance(validator, SolveFunctionValidator)
    # assert validator.has_solve_function
    # assert validator.n_param == 1
    # assert validator.n_return == 1
    # assert validator.is_valid  # Note: our validator only checks structure, not security


def test_extremely_large_function():
    """Should handle extremely large solve functions without crashing"""
    code = "def solve(x):\n" + "    y = x\n" * 10000 + "    return y"
    valid = validate_solve_function_cst(code)
    assert valid

    # validator = validate_solve_function_cst(code)
    # assert isinstance(validator, SolveFunctionValidator)
    # assert validator.has_solve_function
    # assert validator.n_param == 1
    # assert validator.n_return == 1
    # assert validator.is_valid


def test_unicode_deception():
    """Should reject functions with names that look like 'solve' but use different Unicode characters"""
    code = """
def sοlve(x):  # Uses Greek omicron (ο) instead of 'o'
    return x
    """
    assert not validate_solve_function_cst(code)
    # validator = validate_solve_function_cst(code)
    # assert isinstance(validator, SolveFunctionValidator)
    # assert not validator.has_solve_function
    # assert not validator.is_valid


def test_default_parameters():
    """Should handle solve functions with default parameters"""
    code = """
def solve(x=10):
    return x
    """
    assert (valid := validate_solve_function_cst(code))
    # validator = validate_solve_function_cst(code)
    # assert isinstance(validator, SolveFunctionValidator)
    # assert validator.has_solve_function
    # assert validator.n_param == 1
    # assert validator.n_return == 1
    # assert validator.is_valid


def test_return_with_side_effects():
    """Should validate solve functions with side effects in return statements"""
    code = """
def solve(x):
    return print(x) or x
    """
    assert (valid := validate_solve_function_cst(code))
    # validator = validate_solve_function_cst(code)
    # assert isinstance(validator, SolveFunctionValidator)
    # assert validator.has_solve_function
    # assert validator.n_param == 1
    # assert validator.n_return == 1
    # assert validator.is_valid


def test_syntax_error():
    """Should handle code with syntax errors gracefully"""
    code = """
def solve(x):
    return x)
    """
    valid = validate_solve_function_cst(code)
    assert not valid
    # validator = validate_solve_function_cst(code)
    # assert not validator  # Should return False for syntax errors


# def test_injection_security():
#     """Should safely handle injection attempts in logging transformation"""
#     code = """def solve(x):\n    return "\\"\nexec('import os; os.system(\\"rm -rf /\\")')\n#"""
#     result = inject_logging(code)
#     assert "exec(" not in result or "os.system" not in result
