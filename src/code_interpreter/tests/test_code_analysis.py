"""
Tests for the code analysis module.
"""

import pytest
from src.code_interpreter.code_analysis import (
    collect_static_metrics,
    attempt_code_repair,
    format_error,
    get_suggestion,
)


# Should collect basic metrics from code
def test_collect_static_metrics_basic():
    code = """
def solve(I):
    return I
    """.strip()

    metrics = collect_static_metrics(code)

    assert metrics.num_lines > 0
    assert metrics.avg_line_length > 0
    assert metrics.compilation_success is True


# Should detect control structures in code
def test_detect_control_structures():
    code = """
def solve(I):
    if not I:
        return []
    
    result = []
    for row in I:
        new_row = []
        for item in row:
            new_row.append(item)
        result.append(new_row)
    
    while len(result) < 5:
        result.append([0])
        
    return result
    """.strip()

    metrics = collect_static_metrics(code)

    assert metrics.num_if_statements == 1
    assert metrics.num_for_loops == 2
    assert metrics.num_while_loops == 1


# Should detect imports in code
def test_detect_imports():
    code = """
import numpy as np
from typing import List
import scipy.ndimage

def solve(I):
    return np.array(I).tolist()
    """.strip()

    metrics = collect_static_metrics(code)

    assert "numpy" in metrics.imports_used
    assert "typing" in metrics.imports_used
    assert "scipy.ndimage" in metrics.imports_used


# Should detect DSL function calls in code
def test_detect_dsl_functions():
    code = """
def solve(I):
    # Use some DSL functions
    rotated = rot90(I)
    mirrored = hmirror(rotated)
    return canvas(0, shape(mirrored))
    """.strip()

    metrics = collect_static_metrics(code)

    assert "rot90" in metrics.dsl_functions_used
    assert "hmirror" in metrics.dsl_functions_used
    assert "canvas" in metrics.dsl_functions_used
    assert "shape" in metrics.dsl_functions_used


# Should handle syntax errors properly
def test_syntax_error():
    code = """
def solve(I)
    # Missing colon
    return I
    """.strip()

    metrics = collect_static_metrics(code)

    assert metrics.compilation_success is False


# Should repair code with missing numpy import
def test_repair_missing_numpy_import():
    code = """
    def solve(I):
        return np.array(I).tolist()
    """

    error_msg = "NameError: name 'np' is not defined"

    repaired_code = attempt_code_repair(code, error_msg)

    assert "import numpy as np" in repaired_code


# Should repair code with missing DSL function import
def test_repair_missing_dsl_import():
    code = """
def solve(I):
    return rot90(I)
    """.strip()

    error_msg = "NameError: name 'rot90' is not defined"

    repaired_code = attempt_code_repair(code, error_msg)

    assert "from dsl import *" in repaired_code


# Should add bounds checking for index errors in nested loops
def test_repair_index_error():
    code = """
def solve(I):
    rows = len(I)
    cols = len(I[0])
    result = []
    for i in range(rows):
        for j in range(cols+5):  # This will cause index errors
            val = I[i][j]
    return result
    """.strip()

    error_msg = "IndexError: list index out of range"

    repaired_code = attempt_code_repair(code, error_msg)

    # Should add bounds checking
    assert "if j >= len" in repaired_code


# Should return original code when no repair is needed
def test_no_repair_needed():
    code = """
def solve(I):
    return I
    """.strip()

    error_msg = "Some random error that we don't handle"

    repaired_code = attempt_code_repair(code, error_msg)

    # Should return original code unchanged
    assert repaired_code == code


# Should show diff when repairing code
def test_repair_shows_diff():
    code = """
def solve(I):
    # Will have TypeError: 'NoneType' object...
    result = process_grid(I)
    # Missing return
    """.strip()

    error_msg = "TypeError: 'NoneType' object is not subscriptable"

    # Note: this test needs special handling since we can't easily capture debug output
    # In a real environment, we'd mock the debug function and assert on calls
    repaired_code = attempt_code_repair(code, error_msg)

    # At minimum, check the repair was made
    assert "return I" in repaired_code or "return result" in repaired_code


# Should format error with line number information
def test_format_error_with_line_number():
    code = """
def solve(I):
    x = 1 / 0  # Line with error
    return I
    """.strip()

    error_msg = 'Traceback (most recent call last):\n  File "temp.py", line 2, in <module>\n    x = 1 / 0\nZeroDivisionError: division by zero'

    formatted = format_error(error_msg, code)

    # Should include line number, context, and suggestion
    assert "ZeroDivisionError" in formatted
    assert "line 2" in formatted or "Line with error" in formatted
    assert "Suggestion:" in formatted


# Should format error without line number information
def test_format_error_without_line_number():
    code = """
def solve(I):
    return I
    """.strip()

    error_msg = "Some generic error without line numbers"

    formatted = format_error(error_msg, code)

    # Should still include error and suggestion
    assert "generic error" in formatted
    assert "Suggestion:" in formatted


# Should provide appropriate suggestions for different error types
def test_get_suggestions_for_error_types():
    from src.code_interpreter.code_analysis import ErrorType

    # Test suggestions for various error types
    suggestions = {
        ErrorType.INDEX_ERROR: get_suggestion(
            ErrorType.INDEX_ERROR, "IndexError: list index out of range"
        ),
        ErrorType.TYPE_ERROR: get_suggestion(
            ErrorType.TYPE_ERROR, "TypeError: 'NoneType' object is not subscriptable"
        ),
        ErrorType.NAME_ERROR: get_suggestion(
            ErrorType.NAME_ERROR, "NameError: name 'foo' is not defined"
        ),
        ErrorType.VALUE_ERROR: get_suggestion(
            ErrorType.VALUE_ERROR, "ValueError: not enough values to unpack"
        ),
        ErrorType.KEY_ERROR: get_suggestion(
            ErrorType.KEY_ERROR, "KeyError: 'missing_key'"
        ),
        ErrorType.ZERO_DIVISION_ERROR: get_suggestion(
            ErrorType.ZERO_DIVISION_ERROR, "ZeroDivisionError: division by zero"
        ),
        ErrorType.UNKNOWN: get_suggestion(ErrorType.UNKNOWN, "Some unknown error type"),
    }

    # Verify each suggestion is appropriate
    assert "indices" in suggestions[ErrorType.INDEX_ERROR].lower()
    assert "none" in suggestions[ErrorType.TYPE_ERROR].lower()
    assert "defined" in suggestions[ErrorType.NAME_ERROR].lower()
    assert "key" in suggestions[ErrorType.KEY_ERROR].lower()
    assert "zero" in suggestions[ErrorType.ZERO_DIVISION_ERROR].lower()

    # All suggestions should be non-empty and meaningful
    for suggestion in suggestions.values():
        assert suggestion
        assert isinstance(suggestion, str)
        assert len(suggestion) > 10
