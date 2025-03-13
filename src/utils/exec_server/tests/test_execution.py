"""
Tests for the Python execution module.
"""

import asyncio
import pytest
from typing import List, Set, Optional

from src.utils.exec_server.execution import (
    run_python_transform_sync,
    run_python_transform_async,
    run_python_transforms,
    ResourceLimits,
    PythonResult,
    PythonException,
    TransformInput,
)


# Sample happy path code with transform function
VALID_TRANSFORM_CODE = """
def solve(grid_list: list[list[int]]) -> list[list[int]]:
    # Rotate 90 degrees clockwise
    if not grid_list:
        return []
    rows = len(grid_list)
    cols = len(grid_list[0])
    result = [[0] * rows for _ in range(cols)]
    
    for i in range(rows):
        for j in range(cols):
            result[j][rows - 1 - i] = grid_list[i][j]
    
    return result
"""

# Sample happy path code with solve function (ARC-DSL style)
VALID_SOLVE_CODE = """
def solve(I: list[list[int]]) -> list[list[int]]:
    # Rotate 90 degrees clockwise
    if not I:
        return []
    rows = len(I)
    cols = len(I[0])
    result = [[0] * rows for _ in range(cols)]
    
    for i in range(rows):
        for j in range(cols):
            result[j][rows - 1 - i] = I[i][j]
    
    return result
"""

# Sample test grid
TEST_GRID = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# Expected rotated grid
EXPECTED_RESULT = [[7, 4, 1], [8, 5, 2], [9, 6, 3]]


@pytest.fixture
def resource_limits():
    """Test resource limits"""
    return ResourceLimits(max_time_seconds=5)


# Should execute valid code with transform function and return correct result
def test_valid_transform_code_execution(resource_limits):
    result = run_python_transform_sync(
        VALID_TRANSFORM_CODE, grid_lists=[TEST_GRID], resource_limits=resource_limits
    )

    assert result.return_code == 0
    assert not result.timed_out
    assert result.transform_results is not None
    assert result.transform_results[0] == EXPECTED_RESULT
    assert result.metrics["static"]["compilation_success"] is True
    assert result.metrics["static"]["execution_success"] is True


# Should execute valid code with solve function and return correct result
def test_valid_solve_code_execution(resource_limits):
    result = run_python_transform_sync(
        VALID_SOLVE_CODE, grid_lists=[TEST_GRID], resource_limits=resource_limits
    )

    assert result.return_code == 0
    assert not result.timed_out
    assert result.transform_results is not None
    assert result.transform_results[0] == EXPECTED_RESULT
    assert result.metrics["static"]["compilation_success"] is True
    assert result.metrics["static"]["execution_success"] is True


# Should handle multiple input grids correctly
def test_valid_code_with_multiple_grids(resource_limits):
    result = run_python_transform_sync(
        VALID_TRANSFORM_CODE,
        grid_lists=[TEST_GRID, TEST_GRID],
        resource_limits=resource_limits,
    )

    assert result.return_code == 0
    assert len(result.transform_results) == 2
    assert result.transform_results[0] == EXPECTED_RESULT
    assert result.transform_results[1] == EXPECTED_RESULT


# Should execute code asynchronously and return correct result
@pytest.mark.asyncio
async def test_async_execution(resource_limits):
    result = await run_python_transform_async(
        VALID_TRANSFORM_CODE, grid_lists=[TEST_GRID], resource_limits=resource_limits
    )

    assert result is not None
    assert result.return_code == 0
    assert result.transform_results[0] == EXPECTED_RESULT


# Should execute multiple transforms in parallel
@pytest.mark.asyncio
async def test_parallel_execution(resource_limits):
    inputs = [
        TransformInput(
            code=VALID_TRANSFORM_CODE,
            grid_lists=[TEST_GRID],
            resource_limits=resource_limits,
        ),
        TransformInput(
            code=VALID_SOLVE_CODE,
            grid_lists=[TEST_GRID, TEST_GRID],
            resource_limits=resource_limits,
        ),
    ]

    results = await run_python_transforms(inputs)

    assert len(results) == 2
    assert results[0].transform_results[0] == EXPECTED_RESULT
    assert len(results[1].transform_results) == 2


# Should handle empty grid input
def test_empty_grid(resource_limits):
    result = run_python_transform_sync(
        VALID_TRANSFORM_CODE, grid_lists=[[]], resource_limits=resource_limits
    )

    assert result.return_code == 0
    assert result.transform_results[0] == []


# Should detect syntax errors in code
def test_syntax_error(resource_limits):
    invalid_code = """
def solve(I: list[list[int]]) -> list[list[int]]:
    # Syntax error - missing colon
    if not I
        return []
    return I
    """.strip()

    result = run_python_transform_sync(
        invalid_code, grid_lists=[TEST_GRID], resource_limits=resource_limits
    )

    assert result.return_code != 0
    assert result.transform_results is None
    assert result.metrics["static"]["compilation_success"] is False


# Should handle runtime errors properly
def test_runtime_error(resource_limits):
    invalid_code = """
def solve(I: list[list[int]]) -> list[list[int]]:
    # Runtime error - division by zero
    x = 1 / 0
    return I
    """.strip()

    result = run_python_transform_sync(
        invalid_code, grid_lists=[TEST_GRID], resource_limits=resource_limits
    )

    assert result.return_code != 0
    assert result.transform_results is None
    assert "ZeroDivisionError" in result.stderr


# Should terminate execution if code exceeds timeout
def test_timeout():
    infinite_loop_code = """
def solve(I: list[list[int]]) -> list[list[int]]:
    # Infinite loop
    while True:
        pass
    return I
    """.strip()

    result = run_python_transform_sync(
        infinite_loop_code,
        grid_lists=[TEST_GRID],
        resource_limits=ResourceLimits(max_time_seconds=1),
    )

    assert result.timed_out
    assert result.transform_results is None


# Should detect invalid return types
def test_invalid_return_type(resource_limits):
    invalid_code = """
def solve(I: list[list[int]]) -> list[list[int]]:
    # Return a string instead of a list
    return "not a grid"
    """.strip()

    result = run_python_transform_sync(
        invalid_code, grid_lists=[TEST_GRID], resource_limits=resource_limits
    )

    assert result.return_code != 0
    assert result.transform_results is None


# Should handle missing function definitions
def test_missing_function_definition(resource_limits):
    invalid_code = """
# No solve or transform function defined
def some_other_function():
    return "hello"
    """.strip()

    result = run_python_transform_sync(
        invalid_code, grid_lists=[TEST_GRID], resource_limits=resource_limits
    )

    assert result.return_code != 0
    assert result.transform_results is None
    assert "not found" in result.stderr.lower()


# Should auto-repair code with missing imports
def test_auto_repair_missing_import(resource_limits):
    code_missing_import = """
def solve(I: list[list[int]]) -> list[list[int]]:
    # Missing import
    return np.rot90(np.array(I), k=3).tolist()
    """.strip()

    result = run_python_transform_sync(
        code_missing_import, grid_lists=[TEST_GRID], resource_limits=resource_limits
    )

    # Should succeed after auto-repair
    assert result.return_code == 0
    assert result.transform_results is not None
    # Check if result is equivalent to expected (may be implemented differently)
    assert result.transform_results[0] == EXPECTED_RESULT


# Should use cached results for identical inputs
# def test_cache_reuse(resource_limits):
#     # First execution
#     start_time = asyncio.get_event_loop().time()
#     result1 = run_python_transform_sync(
#         VALID_TRANSFORM_CODE,
#         grid_lists=[TEST_GRID],
#         resource_limits=resource_limits,
#         use_cache=True,
#     )
#     first_exec_time = asyncio.get_event_loop().time() - start_time

#     # Second execution, should be cached
#     start_time = asyncio.get_event_loop().time()
#     result2 = run_python_transform_sync(
#         VALID_TRANSFORM_CODE,
#         grid_lists=[TEST_GRID],
#         resource_limits=resource_limits,
#         use_cache=True,
#     )
#     second_exec_time = asyncio.get_event_loop().time() - start_time

#     assert result1.transform_results == result2.transform_results
#     # Second execution should be much faster (almost instant)
#     assert second_exec_time < first_exec_time / 2


# Should raise exceptions when requested
def test_exception_raising(resource_limits):
    invalid_code = """
def solve(I: list[list[int]]) -> list[list[int]]:
    # Runtime error
    raise ValueError("Test error")
    return I
    """.strip()

    with pytest.raises(PythonException):
        run_python_transform_sync(
            invalid_code,
            grid_lists=[TEST_GRID],
            resource_limits=resource_limits,
            raise_exception=True,
        )


# Should detect and import missing DSL functions
def test_dsl_function_usage(resource_limits):
    dsl_code = """
def solve(I: list[list[int]]) -> list[list[int]]:
    # Use a DSL function
    return rot90(I)
    """.strip()

    result = run_python_transform_sync(
        dsl_code, grid_lists=[TEST_GRID], resource_limits=resource_limits
    )

    # Should add the missing import and succeed
    assert result.return_code == 0
    assert result.transform_results is not None
    assert "rot90" in result.metrics["static"]["dsl_functions_used"]


# Should collect comprehensive code metrics
def test_metrics_collection(resource_limits):
    complex_code = """
def solve(I: list[list[int]]) -> list[list[int]]:
    # Complex code with multiple constructs
    if not I:
        return []
        
    result = []
    for row in I:
        new_row = []
        for val in row:
            if val > 5:
                new_row.append(10 - val)
            else:
                new_row.append(val)
        result.append(new_row)
        
    return result
    """.strip()

    result = run_python_transform_sync(
        complex_code, grid_lists=[TEST_GRID], resource_limits=resource_limits
    )

    # Check metrics collection
    assert result.metrics["static"]["num_if_statements"] >= 2
    assert result.metrics["static"]["num_for_loops"] >= 2
    assert result.metrics["static"]["compilation_success"] is True
    assert result.metrics["static"]["execution_success"] is True
