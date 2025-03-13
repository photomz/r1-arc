"""
Snapshot tests for solvers with DSL-specific profiling.
"""

import pytest
from syrupy.assertion import SnapshotAssertion
from typing import List, Dict, Any, Tuple

from src.code_interpreter.execution import (
    run_python_transform_sync,
    ResourceLimits,
    PythonResult,
)
import random
from src.code_interpreter.tracer import DSLTracer

# from src.dsl.dsl import *
# from src.dsl.constants import *
# from src.dsl.arc_types import *

f_67a3c6ac = """def solve(I):
    O = vmirror(I)
    return O"""


f_68b16354 = """def solve(I):
    O = hmirror(I)
    return O"""


# Fixture w random seed, returns nxn tuple[tuple[int]] (n is arg)
@pytest.fixture
def grid() -> Tuple[Tuple[int]]:
    n = 10
    random.seed(0)
    return tuple(tuple(random.randint(0, 9) for _ in range(n)) for _ in range(n))


def profile_dsl(code_literal, grid):
    tracer = DSLTracer()
    tracer.instrument("src.dsl.dsl")
    tracer.clear()
    tracer.enabled = True

    # Run the solver
    result = run_python_transform_sync(
        code_literal,
        grid_lists=grid,
        resource_limits=ResourceLimits(max_time_seconds=5),
    )
    return result

    return tracer


@pytest.mark.parametrize("code", [f_67a3c6ac, f_68b16354])
def test_dsl_trace_sanity(code, grid, snapshot):
    trace = profile_dsl(code.strip(), [grid])
    assert trace == snapshot
