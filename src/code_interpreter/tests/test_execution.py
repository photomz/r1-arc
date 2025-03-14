"""
Snapshot tests for solvers with DSL-specific profiling.
"""

import pytest
from typing import Tuple

from src.code_interpreter.static_analysis import StaticMetrics
from src.code_interpreter.execution import Interpreter
from src.utils.tasks import TASKS, deep_tuple
from src.utils import get_shape, FORMATTERS, FormatterNames
import random


f_67a3c6ac = """
import os
def solve(I):
    O = vmirror(I)
    return O""".strip()


f_68b16354 = """
import os
def solve(I):
    O = hmirror(I)
    return O""".strip()


f_4290ef0e = """
def solve(I):
    x1 = mostcolor(I)
    x2 = fgpartition(I)
    x3 = objects(I, T, F, T)
    x4 = rbind(valmax, width)
    x5 = lbind(colorfilter, x3)
    x6 = compose(x5, color)
    x7 = compose(double, x4)
    x8 = lbind(prapply, manhattan)
    x9 = fork(x8, identity, identity)
    x10 = lbind(remove, ZERO)
    x11 = compose(x10, x9)
    x12 = rbind(branch, NEG_TWO)
    x13 = fork(x12, positive, decrement)
    x14 = chain(x13, minimum, x11)
    x15 = fork(add, x14, x7)
    x16 = compose(x15, x6)
    x17 = compose(invert, x16)
    x18 = order(x2, x17)
    x19 = rbind(argmin, centerofmass)
    x20 = compose(initset, vmirror)
    x21 = fork(insert, dmirror, x20)
    x22 = fork(insert, cmirror, x21)
    x23 = fork(insert, hmirror, x22)
    x24 = compose(x19, x23)
    x25 = apply(x24, x18)
    x26 = size(x2)
    x27 = apply(size, x2)
    x28 = contained(ONE, x27)
    x29 = increment(x26)
    x30 = branch(x28, x26, x29)
    x31 = double(x30)
    x32 = decrement(x31)
    x33 = apply(normalize, x25)
    x34 = interval(ZERO, x30, ONE)
    x35 = pair(x34, x34)
    x36 = mpapply(shift, x33, x35)
    x37 = astuple(x32, x32)
    x38 = canvas(x1, x37)
    x39 = paint(x38, x36)
    x40 = rot90(x39)
    x41 = paint(x40, x36)
    x42 = rot90(x41)
    x43 = paint(x42, x36)
    x44 = rot90(x43)
    O = paint(x44, x36)
    return O
"""

bad_gen_code1 = """
def main():
    import sys
    from functools import reduce

    # Read the input
    input_data = sys.stdin.read().split()
    input_lines = []
    line_length = 20
    idx = 0
    for i in range(20):
        row = []
        while idx < len(input_data):
            if input_data[idx].lstrip('0') == '':
                idx += 1
                continue
            row.append(int(input_data[idx]))
            idx += 1
        input_lines.append(row)
    
    # Define the Rule application function
    def rule3(grid, prefix_length=10):
        n = grid.shape[0]
        output = np.zeros((n, n), dtype=np.uint8)
        
        for i in range(n):
            for j in range(n):
                val = grid[i, j]
                if val == 0:
                    output[i, j] = val
                    continue
                
                binary = bin(val)[2:]  # Convert to binary without '0b'
                # Prepend the prefix to the binary string
                prefixed = binary.zfill(prefix_length)
                
                # Determine the position to place the prefixed binary
                if len(prefixified) > prefix_length:
                    pass  # Truncate if longer than prefix
                else:
                    start = prefix_length
                    if start + len(prefixified) > n:
                        start = n - len(prefixified)
                    
                    output[i, j] = np.uint8(prefixified)
                    output[i, j+1] = output[i, j]
                    output[i, j+2] = output[i, j]
                    output[i, j+3] = output[i, j]
                    output[i, j+4] = output[i, j]
                    output[i, j+5] = output[i, j]
                    output[i, j+6] = output[i, j]
                    output[i, j+7] = output[i, j]
                    if len(prefixified) > 1:
                         output[i, j+1] = np.uint8(prefixified[:-1])
                    if len(prefixified) > 2:
                         output[i, j+2] = np.uint8(prefixified[:-2])
                    if len(prefixified) > 3:
                         output[i, j+3] = np.uint8(prefixified[:-3])
                    if len(prefixified) > 4:
                         output[i, j+4] = np.uint8(prefixified[:-4])
                    if len(prefixified) > 5:
                         output[i, j+5] = np.uint8(prefixified[:-5])
                    if len(prefixified) > 6:
                         output[i, j+6] = np.uint8(prefixified[:-6])
        
        tracer.log_locals(list(locals().items()))
        
        return output
    
    # Execute the rule on the input grid
    output_grid = rule3(input_lines)
    
    # Convert output to image representation
    output_image = np.where(output_grid != 0, output_grid, 0)
    
    # Print the output image
    for row in output_image:
        print(''.join(map(str, row)))
    
    tracer.log_locals(list(locals().items()))

if __name__ == "__main__":
    main()
""".strip()

CODES = [f_67a3c6ac, f_68b16354, f_4290ef0e]  # , bad_gen_code1]


# Fixture w random seed, returns nxn tuple[tuple[int]] (n is arg)
@pytest.fixture
def grid() -> Tuple[Tuple[int]]:
    n = 10
    random.seed(0)
    return tuple(tuple(random.randint(0, 9) for _ in range(n)) for _ in range(n))


@pytest.mark.parametrize("code", CODES)
def test_hardcoded_exec(code, grid, snapshot):
    # TODO: pytest catch error
    trace = Interpreter.run(
        code.strip(),
        id=["1", "2", "3"],
        inputs=[grid],
    )
    assert trace == snapshot


train_ids = [t.id for t in TASKS.values() if t.split == "train"]

# Only when --stress flag is passed


def to_2d(O):
    if isinstance(O, int):
        return ((O,),)  # if int -> Tuple[Tuple[int]]
    if isinstance(O, tuple) and all(isinstance(row, int) for row in O):
        return (O,)  # if Tuple[int] -> Tuple[Tuple[int]]
    return O


fmt = FORMATTERS[FormatterNames.CELL_DIFF]


@pytest.mark.stress
@pytest.mark.parametrize("func", ["solver_repr", "verifier_repr"])
@pytest.mark.parametrize("task_id", train_ids)
def test_solve_any_id_in_interpreter(task_id, func, snapshot):
    task = TASKS[task_id]
    test_Is = [ex.I for ex in task.train + task.test]

    codestring = getattr(task, func)
    res = Interpreter.run(codestring, inputs=test_Is, id=task_id, cleanup=False)
    metrics = StaticMetrics.from_literal(codestring)

    assert metrics.syntax_ok and res.run_ok

    shell_Is = res.traces.I
    assert shell_Is == test_Is

    shell_Os = res.traces.O
    test_Os = [deep_tuple(ex.O) for ex in task.train + task.test]
    assert get_shape(shell_Os) == get_shape(test_Os)
    for want, expect in zip(shell_Os, test_Os):
        assert want == expect, fmt.diff(
            want, expect, get_shape(want), get_shape(expect)
        )

    assert (res, metrics) == snapshot
