
import sys
import json
import traceback
import numpy as np
from typing import *
from pathlib import Path

root = Path(__file__).resolve()
while not (root / ".git").exists():
    root = root.parent
sys.path.append(str(root))

from src.dsl.arc_types import *
from src.dsl.dsl import *
from src.utils.devtools import verbose_debug as original_verbose_debug

# Custom verbose_debug function to capture and print output
def verbose_debug(x):
    """Wrapper for verbose_debug that prints to stdout with a special marker"""
    result = original_verbose_debug(x)
    # Print debug output with special marker for parsing
    if isinstance(x, int) and x == 0:
        print("DEBUG_OUTPUT:" + result)
    return x

# Original code with solve/transform function
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

# Input grid
grid_lists = [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]

def to_python_array(obj):
    """Convert numpy arrays back to normal Python lists"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return to_python_array(obj.tolist())
    elif isinstance(obj, list):
        return [to_python_array(item) for item in obj]
    return obj

try:
    results: list[list[list[int]]] = []
    for grid_list in grid_lists:
        # Try to call solve function (preferred) or fall back to transform

        if 'solve' in globals():
            result = solve(grid_list)
        else:
            print("Error: `solve` function not found", file=sys.stderr)
            sys.exit(1)
            
        result = to_python_array(result)
        
        results.append(result)
    
    # Print results as JSON with special marker for parsing
    print("TRANSFORM_RESULT:" + json.dumps(results))
except Exception as e:
    print(f"Error executing code: {e}\n{traceback.format_exc()}", file=sys.stderr)
    sys.exit(1)