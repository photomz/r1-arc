from abc import ABC, abstractmethod
from dataclasses import dataclass
from collections import defaultdict
from enum import Enum
from typing import TypeVar
from itertools import product
import numpy as np
import typer
from pathlib import Path
import json
from devtools import debug

from src.utils.tasks import TASKS
from src.utils.devtools import debug
from src.dsl.arc_types import *
from src.utils.grid import *
import random

app = typer.Typer()
random_id = int(random.random() * 800)


@app.command()
def format(id: str = None, style: FormatterNames = FormatterNames.GRID):
    """Format or diff all grids in ARC task"""
    formatter = FORMATTERS[style]
    is_diff = style in {FormatterNames.LIST_DIFF, FormatterNames.CELL_DIFF}

    if id is None:
        id = list(TASKS.values())[random_id].id
    grids = TASKS[id].train + TASKS[id].test

    for i, g in enumerate(grids, 1):
        print(f"> Grid {i if i < len(grids) else 'TEST'}")

        A = np.array(g.I)
        B = np.array(g.O)

        # B.size skips held-out test grid output
        if is_diff:
            diff = formatter.diff(A, B, A.shape, B.shape)
            debug(diff)
        else:
            finput = formatter.format(A, A.shape)
            debug(finput)

            foutput = formatter.format(B, B.shape)
            debug(foutput)
    debug(id)

    print(TASKS[id].format_prompt())


if __name__ == "__main__":
    app()
