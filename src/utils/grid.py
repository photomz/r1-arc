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

from src.utils.devtools import debug
from src.dsl.arc_types import *

# (A - Z) + (AA - AD)
ALPHA = list(chr(ord("A") + i) for i in range(26)) + list(
    f"A{chr(ord('A') + i)}" for i in range(4)
)


def bucket_diffs(g1: Grid, g2: Grid) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """Bucket diffs by index (color)"""
    x, y = (g1 != g2).nonzero()
    diffs = defaultdict(list)

    for i, j in zip(x, y, strict=True):
        pair = (g1[i][j], g2[i][j])
        diffs[pair].append((int(i), int(j)))

    return diffs


@dataclass
class Formatter(ABC):
    sep: str = "|"
    newline: str = "\n"
    n_max: int = 30

    @classmethod
    def format(cls, grid: Grid, shape: Shape) -> str:
        """Base formatter"""
        if shape[0] > cls.n_max or shape[1] > cls.n_max:
            raise ValueError(f"Grid dimensions exceed maximum size of {cls.n_max}")
        return ""


@dataclass
class GridFormatter(Formatter):
    """
    0 | 0 | 0
    0 | 1 | 0
    0 | 0 | 0
    """

    @classmethod
    def format(cls, grid: Grid, shape) -> str:
        super().format(grid, shape)
        return cls.newline.join(cls.sep.join(str(x) for x in row) for row in grid)


@dataclass
class SpreadsheetFormatter(Formatter):
    """
        A | B | C
    1	0 | 0 | 0
    2	0 | 1 | 0
    3 	0 | 0 | 0
    """

    @classmethod
    def format(cls, grid: Grid, shape) -> str:
        super().format(grid, shape)
        _, n_cols = shape

        header = "\t" + cls.sep.join(ALPHA[:n_cols])
        rows = "\n".join(
            str(i + 1) + "\t" + cls.sep.join([str(x) for x in row])
            for i, row in enumerate(grid)
        )
        return cls.newline.join([header, rows])


@dataclass
class Differ(Formatter):
    change: str = "->"
    blank: str = "    "

    @classmethod
    def diff(self, A: Grid, B: Grid, A_shape, B_shape) -> str:
        """Base diff 2 grids"""
        super().format(A, A_shape)
        super().format(B, B_shape)

        if A_shape != B_shape:
            raise ValueError(f"Unexpected shape: A {A_shape}, B {B_shape}")
        return ""


@dataclass
class ListDiffer(Differ):

    @classmethod
    def diff(self, A: Grid, B: Grid, a, b) -> str:
        """List A->B color diffs like adjacency list"""
        super().diff(A, B, a, b)

        diffs = bucket_diffs(A, B)
        output = []

        for (cA, cB), coords in sorted(diffs.items()):
            cells = " ".join(f"{ALPHA[i]}{j+1}" for i, j in sorted(coords))
            output.append(f"{cA}{self.change}{cB}: {cells}")

        return "\n".join(output)


@dataclass
class CellDiffer(Differ):

    @classmethod
    def diff(self, A: Grid, B: Grid, A_shape, B_shape) -> str:
        """Normal grid, but each cell is (ij)->(ij)' diff"""
        super().diff(A, B, A_shape, B_shape)

        rows, cols = A_shape
        # Init with blank cells
        diff_grid = [[self.blank for _ in range(cols)] for _ in range(rows)]

        for i, j in product(range(rows), range(cols)):
            if A[i][j] != B[i][j]:
                diff_grid[i][j] = f"{A[i][j]}{self.change}{B[i][j]}"

        make_row = lambda row: self.sep.join(str(x) for x in row)

        return self.newline.join(map(make_row, diff_grid))


class FormatterNames(str, Enum):
    GRID = "grid"
    SPREADSHEET = "spreadsheet"
    LIST_DIFF = "diff-list"
    CELL_DIFF = "diff-cell"


TASK_PATH = Path(__file__).parent.parent.parent / "tasks"
FORMATTERS = {
    FormatterNames.GRID: GridFormatter,
    FormatterNames.SPREADSHEET: SpreadsheetFormatter,
    FormatterNames.LIST_DIFF: ListDiffer,
    FormatterNames.CELL_DIFF: CellDiffer,
}
