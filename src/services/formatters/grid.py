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

# (A - Z) + (AA - AD)
ALPHA = list(chr(ord("A") + i) for i in range(26)) + list(
    f"A{chr(ord('A') + i)}" for i in range(4)
)

Grid = TypeVar("Grid", bound=np.ndarray)


def bucket_diffs(g1: Grid, g2: Grid) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """Bucket diffs by index (color)"""
    x, y = (g1 != g2).nonzero()
    diffs = defaultdict(list)

    for i, j in zip(x, y, strict=True):
        pair = (g1[i, j], g2[i, j])
        diffs[pair].append((int(i), int(j)))

    return diffs


@dataclass
class Formatter(ABC):
    sep: str = "|"
    newline: str = "\n"
    n_max: int = 30

    @classmethod
    def format(cls, grid: Grid) -> str:
        """Base formatter"""
        if grid.shape[0] > cls.n_max or grid.shape[1] > cls.n_max:
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
    def format(cls, grid: Grid) -> str:
        super().format(grid)
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
    def format(cls, grid: Grid) -> str:
        super().format(grid)
        _, n_cols = grid.shape

        header = "\t" + cls.sep.join([""] + ALPHA[:n_cols])
        rows = "\n".join(
            str(i + 1) + "\t" + cls.sep.join([str(x) for x in row])
            for i, row in enumerate(grid)
        )
        return cls.newline.join([header, rows])


@dataclass
class Differ(Formatter):
    change: str = " -> "
    blank: str = "  --  "

    @classmethod
    def diff(self, A: Grid, B: Grid) -> str:
        """Base diff 2 grids"""
        super().format(A)
        super().format(B)

        if A.shape != B.shape:
            raise ValueError(f"Unexpected shape: A {A.shape}, B {B.shape}")
        return ""


@dataclass
class ListDiffer(Differ):

    @classmethod
    def diff(self, A: Grid, B: Grid) -> str:
        """List A->B color diffs like adjacency list"""
        super().diff(A, B)

        diffs = bucket_diffs(A, B)
        output = []

        for (cA, cB), coords in sorted(diffs.items()):
            cells = " ".join(f"{ALPHA[i]}{j+1}" for i, j in sorted(coords))
            output.append(f"{cA}{self.change}{cB}: {cells}")

        return "\n".join(output)


@dataclass
class CellDiffer(Differ):

    @classmethod
    def diff(self, A: Grid, B: Grid) -> str:
        """Normal grid, but each cell is (ij)->(ij)' diff"""
        super().diff(A, B)

        rows, cols = A.shape
        # Init with blank cells
        diff_grid = [[self.blank for _ in range(cols)] for _ in range(rows)]

        for i, j in product(range(rows), range(cols)):
            if A[i, j] != B[i, j]:
                diff_grid[i][j] = f"{A[i,j]}{self.change}{B[i,j]}"

        make_row = lambda row: self.sep.join(str(x) for x in row)

        return self.newline.join(map(make_row, diff_grid))


app = typer.Typer()


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


@app.command()
def format(
    id: str, filename: str = "train.json", type: FormatterNames = FormatterNames.GRID
):
    """Format or diff all grids in ARC task"""
    with open(TASK_PATH / filename) as f:
        task = json.load(f)[id]

    formatter = FORMATTERS[type]
    is_diff = type in {FormatterNames.LIST_DIFF, FormatterNames.CELL_DIFF}

    grids = task["train"] + task["test"]

    for i, g in enumerate(grids, 1):
        print(f"> Grid {i if i < len(grids) else 'TEST'}")

        A = np.array(g["input"])
        B = np.array(g.get("output", []))

        # B.size skips held-out test grid output
        if is_diff:
            if B.size:
                diff = formatter.diff(A, B)
                debug(diff)
        else:
            finput = formatter.format(A)
            debug(finput)

            if B.size:
                foutput = formatter.format(B)
                debug(foutput)


if __name__ == "__main__":
    app()
