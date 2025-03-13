from src.dsl.arc_types import *
from typing import *
from dataclasses import dataclass
from types import CodeType
from functools import cached_property
from src.utils.grid import FORMATTERS, FormatterNames
import inspect
from pydantic import BaseModel, Field


def grid_shape(grid: Grid) -> Shape:
    assert all(len(r) == len(r0 := grid[0]) for r in grid)
    return (len(grid), len(r0))


ExampleSplit = Literal["train", "test"]  # ICL
TaskSplit = Literal["train", "eval"]  # Dataset


@dataclass
class Example:
    I: Grid
    O: Grid
    split: ExampleSplit
    n: int

    def __post_init__(self):
        # Assert I,O are tuples of tuples
        assert isinstance(self.I, tuple)
        assert isinstance(self.O, tuple)
        assert all(isinstance(r, tuple) for r in self.I)
        assert all(isinstance(r, tuple) for r in self.O)

    @cached_property
    def I_shape(self):
        return grid_shape(self.I)

    @cached_property
    def O_shape(self):
        return grid_shape(self.O)

    def format(
        self, style=FormatterNames.SPREADSHEET, diff_style=FormatterNames.CELL_DIFF
    ):
        fmt1 = FORMATTERS[style].format
        fmt2 = FORMATTERS[diff_style].diff

        s = f"Input {self.n+1}"
        s += "\n\n"
        s += fmt1(self.I, self.I_shape)
        s += "\n\n"
        s += f"Output {self.n+1}"
        s += "\n\n"
        s += fmt1(self.O, self.O_shape)

        if self.I_shape == self.O_shape:
            s += "\n\n"
            s += f"Diff {self.n+1} (I->O)"
            s += "\n\n"
            s += fmt2(self.I, self.O, self.I_shape, self.O_shape)

        return s

    def __repr__(self):
        return self.format()

    def __str__(self):
        return self.format()


class TaskDef(BaseModel):
    id: str
    split: TaskSplit
    train: List[Example]
    test: List[Example]

    keeps_shape: bool

    solver: Optional[Callable] = None
    verifier: Optional[Callable] = None
    generator: Optional[Callable] = None

    @cached_property
    def keeps_shape(self):
        return all(ex.I_shape == ex.O_shape for ex in self.examples)

    @property
    def examples(self):
        return self.train + self.test

    @property
    def solver_repr(self) -> str:
        # replace def solve_*(I): with def solve(I):
        try:
            source = inspect.getsource(self.solver)
        except TypeError:
            debug(f"Solver not found for {self.id}: {self.solver}")
            return None
        sanitized_source = source.replace(f"def solve_{self.id}(I):", "def solve(I):")
        return sanitized_source

    @property
    def verifier_repr(self) -> str:
        try:
            source = inspect.getsource(self.verifier)
        except TypeError:
            debug(f"Verifier not found for {self.id}: {self.verifier}")
            return None
        sanitized_source = source.replace(f"def verify_{self.id}", "def solve")
        return sanitized_source
