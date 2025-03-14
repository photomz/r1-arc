from src.dsl.arc_types import *
from typing import *
from dataclasses import dataclass
from types import CodeType
from functools import cached_property
from src.utils.types import deep_tuple
from src.utils.grid import FORMATTERS, FormatterNames
import inspect
from pydantic import BaseModel, Field


def grid_shape(grid: Grid) -> Shape:
    assert all(len(r) == len(r0 := grid[0]) for r in grid)
    return (len(grid), len(r0))


PuzzleSplit = Literal["train", "test"]  # ICL
DatasetSplit = Literal["train", "eval"]  # Dataset


@dataclass
class Example:
    I: Grid
    O: Grid
    split: PuzzleSplit
    n: int

    def __post_init__(self):
        # Assert I,O are tuples of tuples
        self.I = deep_tuple(self.I)
        self.O = deep_tuple(self.O)

        # assert I and O are Tuple[Tuple[int]], depth = 2
        # assert isinstance(self.I, tuple)
        # assert isinstance(self.O, tuple)
        # assert all(isinstance(r, tuple) for r in self.I)
        # assert all(isinstance(r, tuple) for r in self.O)
        # assert all(isinstance(c, int) for r in self.I for c in r)
        # assert all(isinstance(c, int) for r in self.O for c in r)

    @cached_property
    def I_shape(self):
        return grid_shape(self.I)

    @cached_property
    def O_shape(self):
        return grid_shape(self.O)

    def format(
        self,
        style=FormatterNames.SPREADSHEET,
        diff_style=FormatterNames.CELL_DIFF,
        no_output=False,
    ):
        fmt1 = FORMATTERS[style].format
        fmt2 = FORMATTERS[diff_style].diff

        s = f"Input {self.n+1}"
        s += "\n\n"
        s += fmt1(self.I, self.I_shape)

        if not no_output:
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

    def dumps(self) -> Tuple[Grid, Grid]:
        return (deep_tuple(self.I), deep_tuple(self.O))


@dataclass
class TaskDef:
    id: str
    split: DatasetSplit
    train: List[Example]
    test: List[Example]

    solver: Optional[Callable] = None
    verifier: Optional[Callable] = None
    generator: Optional[Callable] = None

    @property
    def data(self):
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

    def format(
        self,
        hold_out=[],
        split: Literal["train", "test", "any"] = "any",
        no_output=False,
    ) -> str:
        arr = []
        if split == "train":
            arr = self.train
        elif split == "test":
            arr = self.test
        elif split == "any":
            arr = self.train + self.test
        s = ""
        for i, ex in enumerate(arr):
            # Positive and negative indexing
            if i in hold_out or i - len(arr) in hold_out:
                continue
            s += ex.format(no_output=no_output)
            s += "\n\n"
        return s

    def format_prompt(self):
        return (
            "## EXAMPLES\n"
            + self.format(split="train")
            + "\n## TEST\n"
            + self.format(split="test", no_output=True)
        )

    def dumps(self):
        return {
            "id": self.id,
            "train": [ex.dumps() for ex in self.train],
            "test": [ex.dumps() for ex in self.test],
            "solver": self.solver_repr if self.solver else None,
            "augment": None,
        }

    @classmethod
    def from_hf(self, x) -> "TaskDef":

        ex = {
            split: [
                Example(I=i, O=o, split=split, n=n) for n, (i, o) in enumerate(x[split])
            ]
            for split in ["train", "test"]
        }  # should recursively init, oh well depth = 1.

        return TaskDef(
            id=x["id"],
            split="split",
            train=ex["train"],
            test=ex["test"],
        )
