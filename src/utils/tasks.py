from src.dsl.arc_types import *
from typing import *
import src.dsl.solvers as solvers_module
import src.re_arc.verifiers as verifiers_module
import src.re_arc.generators as generators_module

from src.utils.types import ROOT
import orjson as json
import inspect
from src.utils.models import Example, TaskDef
from src.utils.types import deep_tuple

DATA_DIR = ROOT / "src/data"


def get_tasks() -> Dict[str, TaskDef]:
    tasks = {}
    for split in {"train", "eval"}:
        with (DATA_DIR / f"{split}.jsonl").open() as f:
            lines = [json.loads(line) for line in f]
        for l in lines:
            id = l["task_id"]
            solver, verifier, generator = None, None, None
            if split == "train":
                solver = getattr(solvers_module, f"solve_{id}", None)
                verifier = getattr(verifiers_module, f"verify_{id}", None)
                generator = getattr(generators_module, f"generate_{id}", None)
            tasks[id] = TaskDef(
                id=id,
                split=split,
                train=[
                    Example(
                        I=deep_tuple(ex["input"]),
                        O=deep_tuple(ex["output"]),
                        split="train",
                        n=i,
                    )
                    for i, ex in enumerate(l["train"])
                ],
                test=[
                    Example(
                        I=deep_tuple(ex["input"]),
                        O=deep_tuple(ex["output"]),
                        split="test",
                        n=i,
                    )
                    for i, ex in enumerate(l["test"])
                ],
                solver=solver,
                verifier=verifier,
                generator=generator,
            )

    return tasks


TASKS = get_tasks()

if __name__ == "__main__":
    debug(list(TASKS.items())[:4])
