from src.dsl.arc_types import Shape
from pathlib import Path
import orjson
import dataclasses
from src.utils.grid import FORMATTERS, FormatterNames

# root is recurse until .git found
ROOT = Path(__file__).resolve().parents[0]
while not (ROOT / ".git").exists():
    ROOT = ROOT.parent


def serialize(obj) -> str:
    if isinstance(obj, frozenset):
        return list(obj)
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, Path):
        return str(obj)
    # elif is dataclass
    elif dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    # Check if object is a pydantic model
    elif hasattr(obj, "dict") and callable(obj.dict):
        return obj.dict()
    elif callable(obj):
        return f"Function {obj.__name__}"

    raise TypeError(f"Object of type {type(obj)} is not JSON serializable: {obj}")


def jdumps(data: dict) -> str:
    return orjson.dumps(data, option=orjson.OPT_INDENT_2, default=serialize).decode()


def get_shape(O) -> Shape:
    n = len(O)
    m = len(O[0])
    assert all(len(row) == m for row in O)
    return (n, m)
