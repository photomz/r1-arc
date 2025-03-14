from typing import Any
from pathlib import Path


def deep_tuple(a: Any) -> Any:
    if isinstance(a, list):
        return tuple(deep_tuple(x) for x in a)
    return a


# root is recurse until .git found
ROOT = Path(__file__).resolve().parents[0]
while not (ROOT / ".git").exists():
    ROOT = ROOT.parent
