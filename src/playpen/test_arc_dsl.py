import sys
from pathlib import Path

# Add arc-dsl to Python path
root = Path(__file__).resolve()
while not (root / ".git").exists():
    root = root.parent
sys.path.append(str(root / "arc-dsl"))

from dsl import identity, add  # type: ignore

if __name__ == "__main__":
    debug(identity(1))
    debug(add(3, 4))
