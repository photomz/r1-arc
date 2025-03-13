from pathlib import Path

# root is recurse until .git found
ROOT = Path(__file__).resolve().parents[0]
while not (ROOT / ".git").exists():
    ROOT = ROOT.parent
