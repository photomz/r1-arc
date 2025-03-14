from src.utils import TASKS


def test_many_test_examples(snapshot):
    """Should not leak test outputs"""
    assert TASKS["794b24be"].format_prompt() == snapshot
