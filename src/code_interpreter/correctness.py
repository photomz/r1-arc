from src.code_interpreter.execution import InterpreterOutput
from src.utils.tasks import deep_tuple
from dataclasses import dataclass


@dataclass
class InterpreterValidation:
    pass


# def validate(res: InterpreterOutput) -> InterpreterValidation:
#     assert trace.metrics.syntax_ok and trace.metrics.run_ok

#     shell_Is = [deep_tuple(run["I"]) for run in trace.traces]
#     debug(shell_Is)
#     assert shell_Is == test_Is

#     shell_Os = [deep_tuple(run["O"]) for run in trace.traces]
#     test_Os = [deep_tuple(ex.O) for ex in task.train + task.test]
#     debug(shell_Os, test_Os)
#     assert shell_Os == test_Os

#     assert trace == snapshot
