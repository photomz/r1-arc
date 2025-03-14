from src.dsl.arc_types import *
from src.dsl.constants import *
from src.dsl.dsl import *
from src.dsl.main import get_data

# from viztracer import log_sparse, VizTracer
import json
import tempfile
import sys


class VarTracer:  # (VizTracer):
    def __init__(self, *args, **kwargs):
        # super().__init__(
        #     *args, log_func_args=True, log_func_retval=True, log_print=True, **kwargs
        # )
        self.vartraces = []

    def log_locals(self, locale):
        # filter out non-serializable objects
        localvars = {}
        for name, value in locale:
            # if (
            #     not callable(value)
            #     and type(value) != "function"
            #     and isinstance(value, (int, float, str, tuple, list, dict))
            # ):
            # assert name in ["I", "O"] or name[0] == "x", name
            localvars[name] = value
        self.vartraces.append(localvars)

    def export(self):
        self.stop()

        return self.vartraces

        # Create a temporary file that gets automatically cleaned up
        with tempfile.NamedTemporaryFile(suffix=".json") as tmpf:
            self.save(tmpf.name)
            with open(tmpf.name, "r") as f:
                trace_data = json.load(f)
        var_events = list(
            filter(
                lambda d: d.get("cat") == "INSTANT",
                trace_data["traceEvents"],
            )
        )
        varlocals = {}
        for e in var_events:
            varliteral = e["args"]["object"]
            try:
                varvalue = eval(varliteral)  # WARN: Distorts 2d tuple shapes
                varlocals[e["name"]] = varvalue
            except SyntaxError as err:
                debug(e["name"], varliteral)
        self.clear()
        self.traces.append(varlocals)
        return varlocals


if __name__ == "__main__":
    key = "e9afcf9a"
    ex = get_data()["train"][key][0]

    I = ex["input"]

    tracer = VarTracer()
    tracer.start()

    @log_sparse(stack_depth=10)
    def solve_e9afcf9a(I):
        x1 = astuple(TWO, ONE)
        x2 = crop(I, ORIGIN, x1)
        x3 = hmirror(x2)
        x4 = hconcat(x2, x3)
        x5 = hconcat(x4, x4)
        O = hconcat(x5, x4)

        for name, value in locals().items():
            # can't move locals() to func, changes frame.
            tracer.log_var(name, value)
        return O

    O = solve_e9afcf9a(I)
    assert O == ex["output"]

    debug(tracer.export())
