"""
Minimal execution module for Python code in a safe environment.
"""

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Dict

import jinja2
import subprocess

from src.dsl import *
from src.utils.devtools import debug
from src.utils.tasks import deep_tuple
from src.code_interpreter.static_analysis import StaticMetrics, inject_logging
from pygments import highlight
from pygments.formatters import TerminalFormatter
from pygments.lexers import PythonLexer
from src.utils import ROOT

# Setup Jinja2 environment for templates
TEMPLATE_DIR = Path(__file__).parent / "templates"
jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(TEMPLATE_DIR),
    trim_blocks=True,
    lstrip_blocks=True,
)


ALLOWED_MODULES = {}  # TODO: Unused feature.
TIMEOUT = 10

TRACER_HOOK = "src.dsl.dsl"
INPUT_VAR = "I"
OUTPUT_VAR = "O"

pprint = lambda t: print(highlight(t, PythonLexer(), TerminalFormatter()))


@dataclass
class SubprocessResponse:
    stdout = ""
    stderr = ""
    return_code = 0
    timed_out = False


@dataclass
class TracerOutput:
    I: List[Grid]
    O: List[Grid]
    O_shapes: List[Shape]
    I_shapes: List[Shape]
    intermediates: List[Dict[str, Any]]

    @staticmethod
    def from_json(data: dict) -> "TracerOutput":
        return TracerOutput(
            I=[deep_tuple(i) for i in data["I"]],
            O=[deep_tuple(o) for o in data["O"]],
            O_shapes=[deep_tuple(shape) for shape in data["O_shapes"]],
            I_shapes=[deep_tuple(shape) for shape in data["I_shapes"]],
            intermediates=[
                {k: deep_tuple(v) for (k, v) in d.items()}
                for d in data["intermediates"]
            ],
        )


@dataclass
class InterpreterOutput:
    response: SubprocessResponse
    traces: TracerOutput
    codestring: str
    inputs: Any
    run_ok: bool = False


def render_template(code: str, input_data: Any) -> str:
    """Create templated code to execute in subprocess"""
    try:
        template = jinja_env.get_template("wrapper.jinja2")
        return template.render(
            code_literal=code,
            input=repr(input_data),
        )
    except jinja2.exceptions.TemplateError as e:
        debug(f"Error rendering template: {e}")
        raise RuntimeError(f"Template rendering failed: {e}")


def execute_in_subprocess(tempfp: str) -> SubprocessResponse:
    """Run a Python subprocess with the given file"""
    process = subprocess.Popen(
        [sys.executable, "-E", tempfp],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ.copy(),
        shell=False,
        cwd=os.path.dirname(tempfp),
    )

    res = SubprocessResponse()

    try:
        stdout, stderr = process.communicate(timeout=TIMEOUT)
        res.stdout = stdout
        res.stderr = stderr
        res.return_code = process.returncode
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        res.stdout = stdout
        res.stderr = f"Code Interpreter timed out after {TIMEOUT}s"
        res.return_code = -1
        res.timed_out = True
    return res


@dataclass
class Interpreter:

    @classmethod
    def run(
        cls, codestring: str, inputs, id, raise_error=True, cleanup=True
    ) -> InterpreterOutput:
        """Execute Python code with transform function and return results"""

        # Create wrapped code and save to file
        injected_codestr = inject_logging(codestring)
        wrapped_code = render_template(injected_codestr, inputs)
        filename = f'{id}-{time.strftime("%Y%m%d_%H%M")}.py'
        tmp_dir = ROOT / "src/tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        file_path = tmp_dir / filename

        with open(file_path, "w") as f:
            f.write(wrapped_code)

        # Execute the code in a subprocess
        res = execute_in_subprocess(file_path)
        traces = None
        if tracefp := res.stdout.strip().split("\n")[-1]:  # See jinja2 template
            try:
                with open(tracefp, "r") as f:
                    traces = TracerOutput.from_json(json.load(f))
                    # debug(traces)
            except FileNotFoundError:
                debug(res.stdout)
                debug(tracefp)
        if res.stderr:
            debug(res.stdout)
            pprint(f"Subprocess {id} " + res.stderr)
            if raise_error:
                raise RuntimeError(res.stderr)

        if not traces:
            debug(injected_codestr)

        run_ok = (
            not res.stderr  # no error str
            and res.return_code == 0  # no error code
            and traces  # read trace
            and traces.O  # trace has output
            and traces.I  # trace has input
            and len(traces.O)
            == len(traces.I)
            == len(inputs)
            == len(traces.intermediates)  # io len match
        )

        if run_ok and cleanup:
            os.unlink(file_path)
            os.unlink(tracefp)

        return InterpreterOutput(
            response=res,
            traces=traces,
            run_ok=run_ok,
            codestring=codestring,
            inputs=inputs,
        )


# async def run_python_transform_async(*args, **kwargs):
#     return await asyncio.to_thread(run_python_transform_sync, *args, **kwargs)
