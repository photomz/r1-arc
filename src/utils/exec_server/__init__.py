"""
Python Execution Server for ARC

This module provides a secure environment for executing Python code in ARC problems.
It includes:
- Code execution with resource limits
- Static analysis of code
- Error detection and auto-repair
- Caching for performance
- CLI for running solvers on tasks
"""

# Import main components for easy access
from src.utils.exec_server.execution import (
    run_python_transform_sync,
    run_python_transform_async,
    run_python_transforms,
    PythonResult,
    PythonException,
    ResourceLimits,
    TransformInput
)

from src.utils.exec_server.code_analysis import (
    CodeMetrics,
    collect_static_metrics,
    attempt_code_repair
)

# Import CLI
from src.utils.exec_server.cli import app as cli_app

# Version info
__version__ = "0.1.0"