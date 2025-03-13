"""
Execution module for the Python execution server.

This module contains functions for executing Python code in a safe environment.
"""

import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast
import jinja2
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import Terminal256Formatter

from pydantic import BaseModel, Field

from src.utils.devtools import debug
from src.utils.exec_server.code_analysis import (
    CodeMetrics,
    collect_static_metrics,
    attempt_code_repair,
    ErrorType,
    format_error,
)

# Setup Jinja2 environment for templates
TEMPLATE_DIR = Path(__file__).parent / "templates"
GIT_DIR = Path(__file__).parents[2]
jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(TEMPLATE_DIR),
    trim_blocks=True,
    lstrip_blocks=True,
)


class PythonResult(BaseModel):
    """Result of executing Python code"""

    stdout: Optional[str] = None
    stderr: Optional[str] = None
    return_code: int = 0
    timed_out: bool = False
    latency_ms: float = 0
    transform_results: Optional[List[List[List[int]]]] = None
    debug_outputs: List[str] = Field(default_factory=list)  # Added debug outputs

    # Metrics for code analysis
    metrics: Dict[str, Any] = Field(default_factory=dict)


class PythonException(Exception):
    """Exception for Python execution errors"""

    pass


class ResourceLimits(BaseModel):
    """Resource limits for code execution"""

    max_memory_bytes: Optional[int] = None  # Memory limit in bytes
    max_time_seconds: int = 30  # Time limit in seconds (loosened)
    max_subprocesses: int = 0  # Subprocesses limit (0 = no subprocesses)


class ResultCache:
    """Simple in-memory cache for Python execution results"""

    _cache: Dict[str, PythonResult] = {}

    @classmethod
    def get(cls, key: str) -> Optional[PythonResult]:
        """Get a result from cache if it exists"""
        return cls._cache.get(key)

    @classmethod
    def set(cls, key: str, result: PythonResult) -> None:
        """Set a result in the cache"""
        cls._cache[key] = result

    @classmethod
    def clear(cls) -> None:
        """Clear the cache"""
        cls._cache.clear()


def create_wrapped_code(
    code: str, grid_lists: List[List[List[int]]], allowed_modules: Set[str]
) -> str:
    """
    Create the wrapped code to execute in the subprocess using Jinja2 template

    Args:
        code: User code containing solve or transform function
        grid_lists: Input grids to transform
        allowed_modules: Set of allowed module names

    Returns:
        Complete Python code to execute
    """
    debug("Creating wrapped code for execution using Jinja2 template")
    try:
        template = jinja_env.get_template("wrapper.jinja2")
        return template.render(
            user_code=code,
            grid_lists_json=json.dumps(grid_lists),
            allowed_modules=allowed_modules,
        )
    except jinja2.exceptions.TemplateError as e:
        debug(f"Error rendering template: {e}")
        # Fallback to direct string interpolation if template rendering fails
        template_path = TEMPLATE_DIR / "wrapper.jinja2"
        if template_path.exists():
            with open(template_path, "r") as f:
                template_content = f.read()
                return (
                    template_content.replace("{{ user_code }}", code)
                    .replace("{{ grid_lists_json }}", json.dumps(grid_lists))
                    .replace("{{ allowed_modules }}", repr(allowed_modules))
                )
        else:
            raise RuntimeError(
                f"Template file not found at {template_path} and Jinja2 rendering failed"
            )


def drop_privileges() -> None:
    """Drop privileges of the current process for security"""
    debug("Attempting to drop privileges")
    # This is a no-op on many systems but useful on Linux/Unix
    # On Windows, you'd typically use job objects to limit privileges
    if hasattr(os, "setgid") and hasattr(os, "setuid"):
        try:
            # Set group and user to nobody or similar low-privilege user if available
            os.setgid(65534)  # nobody group
            os.setuid(65534)  # nobody user
            debug("Successfully dropped privileges")
        except Exception as e:
            debug(f"Failed to drop privileges: {e}")
            # Ignore if setting privileges fails
            pass


def get_adaptive_timeout(failure_count: int, base_timeout: int = 30) -> int:
    """
    Calculate an adaptive timeout based on previous failures

    Args:
        failure_count: Number of previous failures
        base_timeout: Base timeout in seconds

    Returns:
        Adjusted timeout in seconds
    """
    timeout = min(base_timeout * (1 + 0.5 * failure_count), 120)
    debug(f"Using adaptive timeout of {timeout}s (failure count: {failure_count})")
    return timeout


def cache_key(code: str, grid_lists: List[Any]) -> str:
    """
    Create a cache key from code and input grids

    Args:
        code: The code to execute
        grid_lists: The input grids

    Returns:
        MD5 hash string to use as cache key
    """
    serialized = code + json.dumps(grid_lists)
    return hashlib.md5(serialized.encode()).hexdigest()


def parse_transform_results(stdout: str) -> Tuple[Optional[List[List[List[int]]]], List[str]]:
    """
    Parse transform results and debug output from stdout

    Args:
        stdout: Standard output from subprocess

    Returns:
        Tuple of (transform_results, debug_outputs) where:
        - transform_results: Parsed transform results or None if parsing failed
        - debug_outputs: List of debug output messages
    """
    debug("Parsing transform results and debug output from stdout")
    transform_results = None
    debug_outputs = []
    
    if stdout:
        for line in stdout.splitlines():
            if line.startswith("TRANSFORM_RESULT:"):
                try:
                    transform_results = json.loads(
                        line.replace("TRANSFORM_RESULT:", "", 1)
                    )
                    debug(f"Successfully parsed transform results")
                except json.JSONDecodeError as e:
                    debug(f"Failed to parse transform results: {e}")
            elif line.startswith("DEBUG_OUTPUT:"):
                # Extract debug output
                debug_message = line.replace("DEBUG_OUTPUT:", "", 1).strip()
                debug_outputs.append(debug_message)
                debug(f"Found debug output: {debug_message[:50]}...")
    
    return transform_results, debug_outputs


def validate_output_shape(
    transform_results: List[List[List[int]]], grid_lists: List[List[List[int]]]
) -> bool:
    """
    Validate that output shape is correct

    Args:
        transform_results: The transform results
        grid_lists: The input grids

    Returns:
        True if all output shapes are valid
    """
    debug("Validating output shape")
    if transform_results and grid_lists:
        all_correct = True
        for i, (result, input_grid) in enumerate(zip(transform_results, grid_lists)):
            if not isinstance(result, list) or not result:
                debug(f"Result {i} is not a valid grid")
                all_correct = False
                break
            # Optional deeper validation could be added here
        return all_correct
    return False


def execute_subprocess(
    temp_file: str,
    timeout: int,
) -> Tuple[str, str, int, bool]:
    """
    Execute a Python subprocess with the given file

    Args:
        temp_file: Path to Python file to execute
        timeout: Timeout in seconds

    Returns:
        Tuple of (stdout, stderr, return_code, timed_out)
    """
    debug(f"Executing subprocess with timeout {timeout}s")

    def set_limits():
        import resource

        # Set file descriptor limit to minimum required (3 for stdin/stdout/stderr)
        # This prevents opening new files while keeping essential streams
        resource.setrlimit(resource.RLIMIT_NOFILE, (3, 3))

        # Drop privileges
        drop_privileges()

    # Create restricted environment
    restricted_env = os.environ.copy()
    # Remove environment variables that might contain file paths
    for var in ["PYTHONPATH", "PATH", "HOME", "TEMP", "TMP"]:
        restricted_env.pop(var, None)

    process = subprocess.Popen(
        [
            sys.executable,
            "-E",  # -E prevents use of PYTHON* environment variables
            temp_file,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=restricted_env,
        # preexec_fn=set_limits if os.name != "nt" else None,  # Unix only
        shell=False,
        # Prevent access to parent directories
        cwd=os.path.dirname(temp_file),
    )

    stdout = ""
    stderr = ""
    return_code = 0
    timed_out = False

    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return_code = process.returncode
        debug(f"Subprocess completed with return code {return_code}")
    except subprocess.TimeoutExpired:
        debug("Subprocess timed out, killing process")
        process.kill()
        stdout, stderr = process.communicate()
        stderr = f"Execution timed out after {timeout} seconds"
        return_code = -1
        timed_out = True

    return stdout, stderr, return_code, timed_out


def prepare_result(
    stdout: str,
    stderr: str,
    return_code: int,
    timed_out: bool,
    transform_results: Optional[List[List[List[int]]]],
    debug_outputs: List[str],
    latency_ms: float,
    static_metrics: CodeMetrics,
) -> PythonResult:
    """
    Prepare the final result object

    Args:
        stdout: Standard output from subprocess
        stderr: Standard error from subprocess
        return_code: Return code from subprocess
        timed_out: Whether execution timed out
        transform_results: Parsed transform results
        debug_outputs: Parsed debug outputs
        latency_ms: Execution time in milliseconds
        static_metrics: Static code metrics

    Returns:
        PythonResult object
    """
    debug("Preparing result object")
    static_metrics.execution_success = (
        return_code == 0 and transform_results is not None
    )

    return PythonResult(
        stdout=stdout,
        stderr=stderr,
        return_code=return_code,
        timed_out=timed_out,
        latency_ms=latency_ms,
        transform_results=transform_results,
        debug_outputs=debug_outputs,
        metrics={
            "static": static_metrics.dict(),
            "runtime": {
                "latency_ms": latency_ms,
                "return_code": return_code,
                "timed_out": timed_out,
            },
        },
    )


def generate_filename(code: str) -> str:
    """Generate a filename based on datetime and code hash"""
    timestamp = time.strftime("%Y%m%d_%H%M")  # Format: YYYYMMDD_HHMM
    code_hash = hashlib.md5(code.encode()).hexdigest()[:8]
    return f"{timestamp}-{code_hash}.py"


def run_python_transform_sync(
    code: str,
    grid_lists: List[List[List[int]]],
    resource_limits: Optional[ResourceLimits] = None,
    raise_exception: bool = False,
    use_cache: bool = True,
    failure_count: int = 0,
    allowed_modules: Optional[Set[str]] = None,
) -> PythonResult:
    """
    Execute a Python string containing a transform function and call it with the provided grid.

    Args:
        code: Python code containing transform function
        grid_lists: Input grids to transform
        resource_limits: Optional resource limits for execution
        raise_exception: Whether to raise exceptions on failure
        use_cache: Whether to use the result cache
        failure_count: Number of previous failures (for adaptive timeout)
        allowed_modules: Set of allowed module names

    Returns:
        PythonResult containing execution results and transformed grid
    """
    debug(f"Running Python transform (failure_count={failure_count})")

    # Initialize defaults
    if resource_limits is None:
        resource_limits = ResourceLimits()

    if allowed_modules is None:
        allowed_modules = {
            "numpy",
            "np",  # Common data manipulation
            "typing",  # Type hints
            "collections",  # Container datatypes
            "itertools",  # Iterator functions
            "functools",  # Higher-order functions
            "math",  # Math functions
            "random",  # Random number generation
            "json",
            "sys",
            "os",  # Standard libraries
            "arc_types",  # ARC DSL type definitions
            "dsl",
            "arc_dsl",  # ARC DSL functions
        }

    # Try to get from cache if enabled
    if use_cache:
        cache_key_str = cache_key(code, grid_lists)
        cached_result = ResultCache.get(cache_key_str)
        if cached_result is not None:
            debug("Using cached result")
            return cached_result

    # Collect static metrics
    static_metrics = collect_static_metrics(code)

    # Start timing
    start = time.time()

    # Create wrapped code
    wrapped_code = create_wrapped_code(code, grid_lists, allowed_modules)

    print(highlight(wrapped_code, PythonLexer(), Terminal256Formatter()))

    # Create file in src/tmp directory with datetime and hash
    filename = generate_filename(code)
    file_path = GIT_DIR / "tmp" / filename

    with open(file_path, "w") as f:
        f.write(wrapped_code)
        debug(f"Created file: {file_path}")

    try:
        # Calculate timeout using adaptive algorithm
        timeout = get_adaptive_timeout(failure_count, resource_limits.max_time_seconds)

        # Execute the code in a subprocess
        stdout, stderr, return_code, timed_out = execute_subprocess(file_path, timeout)

        debug(stdout, stderr, return_code, timed_out)

        # Parse transform results and debug output
        transform_results, debug_outputs = parse_transform_results(stdout)

        # Calculate execution time
        latency_ms = (time.time() - start) * 1000
        debug(f"Execution completed in {latency_ms:.2f}ms")

        # Attempt to repair and retry if execution failed
        if (return_code != 0 or not transform_results) and failure_count < 2:
            repaired_code = attempt_code_repair(code, stderr)
            if repaired_code != code:
                # We made a repair, try again with the repaired code
                debug("Using repaired code for retry")
                return run_python_transform_sync(
                    repaired_code,
                    grid_lists,
                    resource_limits,
                    raise_exception,
                    use_cache,
                    failure_count + 1,
                    allowed_modules,
                )

        # Validate output shape
        if transform_results:
            static_metrics.output_shape_correct = validate_output_shape(
                transform_results, grid_lists
            )

        # Prepare the result
        result = prepare_result(
            stdout=stdout,
            stderr=stderr,
            return_code=return_code,
            timed_out=timed_out,
            transform_results=transform_results,
            debug_outputs=debug_outputs,
            latency_ms=latency_ms,
            static_metrics=static_metrics,
        )

        # Raise exception if requested and execution failed
        if not transform_results and raise_exception:
            formatted_error = format_error(stderr, code)
            debug(f"Raising exception with formatted error")
            raise PythonException(formatted_error)

        # Cache successful results
        if use_cache and transform_results:
            debug("Caching successful result")
            ResultCache.set(cache_key(code, grid_lists), result)

        return result

    finally:
        # Clean up the temporary file
        # os.unlink(file_path)
        debug(f"Cleaned up temporary file")


async def run_python_transform_async(
    code: str,
    grid_lists: List[List[List[int]]],
    resource_limits: Optional[ResourceLimits] = None,
    raise_exception: bool = False,
    use_cache: bool = True,
    allowed_modules: Optional[Set[str]] = None,
) -> Optional[PythonResult]:
    """
    Async wrapper for run_python_transform_sync

    Args:
        code: Python code containing transform function
        grid_lists: Input grids to transform
        resource_limits: Optional resource limits for execution
        raise_exception: Whether to raise exceptions on failure
        use_cache: Whether to use the result cache
        allowed_modules: Set of allowed module names

    Returns:
        PythonResult containing execution results and transformed grid
    """
    debug("Running Python transform async")
    try:
        # Use asyncio.to_thread in Python 3.9+, or run_in_executor in earlier versions
        if hasattr(asyncio, "to_thread"):
            debug("Using asyncio.to_thread")
            result = await asyncio.to_thread(
                run_python_transform_sync,
                code=code,
                grid_lists=grid_lists,
                resource_limits=resource_limits,
                raise_exception=raise_exception,
                use_cache=use_cache,
                allowed_modules=allowed_modules,
            )
        else:
            debug("Using loop.run_in_executor")
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: run_python_transform_sync(
                    code=code,
                    grid_lists=grid_lists,
                    resource_limits=resource_limits,
                    raise_exception=raise_exception,
                    use_cache=use_cache,
                    allowed_modules=allowed_modules,
                ),
            )
        return result
    except Exception as e:
        debug(f"ERROR RUNNING PYTHON: {e}")
        return None


class TransformInput(BaseModel):
    """Input for transform function"""

    code: str
    grid_lists: List[List[List[int]]]
    resource_limits: Optional[ResourceLimits] = None
    raise_exception: bool = False
    use_cache: bool = True
    allowed_modules: Optional[Set[str]] = None


async def run_python_transforms(
    inputs: List[TransformInput],
) -> List[Optional[PythonResult]]:
    """
    Run multiple Python transforms in parallel

    Args:
        inputs: List of transform inputs

    Returns:
        List of transform results
    """
    debug(f"Running {len(inputs)} Python transforms in parallel")
    return await asyncio.gather(
        *[
            run_python_transform_async(
                input.code,
                input.grid_lists,
                input.resource_limits,
                input.raise_exception,
                input.use_cache,
                input.allowed_modules,
            )
            for input in inputs
        ]
    )
