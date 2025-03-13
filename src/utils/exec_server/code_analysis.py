"""
Code analysis module for the Python execution server.

This module contains functions for analyzing Python code, including:
- Collecting static metrics
- Analyzing AST nodes
- Detecting DSL function calls
"""

import ast
import difflib
import json
from pathlib import Path
import re
import sys
from typing import Dict, List, Set, Optional, Any, Type, cast
from enum import Enum, auto

from pydantic import BaseModel, Field
from src.utils.devtools import debug
from src.utils import ROOT

dsl_info = json.load((ROOT / "src/dsl/out/dsl_typedefs.json").open())


class CodeMetrics(BaseModel):
    """Metrics for code analysis"""

    num_lines: int = 0
    avg_line_length: float = 0
    num_if_statements: int = 0
    num_for_loops: int = 0
    num_while_loops: int = 0
    dsl_functions_used: Set[str] = Field(default_factory=set)
    imports_used: Set[str] = Field(default_factory=set)
    ast_node_counts: Dict[str, int] = Field(default_factory=dict)
    compilation_success: bool = False
    execution_success: bool = False
    output_shape_correct: bool = False


def collect_static_metrics(code: str) -> CodeMetrics:
    """
    Collect static metrics from code using AST parsing

    Args:
        code: The Python code to analyze

    Returns:
        CodeMetrics object with various code metrics
    """
    debug("Collecting static metrics for code")
    metrics = CodeMetrics()

    # Basic string metrics
    lines = code.strip().split("\n")
    metrics.num_lines = len(lines)
    if lines:
        metrics.avg_line_length = sum(len(line) for line in lines) / len(lines)

    # Parse the AST
    try:
        tree = ast.parse(code)
        metrics.compilation_success = True
        debug(f"AST parsed successfully, code compiles")

        # Track node types
        node_counts: Dict[str, int] = {}
        # Analyze each node in the AST
        for node in ast.walk(tree):
            _analyze_ast_node(node, node_counts, metrics, dsl_info["functions"])

        metrics.ast_node_counts = node_counts
        debug(
            f"AST analysis complete: {metrics.num_if_statements} ifs, {metrics.num_for_loops} for loops"
        )

    except SyntaxError as e:
        debug(f"AST parsing failed: {e}")
        metrics.compilation_success = False

    return metrics


def _analyze_ast_node(
    node: ast.AST,
    node_counts: Dict[str, int],
    metrics: CodeMetrics,
    dsl_functions: Set[str],
) -> None:
    """
    Analyze an AST node and update metrics accordingly

    Args:
        node: The AST node to analyze
        node_counts: Dictionary to track node type counts
        metrics: CodeMetrics object to update
        dsl_functions: Set of DSL function names to look for
    """
    # Count node types
    node_type = type(node).__name__
    node_counts[node_type] = node_counts.get(node_type, 0) + 1

    # Count specific constructs
    if isinstance(node, ast.If):
        metrics.num_if_statements += 1
    elif isinstance(node, ast.For):
        metrics.num_for_loops += 1
    elif isinstance(node, ast.While):
        metrics.num_while_loops += 1

    # Extract imports
    if isinstance(node, ast.Import):
        for name in node.names:
            metrics.imports_used.add(name.name)
    elif isinstance(node, ast.ImportFrom):
        if node.module:
            metrics.imports_used.add(node.module)

    # Try to detect DSL function calls
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name in dsl_functions:
                metrics.dsl_functions_used.add(func_name)
        elif isinstance(node.func, ast.Attribute) and isinstance(
            node.func.value, ast.Name
        ):
            # Check for module.function() style calls
            if node.func.value.id == "dsl" and node.func.attr in dsl_functions:
                metrics.dsl_functions_used.add(f"dsl.{node.func.attr}")


class ErrorType(Enum):
    """Error type enumeration for more type-safe error handling"""

    NAME_ERROR = auto()
    INDEX_ERROR = auto()
    TYPE_ERROR = auto()
    VALUE_ERROR = auto()
    KEY_ERROR = auto()
    ATTRIBUTE_ERROR = auto()
    ZERO_DIVISION_ERROR = auto()
    SYNTAX_ERROR = auto()
    UNKNOWN = auto()


def identify_error_type(error_msg: str) -> ErrorType:
    """
    Identify the type of error from an error message

    Args:
        error_msg: The error message

    Returns:
        ErrorType enum value
    """
    error_patterns = {
        ErrorType.NAME_ERROR: ["NameError"],
        ErrorType.INDEX_ERROR: ["IndexError"],
        ErrorType.TYPE_ERROR: ["TypeError"],
        ErrorType.VALUE_ERROR: ["ValueError"],
        ErrorType.KEY_ERROR: ["KeyError"],
        ErrorType.ATTRIBUTE_ERROR: ["AttributeError"],
        ErrorType.ZERO_DIVISION_ERROR: ["ZeroDivisionError", "division by zero"],
        ErrorType.SYNTAX_ERROR: ["SyntaxError"],
    }

    for error_type, patterns in error_patterns.items():
        if any(pattern in error_msg for pattern in patterns):
            return error_type

    return ErrorType.UNKNOWN


def attempt_code_repair(code: str, error_msg: str) -> str:
    """
    Attempt to automatically repair common code errors

    Args:
        code: The original code
        error_msg: The error message from execution

    Returns:
        Repaired code or the original code if no repair was possible
    """
    debug(error_msg)
    debug(f"Attempting to repair code based on error: {error_msg[:100]}...")

    # Identify error type
    error_type = identify_error_type(error_msg)

    # Create repaired code - initially set to original
    repaired_code = code

    # Apply repairs based on error type
    if error_type == ErrorType.NAME_ERROR:
        # Extract the undefined name
        missing_name_match = re.search(r"name '([^']+)' is not defined", error_msg)
        if missing_name_match:
            fname = missing_name_match.group(1)

            # Fix missing numpy import
            if fname == "np":
                debug("Adding numpy import")
                repaired_code = "import numpy as np\n" + code

            # Fix missing DSL function imports
            elif fname in dsl_info["functions"]:
                debug(f"Adding DSL import for {fname}")
                repaired_code = "from dsl import *\n" + code

    elif error_type == ErrorType.INDEX_ERROR:
        debug("Adding bounds checking for array indexing")
        # Add bounds checking in nested loops
        if re.search(r"for\s+\w+\s+in\s+range\(", code):
            lines = code.split("\n")
            for i, line in enumerate(lines):
                if "for" in line and "range(" in line and "in" in line:
                    indent = len(line) - len(line.lstrip())
                    if (
                        i + 1 < len(lines)
                        and "for" in lines[i + 1]
                        and "range(" in lines[i + 1]
                    ):
                        # Nested loop - might need bounds checking
                        inner_indent = len(lines[i + 1]) - len(lines[i + 1].lstrip())
                        loop_var = re.search(r"for\s+(\w+)\s+in", lines[i + 1])
                        if loop_var:
                            var_name = loop_var.group(1)
                            check_line = (
                                " " * inner_indent
                                + f"if {var_name} >= len(grid_list[i]): continue"
                            )
                            lines.insert(i + 2, check_line)
                            debug(f"Added bounds check: {check_line}")
                            repaired_code = "\n".join(lines)

    elif error_type == ErrorType.TYPE_ERROR:
        # Fixing missing return value for NoneType errors
        if "'NoneType' object" in error_msg:
            debug("Attempting to fix missing return value")
            # Check if the function is missing a return
            func_pattern = r"def\s+(solve|transform)"
            func_match = re.search(func_pattern, code)

            if func_match:
                func_name = func_match.group(1)
                lines = code.split("\n")
                func_line = None

                # Find the function definition line
                for i, line in enumerate(lines):
                    if re.search(func_pattern, line):
                        func_line = i
                        break

                if func_line is not None:
                    # Look for the end of the function to add a return
                    indent = len(lines[func_line]) - len(lines[func_line].lstrip())
                    for i in range(func_line + 1, len(lines)):
                        if (
                            i == len(lines) - 1
                            or (len(lines[i]) - len(lines[i].lstrip())) <= indent
                        ):
                            # This is the end of the function or the file
                            # Add a return statement before this line
                            param_name = "I" if func_name == "solve" else "grid_list"
                            return_line = (
                                " " * (indent + 4)
                                + f"return {param_name}  # Auto-added return"
                            )
                            lines.insert(i, return_line)
                            debug(f"Added return statement: {return_line}")
                            repaired_code = "\n".join(lines)
                            break

    # Show diff of changes if code was modified
    if repaired_code != code:
        debug("Code repairs applied. Showing diff:")
        diff = difflib.unified_diff(
            code.splitlines(keepends=True),
            repaired_code.splitlines(keepends=True),
            fromfile="original",
            tofile="repaired",
        )
        debug("".join(diff))
    else:
        debug("No repair found for this error")

    return repaired_code


def format_error(exception: str, code_snippet: str) -> str:
    """
    Format error message to be more helpful

    Args:
        exception: The exception message
        code_snippet: The code that generated the exception

    Returns:
        Formatted error message with context and suggestions
    """
    debug("Formatting error message")

    # Extract the most relevant part of the error
    error_lines = exception.strip().split("\n")
    error_type_str = error_lines[-1] if error_lines else "Unknown error"

    # Get the error type enum
    error_type = identify_error_type(exception)

    # Get line number from traceback if available
    line_match = re.search(r"line (\d+)", exception)
    line_num = int(line_match.group(1)) if line_match else None

    # Add context if we have a line number
    context_str = ""
    if line_num is not None:
        code_lines = code_snippet.split("\n")
        context_start = max(0, line_num - 3)
        context_end = min(len(code_lines), line_num + 2)

        context_str = "\n".join(
            [
                f"{i+1}: {line}" + (" <<<" if i + 1 == line_num else "")
                for i, line in enumerate(code_lines[context_start:context_end])
            ]
        )

    # Get suggestion based on error type
    suggestion = get_suggestion(error_type, exception)

    if context_str:
        return (
            f"{error_type_str}\n\nContext:\n{context_str}\n\nSuggestion: {suggestion}"
        )
    else:
        return f"{error_type_str}\n\nSuggestion: {suggestion}"


def get_suggestion(error_type: ErrorType, error_msg: str) -> str:
    """
    Get a suggestion for fixing a particular error type

    Args:
        error_type: The type of error (enum)
        error_msg: The full error message

    Returns:
        A human-readable suggestion for fixing the error
    """
    suggestions = {
        ErrorType.INDEX_ERROR: "Check your array indices and make sure they are within bounds",
        ErrorType.NAME_ERROR: "You might be using a variable or function that hasn't been defined",
        ErrorType.ATTRIBUTE_ERROR: "You're trying to access an attribute that doesn't exist on the object",
        ErrorType.ZERO_DIVISION_ERROR: "You're dividing by zero somewhere in your code",
        ErrorType.SYNTAX_ERROR: "Your code has a syntax error. Check for missing colons, parentheses, or brackets.",
    }

    # Special cases that need additional context
    if error_type == ErrorType.TYPE_ERROR:
        if "NoneType" in error_msg:
            return (
                "A variable is None when it shouldn't be. Check your function returns"
            )
        else:
            return "You're trying to use operations on an incorrect type. Check that variables have the expected types."
    elif error_type == ErrorType.VALUE_ERROR:
        if "not enough values to unpack" in error_msg:
            return "You're trying to unpack more values than are available"
        else:
            return "You're trying to perform an operation with an inappropriate value."
    elif error_type == ErrorType.KEY_ERROR:
        key_match = re.search(r"KeyError: '([^']+)'", error_msg)
        if key_match:
            key = key_match.group(1)
            return f"The key '{key}' doesn't exist in the dictionary you're trying to access"
        else:
            return "You're trying to access a dictionary key that doesn't exist"

    # Get default suggestion for the error type, or fallback to generic message
    return suggestions.get(
        error_type, "Review your code logic and check for common mistakes"
    )
