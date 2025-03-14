"""
Code analysis module for the Python execution server.

This module contains functions for analyzing Python code, including:
- Collecting static metrics
- Analyzing AST nodes
- Detecting DSL function calls
"""

import ast
import json
from typing import Set
import libcst as cst
from dataclasses import dataclass
from pydantic import BaseModel, Field
from src.utils.devtools import debug
from src.utils import ROOT

dsl_info = json.load((ROOT / "src/dsl/out/dsl_typedefs.json").open())
dsl_functions = dsl_info["functions"]


class StaticMetrics(BaseModel):
    """AST static analysis of generated code literal"""

    n_lines: int = 0
    n_branch: int = 0
    n_loop: int = 0
    n_returns: int = 0
    n_funcs: int = 0
    n_dsl: Set[str] = Field(default_factory=set)
    imports: Set[str] = Field(default_factory=set)
    syntax_ok: bool = False
    # No error (str, code), has trace, same len I, O, intermediates
    run_ok: bool = False
    format_ok: bool = False

    @staticmethod
    def from_literal(codestr: str) -> "StaticMetrics":
        metrics = StaticMetrics()
        metrics.n_lines = len(lines := codestr.strip().split("\n"))
        metrics.format_ok = validate_solve_function_cst(codestr)

        try:
            tree = ast.parse(codestr)
            metrics.syntax_ok = True

            for node in ast.walk(tree):
                metrics = add_node(node, metrics)

            print("Syntax ok")
        except SyntaxError as e:
            debug(f"AST parsing failed: {e}")
            metrics.syntax_ok = False

        return metrics


def add_node(
    node: ast.AST,
    metrics: StaticMetrics,
) -> StaticMetrics:
    # Count node types
    node_type = type(node).__name__

    # Count specific constructs
    match node:
        case ast.FunctionDef():
            metrics.n_funcs += 1
        case ast.If() | ast.Match() | ast.Try():
            metrics.n_branch += 1
        case ast.For() | ast.While():
            metrics.n_loop += 1
        case ast.Return():
            metrics.n_returns += 1
        case ast.Import():
            for name in node.names:
                metrics.imports.add(name.name)
        case ast.ImportFrom() if node.module:
            metrics.imports.add(node.module)
        case ast.Call():
            if (
                isinstance(node.func, ast.Name)
                and (fid := node.func.id) in dsl_functions
            ):
                metrics.n_dsl.add(fid)
        case _:
            pass
    return metrics


class LoggingInjector(cst.CSTTransformer):
    """Injects logging code into a function definition."""

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        # Parse decorator from string
        decorator_module = cst.parse_module(
            "@log_sparse(stack_depth=10)\ndef dummy(): pass"
        )
        decorator = decorator_module.body[0].decorators[0]

        # Parse logging statement from string
        logging_code = cst.parse_statement(
            """
tracer.log_locals(list(locals().items()))"""
        )
        #         logging_code = cst.parse_statement(
        #             """
        # for name, value in locals().items():
        #     tracer.log_var(name, value)"""
        #         )

        # Get the function body statements
        body_statements = list(updated_node.body.body)

        # Check for return statements in SimpleStatementLine nodes
        has_return = False
        new_body = []

        for stmt in body_statements:
            if isinstance(stmt, cst.SimpleStatementLine) and any(
                isinstance(expr, cst.Return) for expr in stmt.body
            ):
                has_return = True
                new_body.append(logging_code)
            new_body.append(stmt)

        # If no return statements found, add logging at the end
        if not has_return:
            new_body.append(logging_code)

        # Update function with decorator and new body
        return updated_node.with_changes(
            # decorators=[decorator, *updated_node.decorators],
            body=updated_node.body.with_changes(body=new_body),
        )


def inject_logging(code0: str) -> str:
    # try:
    # Parse and transform the code
    module = cst.parse_module(code0)
    edited_tree = module.visit(LoggingInjector())

    # Verify the transformed code is valid Python
    code1 = edited_tree.code
    compile(code1, "<string>", "exec")
    return code1

    # except Exception as e:
    #     debug(str(e))
    #     return code0


@dataclass
class SolveFunctionValidator(cst.CSTVisitor):
    """A valid solver is of type solve: I -> O, and not redefined."""

    def __init__(self):
        super().__init__()
        self.has_solve_function = False
        self.n_param = 0
        self.n_return = 0
        self.current_function = None
        self.indent_depth = 0
        self.n_solvers = 0

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        self.indent_depth += 1

        if node.name.value == "solve" and self.indent_depth == 1:
            self.has_solve_function = True
            self.n_solvers += 1
            self.current_function = node.name.value
            self.n_param = len(node.params.params)

    def leave_FunctionDef(self, node: cst.FunctionDef) -> None:
        if node.name.value == "solve" and self.indent_depth == 1:
            self.current_function = None
        self.indent_depth -= 1

    def visit_Return(self, node: cst.Return) -> None:
        if self.current_function and self.current_function == "solve":
            self.n_return += 1

    @property
    def is_valid(self) -> bool:
        return (
            self.has_solve_function
            and self.n_param == 1
            and self.n_return == 1
            and self.n_solvers == 1
        )


def validate_solve_function_cst(code_str: str) -> bool:
    """Validates solve function using libcst."""
    try:
        module = cst.parse_module(code_str)
        validator = SolveFunctionValidator()
        module.visit(validator)
        return validator.is_valid
    except Exception as e:
        debug(e)
        return False


if __name__ == "__main__":
    code = """@
def solve_e9afcf9a(I):
    x1 = astuple(TWO, ONE)
    x2 = crop(I, ORIGIN, x1)
    x3 = hmirror(x2)
    x4 = hconcat(x2, x3)
    x5 = hconcat(x4, x4)
    O = hconcat(x5, x4)
    return O
    """.strip()

    debug(inject_logging(code))
