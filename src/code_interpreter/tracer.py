"""
DSL tracer for ARC-DSL
Provides runtime tracing of DSL functions without modifying their code
"""

from dataclasses import dataclass, field
from enum import Enum
import time
import sys
import types
import functools
import inspect
import numpy as np
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, Set

from src.dsl.util import find_module_funcs


class ExecStatus(str, Enum):
    PENDING = "pending"
    SUCCESS = "success"
    ERROR = "error"


@dataclass
class DSLFunctionCall:
    """Information about a single DSL function call"""

    task_id: int
    fname: str
    args: tuple
    kwargs: dict

    t0: float
    t1: Optional[float] = None
    result: Any = None
    error: Optional[str] = None
    status: ExecStatus = ExecStatus.PENDING

    parent_id: Optional[int] = None
    line_number: Optional[int] = None
    file_name: Optional[str] = None

    @property
    def duration(self) -> float:
        if self.t1:
            return self.t1 - self.t0
        return time.time() - self.t0


@dataclass
class DSLTracer:
    """Traces all function calls in a DSL module"""

    execution_log: List[DSLFunctionCall] = field(default_factory=list)
    call_stack: List[int] = field(default_factory=list)
    hooks: List[Callable] = field(default_factory=list)
    call_counter: int = 0
    enabled: bool = True

    def clear(self) -> None:
        self.execution_log = []
        self.call_counter = 0

    def wrap_function(self, func):
        """Wrap a function to trace its execution"""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self.enabled:
                return func(*args, **kwargs)

            # Get caller information
            frame = inspect.currentframe().f_back

            # Create function call record
            call_id = self.call_counter
            self.call_counter += 1

            parent_id = self.call_stack[-1] if self.call_stack else None

            call_info = DSLFunctionCall(
                task_id=call_id,
                fname=func.__name__,
                args=args,
                kwargs=kwargs,
                t0=time.time(),
                parent_id=parent_id,
                line_number=frame.f_lineno,
                file_name=frame.f_code.co_filename,
            )

            # Push to call stack
            self.call_stack.append(call_id)

            # Execute function
            try:
                # MAIN RUN
                result = func(*args, **kwargs)
                call_info.status = ExecStatus.SUCCESS
                call_info.result = result

                # # Add grid diff information if applicable
                # call_info.add_diff_info(args, result)
                call_info.postprocess()

                return result
            except Exception as e:
                call_info.status = ExecStatus.ERROR
                call_info.error = str(e)
                raise
            finally:
                # Update timing information
                call_info.t1 = time.time()

                # Record function call
                self.execution_log.append(call_info)

                # Pop from call stack
                self.call_stack.pop()

        return wrapper

    def instrument(self, module_name):
        """Instrument all funcs in module.py with a decorator hook that logs IO"""
        debug(f"Instrumenting module: {module_name}")

        instrumented_funcs = {}

        for module, name, obj in find_module_funcs(module_name):
            if module.__name__ == module_name:
                # Direct module functions
                wrapped = self.wrap_function(obj)
                module.__dict__[name] = wrapped
                instrumented_funcs[obj] = wrapped
            elif obj in instrumented_funcs:
                # Star-imported functions
                module.__dict__[name] = instrumented_funcs[obj]
        self.hooks += list(instrumented_funcs.keys())
        print(f"Add instrument hooks: {self.hooks[:4]} ... (total: {len(self.hooks)})")

    def get_function_stats(self):
        """Return statistics about function calls"""
        stats = {}

        debug(self.execution_log)

        for call in self.execution_log:
            name = call.fname
            if name not in stats:
                stats[name] = {
                    "count": 0,
                    "total_time_ms": 0,
                    "avg_time_ms": 0,
                    "error_count": 0,
                    # "grid_changes": 0,
                }

            stats[name]["count"] += 1
            if call.duration:
                stats[name]["total_time_ms"] += call.duration

            if call.status == "error":
                stats[name]["error_count"] += 1

            # if call.grid_changes:
            #     stats[name]["grid_changes"] += call.grid_changes

        # Calculate averages
        for name, data in stats.items():
            if data["count"] > 0:
                data["avg_time_ms"] = data["total_time_ms"] / data["count"]

        return stats

    def get_call_tree(self):
        """Return a nested call tree"""
        # Build a map of parent -> children
        children_map = {}
        for call in self.execution_log:
            parent_id = call.parent_id
            if parent_id is not None:
                if parent_id not in children_map:
                    children_map[parent_id] = []
                children_map[parent_id].append(call.id)

        # Find root calls
        root_calls = [call for call in self.execution_log if call.parent_id is None]

        # Build tree recursively
        def build_tree(call):
            call_id = call.id
            children_ids = children_map.get(call_id, [])
            children = [
                build_tree(self.execution_log[child_id]) for child_id in children_ids
            ]

            # TODO: To dataclass
            return {
                "id": call.id,
                "function": call.function_name,
                "duration": call.duration,
                "status": call.status,
                "children": children,
            }

        return [build_tree(call) for call in root_calls]
