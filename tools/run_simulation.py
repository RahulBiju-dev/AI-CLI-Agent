"""Run bounded, reproducible discrete-time and Monte Carlo simulations."""

from __future__ import annotations

import ast
import json
import math
import random
import statistics
from typing import Any, Callable


_BINARY = {
    ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b, ast.Mod: lambda a, b: a % b,
}
_UNARY = {ast.UAdd: lambda a: a, ast.USub: lambda a: -a, ast.Not: lambda a: not a}
_COMPARE = {
    ast.Lt: lambda a, b: a < b, ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b, ast.GtE: lambda a, b: a >= b,
    ast.Eq: lambda a, b: a == b, ast.NotEq: lambda a, b: a != b,
}

MAX_VARIABLES = 100
MAX_SCENARIOS = 20
MAX_EXPRESSION_CHARS = 2_000
MAX_POWER_EXPONENT = 100.0


def _safe_power(base: float, exponent: float) -> float:
    """Bound exponentiation before Python allocates an enormous integer."""
    try:
        exponent_value = float(exponent)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Power exponent must be a finite number") from exc
    if not math.isfinite(exponent_value) or abs(exponent_value) > MAX_POWER_EXPONENT:
        raise ValueError(f"Power exponent must be between {-MAX_POWER_EXPONENT:g} and {MAX_POWER_EXPONENT:g}")
    try:
        value = base ** exponent
    except (OverflowError, ZeroDivisionError) as exc:
        raise ValueError(f"Power operation failed: {exc}") from exc
    if isinstance(value, complex) or not math.isfinite(float(value)):
        raise ValueError("Power operation produced a non-finite number")
    return value


_BINARY[ast.Pow] = _safe_power


class _Expression:
    def __init__(self, functions: dict[str, Callable[..., float]]):
        self.functions = functions

    def evaluate(self, expression: str, variables: dict[str, float]) -> float:
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError("Expression must be a non-empty string")
        if len(expression) > MAX_EXPRESSION_CHARS:
            raise ValueError(f"Expression exceeds the {MAX_EXPRESSION_CHARS}-character limit")
        tree = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 100:
            raise ValueError("Expression is too complex")
        value = self._node(tree.body, variables)
        if not isinstance(value, (int, float, bool)) or not math.isfinite(float(value)):
            raise ValueError("Expression produced a non-finite number")
        return float(value)

    def _node(self, node: ast.AST, variables: dict[str, float]) -> Any:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise ValueError(f"Unknown variable: {node.id}")
            return variables[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
            return _BINARY[type(node.op)](self._node(node.left, variables), self._node(node.right, variables))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
            return _UNARY[type(node.op)](self._node(node.operand, variables))
        if isinstance(node, ast.BoolOp):
            values = [bool(self._node(value, variables)) for value in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.Compare):
            left = self._node(node.left, variables)
            for operator, comparator in zip(node.ops, node.comparators):
                right = self._node(comparator, variables)
                if type(operator) not in _COMPARE or not _COMPARE[type(operator)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            return self._node(node.body if self._node(node.test, variables) else node.orelse, variables)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in self.functions:
            return self.functions[node.func.id](*[self._node(arg, variables) for arg in node.args])
        raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def run_simulation(
    variables: dict[str, float],
    equations: dict[str, str],
    steps: int = 10,
    dt: float = 1.0,
    mode: str = "recurrence",
    scenarios: list[dict] | None = None,
    trials: int = 1,
    seed: int | None = None,
) -> str:
    """Simulate state equations; equations are safely parsed, never passed to eval()."""
    if not isinstance(variables, dict) or not isinstance(equations, dict) or not variables or not equations:
        return json.dumps({"error": "variables and equations must be non-empty objects"})
    if len(variables) > MAX_VARIABLES or len(equations) > MAX_VARIABLES:
        return json.dumps({"error": f"At most {MAX_VARIABLES} variables/equations are allowed"})
    invalid_names = sorted(
        str(name) for name in set(variables) | set(equations)
        if not isinstance(name, str) or not name.isidentifier()
    )
    reserved_names = sorted(set(variables) & {"step", "time", "dt", "pi", "e"})
    if invalid_names:
        return json.dumps({"error": "Variable and equation names must be valid identifiers", "invalid": invalid_names})
    if reserved_names:
        return json.dumps({"error": "Variable names conflict with reserved simulation values", "reserved": reserved_names})
    if set(equations) - set(variables):
        return json.dumps({"error": "Every equation target must be declared in variables", "unknown": sorted(set(equations) - set(variables))})
    try:
        step_count = max(1, min(int(steps), 10000))
        trial_count = max(1, min(int(trials), 1000))
        dt_value = float(dt)
    except (TypeError, ValueError, OverflowError):
        return json.dumps({"error": "steps and trials must be integers, and dt must be numeric"})
    if not math.isfinite(dt_value) or dt_value <= 0 or dt_value > 1_000_000:
        return json.dumps({"error": "dt must be finite and between 0 (exclusive) and 1,000,000"})
    try:
        initial_state = {key: float(value) for key, value in variables.items()}
    except (TypeError, ValueError, OverflowError):
        return json.dumps({"error": "Every initial variable value must be numeric"})
    if not all(math.isfinite(value) for value in initial_state.values()):
        return json.dumps({"error": "Every initial variable value must be finite"})
    invalid_expressions = [key for key, value in equations.items() if not isinstance(value, str) or not value.strip()]
    if invalid_expressions:
        return json.dumps({"error": "Every equation must be a non-empty string", "invalid": invalid_expressions})
    mode = str(mode or "").strip().lower()
    if mode not in {"recurrence", "euler"}:
        return json.dumps({"error": "mode must be 'recurrence' or 'euler'"})

    if scenarios is not None and not isinstance(scenarios, list):
        return json.dumps({"error": "scenarios must be an array"})
    scenario_defs = scenarios or [{"name": "baseline", "overrides": {}}]
    if len(scenario_defs) > MAX_SCENARIOS:
        return json.dumps({"error": f"At most {MAX_SCENARIOS} scenarios are allowed"})
    if step_count * trial_count * len(equations) * len(scenario_defs) > 500000:
        return json.dumps({"error": "Simulation exceeds the 500,000 evaluation limit across all scenarios"})
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        return json.dumps({"error": "seed must be an integer or null"})
    try:
        randomizer = random.Random(seed)
    except (TypeError, ValueError):
        return json.dumps({"error": "seed must be an integer or null"})
    evaluator = _Expression({
        "min": min, "max": max, "abs": abs, "sqrt": math.sqrt, "log": math.log,
        "exp": math.exp, "sin": math.sin, "cos": math.cos, "floor": math.floor,
        "ceil": math.ceil, "normal": randomizer.gauss, "uniform": randomizer.uniform,
    })

    results = []
    for scenario_index, scenario in enumerate(scenario_defs):
        if not isinstance(scenario, dict):
            return json.dumps({"error": f"scenarios[{scenario_index}] must be an object"})
        name = str(scenario.get("name") or f"scenario-{scenario_index + 1}")
        if len(name) > 200 or any(ord(char) < 32 for char in name):
            return json.dumps({"error": f"scenarios[{scenario_index}].name is invalid or exceeds 200 characters"})
        overrides = scenario.get("overrides", {})
        if not isinstance(overrides, dict):
            return json.dumps({"error": f"Scenario '{name}' overrides must be an object"})
        unknown_overrides = sorted(str(value) for value in set(overrides) - set(variables))
        if unknown_overrides:
            return json.dumps({"error": f"Scenario '{name}' overrides unknown variables", "unknown": unknown_overrides})
        try:
            scenario_state = {**initial_state, **{key: float(value) for key, value in overrides.items()}}
        except (TypeError, ValueError, OverflowError):
            return json.dumps({"error": f"Scenario '{name}' override values must be numeric"})
        if not all(math.isfinite(value) for value in scenario_state.values()):
            return json.dumps({"error": f"Scenario '{name}' override values must be finite"})
        trial_finals: dict[str, list[float]] = {key: [] for key in variables}
        sample_series = []
        for trial in range(trial_count):
            state = dict(scenario_state)
            series = [{"step": 0, "time": 0.0, **state}]
            for step in range(1, step_count + 1):
                context = {**state, "step": float(step), "time": step * dt_value, "dt": dt_value, "pi": math.pi, "e": math.e}
                try:
                    calculated = {key: evaluator.evaluate(expression, context) for key, expression in equations.items()}
                except (SyntaxError, TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
                    return json.dumps({"error": f"Invalid simulation expression at scenario '{name}', trial {trial + 1}, step {step}: {exc}"})
                if mode == "euler":
                    state = {**state, **{key: state[key] + dt_value * value for key, value in calculated.items()}}
                else:
                    state = {**state, **calculated}
                if not all(math.isfinite(value) for value in state.values()):
                    return json.dumps({"error": f"Simulation produced a non-finite state at scenario '{name}', trial {trial + 1}, step {step}"})
                if trial == 0 and (step_count <= 200 or step in {step_count} or step % max(1, step_count // 100) == 0):
                    series.append({"step": step, "time": step * dt_value, **state})
            if trial == 0:
                sample_series = series
            for key, value in state.items():
                trial_finals[key].append(value)

        summary = {
            key: {
                "mean": statistics.fmean(values),
                "min": min(values), "max": max(values),
                "p05": _percentile(values, 0.05), "p50": _percentile(values, 0.5), "p95": _percentile(values, 0.95),
            }
            for key, values in trial_finals.items()
        }
        results.append({"name": name, "overrides": overrides, "sample_trajectory": sample_series, "final_distribution": summary})

    return json.dumps({
        "mode": mode, "steps": step_count, "dt": dt_value, "trials": trial_count, "seed": seed,
        "scenarios": results,
        "interpretation_note": "Outputs are conditional on the supplied equations and assumptions; they are not independently validated forecasts.",
    }, ensure_ascii=False)
