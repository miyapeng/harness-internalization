"""Bounded executable Python predicates; candidate source is parsed, never exec'd.

The initial search space changes trigger branches, guidance and persistence.
The fixed runtime implements planner, draft-review and recovery model calls.
Arbitrary Python programs are deliberately outside this initial implementation.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from ..core.types import State, digest


def _expr(node, history, step):
    if isinstance(node, ast.Constant) and type(node.value) in (str, int, bool):
        return node.value
    if isinstance(node, ast.Name) and node.id in ("history", "step"):
        return history if node.id == "history" else step
    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        values = [bool(_expr(v, history, step)) for v in node.values]
        return all(values) if isinstance(node.op, ast.And) else any(values)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _expr(node.operand, history, step)
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        left, right = _expr(node.left, history, step), _expr(node.comparators[0], history, step)
        op = node.ops[0]
        if isinstance(op, ast.Eq): return left == right
        if isinstance(op, ast.NotEq): return left != right
        if isinstance(op, ast.Lt): return left < right
        if isinstance(op, ast.LtE): return left <= right
        if isinstance(op, ast.Gt): return left > right
        if isinstance(op, ast.GtE): return left >= right
        if isinstance(op, ast.In) and isinstance(right, str): return left in right
        if isinstance(op, ast.NotIn) and isinstance(right, str): return left not in right
    if isinstance(node, ast.Call) and not node.keywords:
        if isinstance(node.func, ast.Name) and node.func.id == "len" and len(node.args) == 1:
            return len(_expr(node.args[0], history, step))
        if isinstance(node.func, ast.Attribute):
            obj = _expr(node.func.value, history, step)
            args = [_expr(a, history, step) for a in node.args]
            if isinstance(obj, str):
                if node.func.attr == "lower" and not args: return obj.lower()
                if node.func.attr == "count" and len(args) == 1: return obj.count(args[0])
    raise ValueError(f"Unsupported predicate syntax: {ast.dump(node)}")


@dataclass(frozen=True)
class HarnessModule:
    name: str
    kind: str
    instruction: str
    persistence: int
    source: str
    predicate: ast.expr

    @property
    def version(self):
        return digest(self.source)

    def triggered(self, state: State):
        return bool(_expr(self.predicate, state.public_history, state.step))

    @classmethod
    def load(cls, path: Path):
        return cls.from_source(path.read_text())

    @classmethod
    def from_source(cls, source: str):
        if len(source) > 16384:
            raise ValueError("Module source exceeds 16 KiB")
        tree = ast.parse(source)
        if len(list(ast.walk(tree))) > 256:
            raise ValueError("Module AST exceeds limit")
        metadata, predicate = {}, None
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                key = node.targets[0].id
                if key not in ("NAME", "KIND", "INSTRUCTION", "PERSISTENCE") or key in metadata:
                    raise ValueError("Unexpected or repeated module metadata")
                metadata[key] = ast.literal_eval(node.value)
            elif isinstance(node, ast.FunctionDef) and node.name == "trigger" and predicate is None:
                if ([a.arg for a in node.args.args] != ["history", "step"] or node.decorator_list
                    or node.args.defaults or node.args.kwonlyargs or node.args.posonlyargs
                    or node.args.vararg or node.args.kwarg or len(node.body) != 1
                    or not isinstance(node.body[0], ast.Return)):
                    raise ValueError("Expected def trigger(history, step): return <bounded expression>")
                predicate = node.body[0].value
            else:
                raise ValueError("Only literal metadata and a trigger function are allowed")
        if set(metadata) != {"NAME", "KIND", "INSTRUCTION", "PERSISTENCE"} or predicate is None:
            raise ValueError("Incomplete module")
        if metadata["KIND"] not in ("planner", "review", "recovery"):
            raise ValueError("Unsupported control module kind")
        if any(not isinstance(metadata[k], str) or not metadata[k].strip() for k in ("NAME", "INSTRUCTION")):
            raise ValueError("Empty module metadata")
        if type(metadata["PERSISTENCE"]) is not int or not 0 <= metadata["PERSISTENCE"] <= 8:
            raise ValueError("Persistence must be 0..8 steps")
        # Interpreter visits all boolean operands; unsupported code cannot hide in dead branches.
        _expr(predicate, "", 0)
        return cls(metadata["NAME"], metadata["KIND"], metadata["INSTRUCTION"],
                   metadata["PERSISTENCE"], source, predicate)


@dataclass(frozen=True)
class Harness:
    modules: tuple[HarnessModule, ...] = ()

    def __post_init__(self):
        if len({m.name for m in self.modules}) != len(self.modules):
            raise ValueError("Duplicate module names")

    @property
    def version(self):
        return digest([m.version for m in self.modules])

    def without(self, name):
        if name not in {m.name for m in self.modules}: raise ValueError("Missing target module")
        return Harness(tuple(m for m in self.modules if m.name != name))

# Compatibility name for the original project API.
ControlModule = HarnessModule
