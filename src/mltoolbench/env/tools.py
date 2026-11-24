"""Scratchpad-aware tools: the core of the ML-Tool-Bench environment.

Agents never pass dataframes or models around directly. Every object lives in a
*scratchpad* (a named-object store), and tools refer to objects by key:

* ``bindings`` maps a function parameter to a scratchpad key to read from,
* ``output`` names the key the result is stored under,
* ``func_kwargs`` carries plain literal arguments.

How a tool reads and writes is its ``kind`` (see ``catalog.yaml``). This module is
framework-agnostic: :func:`execute` takes a scratchpad and returns a
:class:`ToolResult` (an LLM-facing message plus a scratchpad *delta*). Adapters for
specific agent frameworks (e.g. :mod:`mltoolbench.env.langchain`) build on it.
"""
from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import Any

import yaml

CATALOG_PATH = Path(__file__).resolve().parent.parent / "tools" / "catalog.yaml"


class Kind(StrEnum):
    SET = "set"
    GET = "get"
    GET_SET = "get_set"
    OVERRIDE = "override"

    @property
    def reads(self) -> bool:
        return self is not Kind.SET

    @property
    def takes_output(self) -> bool:
        return self in (Kind.SET, Kind.GET_SET)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    func: Callable[..., Any]
    kind: Kind
    #: Fixed scratchpad keys for tuple-returning tools; overrides ``output``.
    outputs: tuple[str, ...] = ()

    @property
    def description(self) -> str:
        """The description shown to the LLM: framework usage + the function's docstring."""
        return (_DESCRIPTIONS[self.kind].strip() + "\n\n" + _DOC_PREFIX[self.kind] + (self.func.__doc__ or "")).strip()


@dataclass
class ToolResult:
    message: str
    #: Scratchpad entries to add or overwrite. Agents merge this into their scratchpad.
    updates: dict[str, Any] = field(default_factory=dict)


class ToolUsageError(ValueError):
    """The agent called a tool with arguments that violate its scratchpad contract."""


def execute(
    spec: ToolSpec,
    scratchpad: Mapping[str, Any],
    *,
    bindings: Mapping[str, str] | None = None,
    output: str | None = None,
    func_kwargs: Mapping[str, Any] | None = None,
    call_index: int = 0,
) -> ToolResult:
    """Run one tool call against ``scratchpad`` without mutating it.

    ``call_index`` is the number of tool calls made so far; it is echoed in the message.

    Exceptions raised by the underlying function propagate; adapters turn them into
    error messages for the agent.
    """
    bindings = dict(bindings or {})
    if spec.kind is Kind.OVERRIDE and len(bindings) != 1:
        raise ToolUsageError(f"{spec.name} overwrites its input, so it takes exactly one binding; got {bindings}")

    args = {param: scratchpad.get(key) for param, key in bindings.items()} if spec.kind.reads else {}
    args.update(func_kwargs or {})
    result = spec.func(**args)

    head = f"Applied {spec.name} with docstring: {spec.func.__doc__}, annotations: {spec.func.__annotations__}"
    mapping = f" The mapping between the function parameters and the scratchpad keys accessed is {bindings}."
    tail = f" Number of Tools Called: {call_index}"

    if spec.kind is Kind.GET:
        return ToolResult(f"{head}, and result is {result}.{mapping}{tail}")

    if isinstance(result, tuple) and len(result) == 1:
        result = result[0]
    if isinstance(result, tuple):
        values = _tuple_outputs(spec, result)
        keys = tuple(values)
        extra = ""
        if spec.name.startswith("tune_"):
            extra = (f" The Best params and CV score for this method are {values['best_params']}"
                     f" and {values['best_cv_score']} respectively.")
        return ToolResult(
            f"{head}, and stored results in scratchpad as {_quote_list(keys)}.{extra}{mapping}{tail}", values
        )

    key = next(iter(bindings.values())) if spec.kind is Kind.OVERRIDE else output
    if not key:
        raise ToolUsageError(f"{spec.name} needs an `output` scratchpad key to store its result")
    if spec.kind is Kind.SET:
        mapping = ""
    return ToolResult(f"{head}, and stored result in scratchpad as '{key}'.{mapping}{tail}", {key: result})


def _tuple_outputs(spec: ToolSpec, result: tuple) -> dict[str, Any]:
    """Map a tuple result onto the tool's fixed scratchpad keys."""
    if spec.name == "convert_dataframe_to_features_target" and result[-1] is None:
        return {"X_test": result[0]}  # called on test data: there is no target
    if len(spec.outputs) != len(result):
        raise ToolUsageError(f"{spec.name} returned {len(result)} values but declares outputs {spec.outputs}")
    return dict(zip(spec.outputs, result, strict=True))


def _quote_list(keys: tuple[str, ...]) -> str:
    quoted = [f"'{k}'" for k in keys]
    if len(quoted) == 1:
        return quoted[0]
    if len(quoted) == 2:
        return f"{quoted[0]} and {quoted[1]}"
    return ", ".join(quoted[:-1]) + f", and {quoted[-1]}"


# --------------------------------------------------------------------------- catalog

@cache
def load_catalog(path: Path = CATALOG_PATH) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


@cache
def all_tools() -> dict[str, ToolSpec]:
    """Every tool in the benchmark, keyed by name, in catalog order."""
    specs = {}
    for name, entry in load_catalog()["tools"].items():
        module = importlib.import_module(f"mltoolbench.tools.{entry['module']}")
        specs[name] = ToolSpec(name, getattr(module, name), Kind(entry["kind"]), tuple(entry.get("outputs", ())))
    return specs


def stage_names() -> list[str]:
    return list(load_catalog()["stages"])


def tools_for_stage(stage: str | None = None) -> list[ToolSpec]:
    """All tools, or only those unmasked in ``stage`` (hierarchical agents)."""
    tools = all_tools()
    if stage is None:
        return list(tools.values())
    return [tools[name] for name in load_catalog()["stages"][stage]["tools"]]


# --------------------------------------------------------------------------- descriptions
# LLMs see these as part of each tool description.

_FUNC_KWARGS_DOC = """
        func_kwargs : dict
            keyword arguments to pass to the function. This is a dictionary of the arguments that the function expects.
            {
                'arg1': ...,
                'arg2': ...,
                ...
            }
            Default is an empty dictionary."""

_BINDINGS_DOC = """
        bindings : dict
            A mapping of parameter names (e.g. "df1") to scratchpad keys (e.g. "users").
"""

_OUTPUT_DOC = """
        output : str
            The name of the scratchpad key to store the result under.
"""

_DESCRIPTIONS = {
    Kind.GET_SET: """
        This tool reads arguments from the scratchpad using `bindings`, passes them to the internal function,
        and stores the result in the scratchpad under the given `output` key.

        Additional parameters injected by the framework:

        Parameters
        ----------""" + _BINDINGS_DOC + _OUTPUT_DOC + _FUNC_KWARGS_DOC + """
        Returns
        -------
        Command
            A LangGraph Command object that updates the agent's state:
            - Stores the result in `scratchpad[output]`
            - Sends a ToolMessage acknowledging the update
        """,
    Kind.GET: """
        This tool reads arguments from the scratchpad using `bindings`, passes them to the internal function.

        Additional parameters injected by the framework:

        Parameters
        ----------""" + _BINDINGS_DOC + _FUNC_KWARGS_DOC + """

        Returns
        -------
        Command
            A LangGraph Command object that updates the agent's state:
            - Sends a ToolMessage acknowledging the successful tool call
        """,
    Kind.SET: """
        This tool passes the func_kwargs to the internal function and stores the result in the scratchpad under the given `output` key.

        Additional parameters injected by the framework:

        Parameters
        ----------
""" + _OUTPUT_DOC + _FUNC_KWARGS_DOC + """
        Returns
        -------
        Command
            A LangGraph Command object that updates the agent's state:
            - Stores the result in `scratchpad[output]`
            - Sends a ToolMessage acknowledging the update
        """,
    Kind.OVERRIDE: """
        This tool reads arguments from the scratchpad using `bindings`, passes them to the internal function,
        and stores the result in the scratchpad under the given `output` key.

        Additional parameters injected by the framework:

        Parameters
        ----------""" + _BINDINGS_DOC + _FUNC_KWARGS_DOC + """
        Returns
        -------
        Command
            A LangGraph Command object that updates the agent's state:
            - Sends a ToolMessage acknowledging the update
        """,
}

_READ_PREFIX = ("The internal function to which the objects in the scratchpad listed in `bindings`, "
                "and func_kwargs are passed has the following parameters it expects:")
_DOC_PREFIX = {
    Kind.GET_SET: _READ_PREFIX,
    Kind.GET: _READ_PREFIX,
    Kind.OVERRIDE: _READ_PREFIX,
    Kind.SET: "The internal function to which func_kwargs are passed has the following parameters it expects:",
}
