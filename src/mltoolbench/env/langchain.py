"""LangChain / LangGraph adapter for the scratchpad tools.

Each :class:`~mltoolbench.env.tools.ToolSpec` becomes a LangChain tool whose call
returns a LangGraph ``Command`` carrying a ToolMessage and the scratchpad *delta*.
Graph states that keep a single scratchpad should declare it with
:func:`merge_scratchpad` as reducer; tree-search agents store deltas per node.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Annotated, Any, Dict

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from mltoolbench.env.tools import Kind, ToolSpec, execute


def merge_scratchpad(current: dict[str, Any] | None, update: dict[str, Any] | None) -> dict[str, Any]:
    """LangGraph reducer: tool deltas add or overwrite keys, never delete them."""
    return {**(current or {}), **(update or {})}


def to_langchain_tool(spec: ToolSpec) -> BaseTool:
    def run(state: Mapping[str, Any], tool_call_id: str, **call: Any) -> Command:
        calls = state.get("num_tool_calls", 0)
        result = execute(spec, state.get("scratchpad") or {}, call_index=calls,
                         **call)
        return Command(update={
            "num_tool_calls": calls + 1,
            "scratchpad": result.updates,
            "messages": [ToolMessage(result.message, tool_call_id=tool_call_id)],
        })

    wrapper = _SIGNATURES[spec.kind](run)
    wrapper.__name__ = spec.name
    wrapper.__doc__ = spec.description
    return tool(wrapper)


def langchain_tools(specs: list[ToolSpec]) -> list[BaseTool]:
    return [to_langchain_tool(s) for s in specs]


# One explicit signature per kind: LangChain derives the JSON schema the LLM sees from it.
State = Annotated[dict, InjectedState]
CallId = Annotated[str, InjectedToolCallId]


def _get_set(run: Callable) -> Callable:
    def wrapper(state: State, tool_call_id: CallId, bindings: Dict[str, str], output: str,
                func_kwargs: Dict[str, Any] = {}):  # noqa: B006 - schema default, never mutated
        return run(state, tool_call_id, bindings=bindings, output=output, func_kwargs=func_kwargs)
    return wrapper


def _get(run: Callable) -> Callable:
    def wrapper(state: State, tool_call_id: CallId, bindings: Dict[str, str],
                func_kwargs: Dict[str, Any] = {}):  # noqa: B006
        return run(state, tool_call_id, bindings=bindings, func_kwargs=func_kwargs)
    return wrapper


def _set(run: Callable) -> Callable:
    def wrapper(state: State, tool_call_id: CallId, output: str,
                func_kwargs: Dict[str, Any] = {}):  # noqa: B006
        return run(state, tool_call_id, output=output, func_kwargs=func_kwargs)
    return wrapper


def _override(run: Callable) -> Callable:
    def wrapper(state: State, tool_call_id: CallId, bindings: Dict[str, str],
                func_kwargs: Dict[str, Any] = {}):  # noqa: B006
        return run(state, tool_call_id, bindings=bindings, func_kwargs=func_kwargs)
    return wrapper


_SIGNATURES = {Kind.GET_SET: _get_set, Kind.GET: _get, Kind.SET: _set, Kind.OVERRIDE: _override}
