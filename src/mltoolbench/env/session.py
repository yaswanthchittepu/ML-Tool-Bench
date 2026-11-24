"""Framework-free access to the tools, for evaluating your own agent.

    session = ToolSession()                      # or ToolSession(stage="modeling")
    tools = session.openai_tools()               # pass to any function-calling API
    text = session.call("read_data", {"output": "train", "func_kwargs": {"filepath": ...}})

The session owns the scratchpad; tool errors come back as text, as the baselines see them.
"""
from __future__ import annotations

import json
from typing import Any

from mltoolbench.env.tools import Kind, ToolSpec, execute, tools_for_stage

_PARAMS = {
    "bindings": {"type": "object", "additionalProperties": {"type": "string"},
                 "description": "Function parameter name -> scratchpad key to read it from."},
    "output": {"type": "string", "description": "Scratchpad key to store the result under."},
    "func_kwargs": {"type": "object", "default": {},
                    "description": "Literal keyword arguments for the function."},
}
_REQUIRED = {Kind.GET_SET: ["bindings", "output"], Kind.GET: ["bindings"], Kind.SET: ["output"],
             Kind.OVERRIDE: ["bindings"]}


class ToolSession:
    def __init__(self, stage: str | None = None):
        self.specs: dict[str, ToolSpec] = {s.name: s for s in tools_for_stage(stage)}
        self.scratchpad: dict[str, Any] = {}
        self.num_tool_calls = 0

    def openai_tools(self) -> list[dict[str, Any]]:
        """Tool definitions in OpenAI function-calling format."""
        return [{"type": "function", "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": {"type": "object", "required": _REQUIRED[spec.kind],
                           "properties": {p: _PARAMS[p] for p in ("bindings", "output", "func_kwargs")
                                          if p in _REQUIRED[spec.kind] or p == "func_kwargs"}},
        }} for spec in self.specs.values()]

    def call(self, name: str, arguments: dict[str, Any] | str) -> str:
        """Run one tool call and return the message for the agent."""
        if name not in self.specs:
            return f"Error: unknown tool {name!r}. Available: {', '.join(self.specs)}"
        args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
        allowed = {"bindings", "output", "func_kwargs"}
        try:
            result = execute(self.specs[name], self.scratchpad, call_index=self.num_tool_calls,
                             **{k: v for k, v in args.items() if k in allowed})
        except Exception as e:  # surfaced to the agent, like LangGraph's ToolNode does
            return f"Error: {e!r}\n Please fix your mistakes."
        self.scratchpad.update(result.updates)
        self.num_tool_calls += 1
        return result.message
