"""Expansion: ask the LLM for tool calls, run each against a node's scratchpad, and
turn every tool call into one candidate child."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from mltoolbench.env import ToolSpec, all_tools
from mltoolbench.env.langchain import langchain_tools
from mltoolbench.llm import bind_tools, normalize_tool_calls

#: MCTS agents never get this tool: the framework writes the chosen node's submission.
FRAMEWORK_WRITES_SUBMISSION = "save_dataframe_to_csv"


@dataclass
class Candidate:
    """One proposed step: ``[AIMessage(one tool call), ToolMessage]`` plus the scratchpad delta."""
    messages: list[BaseMessage]
    delta: dict[str, Any] = field(default_factory=dict)
    tool_doc: str = ""

    @property
    def tool_name(self) -> str | None:
        calls = getattr(self.messages[0], "tool_calls", None)
        return calls[0]["name"] if calls else None


class Expander:
    def __init__(self, ctx, system_prompt: str, *, bound: Sequence[ToolSpec], executable: Sequence[ToolSpec],
                 parallel_calls: bool, reduce_context_on_error: bool, excluded: Sequence[str] = ()):
        self.question = ctx.prompt()
        self.reduce_context_on_error = reduce_context_on_error
        model = bind_tools(ctx.chat_model(), langchain_tools(list(bound)), ctx.llm,
                           sequential=not parallel_calls)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt), ("user", "{input}"), MessagesPlaceholder(variable_name="messages", optional=True),
        ])
        self.chain = prompt | model
        self.tool_node = ToolNode(langchain_tools(list(executable)))
        # Feedback quotes the description of whatever tool was called, masked or not.
        self.docs = {name: s.description for name, s in all_tools().items() if name not in excluded}

    # ---------------------------------------------------------------- LLM
    def propose(self, messages: list[BaseMessage]) -> AIMessage:
        sizes: list[int | None] = context_sizes(len(messages)) if self.reduce_context_on_error else [None]
        for attempt, size in enumerate(sizes):
            context = messages if size is None or len(messages) <= size else messages[-size:]
            try:
                return normalize_tool_calls(self.chain.invoke({"input": self.question, "messages": context}))
            except Exception:
                if attempt == len(sizes) - 1:
                    raise
        raise AssertionError("unreachable")

    # ---------------------------------------------------------------- tools
    def candidates(self, ai: AIMessage, scratchpad: dict[str, Any]) -> list[Candidate]:
        if not ai.tool_calls:
            return [Candidate([ai])]
        return [self.run_call(ai, call, scratchpad) for call in ai.tool_calls]

    def run_call(self, ai: AIMessage, call: dict, scratchpad: dict[str, Any]) -> Candidate:
        step = AIMessage(content=ai.content, tool_calls=[call], additional_kwargs={"tool_calls": [call]},
                         response_metadata=ai.response_metadata)
        request = AIMessage(content="", tool_calls=[{"name": call["name"], "args": call["args"], "id": call["id"]}])
        out = self.tool_node.invoke({"messages": [request], "scratchpad": scratchpad})
        messages, delta = [], {}
        for item in out if isinstance(out, list) else [out]:
            if isinstance(item, Command):
                delta.update(item.update.get("scratchpad") or {})
                messages.extend(item.update.get("messages", []))
            elif isinstance(item, dict):
                messages.extend(item.get("messages", []))
        return Candidate([step, *messages], delta, self.docs.get(call["name"], ""))


def context_sizes(n: int) -> list[int | None]:
    """Full context first, then the last 10k, ..., 10, 5 and 1 messages ."""
    if n <= 10:
        return [None, 5, 1]
    start = (n // 10) * 10
    if start == n:
        start = n - 10
    return [None, *range(start, 9, -10), 5, 1]


def mcts_tools(stage: str | None = None) -> list[ToolSpec]:
    from mltoolbench.env import tools_for_stage
    return [s for s in tools_for_stage(stage) if s.name != FRAMEWORK_WRITES_SUBMISSION]
