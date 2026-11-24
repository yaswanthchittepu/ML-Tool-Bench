"""A scripted chat model so agents can be tested end-to-end without API calls."""
from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from mltoolbench.llm import ModelConfig

_ids = itertools.count()


def tool_call(name: str, **args: Any) -> AIMessage:
    """An AIMessage requesting one tool call."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{next(_ids)}",
                                              "type": "tool_call"}])


class ScriptedChatModel(BaseChatModel):
    """Replies from ``script``: either a list of messages (consumed in order; text
    replies once exhausted) or a function ``(messages, bound_tool_names) -> AIMessage``."""

    script: Any
    bound: tuple[str, ...] = ()
    calls: list = []

    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **_: Any) -> ScriptedChatModel:
        names = tuple(getattr(t, "name", None) or getattr(t, "__name__", str(t)) for t in tools)
        return self.model_copy(update={"bound": names})

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **_: Any) -> ChatResult:
        self.calls.append(messages)
        if callable(self.script):
            reply = self.script(messages, self.bound)
        else:
            reply = self.script.pop(0) if self.script else AIMessage(content="Done.")
        reply = reply.model_copy(update={"usage_metadata": {"input_tokens": 100, "output_tokens": 10,
                                                            "total_tokens": 110}})
        return ChatResult(generations=[ChatGeneration(message=reply)])


def scripted_config(script: list[AIMessage] | Callable, model: str = "gpt-4.1-mini") -> ModelConfig:
    shared = ScriptedChatModel(script=script)
    return ModelConfig(model=model, factory=lambda _cfg: shared)
