"""Chat-model construction, token/cost accounting and tool-call normalisation.

Any OpenAI-compatible endpoint works (OpenAI, Azure, vLLM, LiteLLM proxies, ...):
configure it with ``ModelConfig`` or the ``OPENAI_MODEL`` / ``OPENAI_BASE_URL`` /
``OPENAI_API_KEY`` / ``TEMPERATURE`` environment variables.
"""
from __future__ import annotations

import ast
import json
import os
import secrets
import string
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import LLMResult

# USD per 1M tokens: (input, cached input, output). Used only for cost reports.
# Override or extend with ModelConfig(prices=...). Unknown models report cost None.
PRICES: dict[str, tuple[float, float, float]] = {
    "gpt-4o": (2.50, 1.25, 10.00),
    "gpt-4o-mini": (0.15, 0.075, 0.60),
    "gpt-4.1": (2.00, 0.50, 8.00),
    "gpt-4.1-mini": (0.40, 0.10, 1.60),
    "gpt-4.1-nano": (0.10, 0.025, 0.40),
}


@dataclass
class ModelConfig:
    model: str
    base_url: str | None = None
    api_key: str | None = None
    temperature: float | None = None
    prices: tuple[float, float, float] | None = None
    #: Escape hatch for tests or non-OpenAI providers: builds the chat model itself.
    factory: Callable[[ModelConfig], BaseChatModel] | None = field(default=None, repr=False)

    @classmethod
    def from_env(cls, **overrides: Any) -> ModelConfig:
        """Environment variables, with explicit ``overrides`` (e.g. ``--model``) taking precedence."""
        temperature = os.environ.get("TEMPERATURE")
        values = {
            "model": os.environ.get("OPENAI_MODEL"),
            "base_url": os.environ.get("OPENAI_BASE_URL"),
            "api_key": os.environ.get("OPENAI_API_KEY"),
            "temperature": float(temperature) if temperature else None,
            **overrides,
        }
        if not values["model"]:
            raise ValueError("No model configured: pass --model or set OPENAI_MODEL")
        return cls(**values)

    @property
    def supports_parallel_tool_calls_flag(self) -> bool:
        """o-series and Claude endpoints reject ``parallel_tool_calls``."""
        return not any(tag in self.model for tag in ("o1", "o3", "o4", "claude"))

    def price(self) -> tuple[float, float, float] | None:
        if self.prices:
            return self.prices
        base = self.model.split("/")[-1]
        return PRICES.get(base) or next((p for name, p in sorted(PRICES.items(), key=lambda kv: -len(kv[0]))
                                         if base.startswith(name)), None)


def make_chat_model(config: ModelConfig, callbacks: list[BaseCallbackHandler] | None = None) -> BaseChatModel:
    if config.factory is not None:
        model = config.factory(config)
        if callbacks:  # a model field (not with_config) so it survives bind_tools()
            model = model.model_copy(update={"callbacks": callbacks})
        return model
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {"model": config.model, "callbacks": callbacks}
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.temperature is not None:
        kwargs["temperature"] = config.temperature
    return ChatOpenAI(**kwargs)


def bind_tools(model: BaseChatModel, tools: list, config: ModelConfig, *, sequential: bool = True, **kwargs: Any):
    """Bind tools, forcing one tool call per turn when ``sequential`` and the endpoint allows it."""
    if sequential and config.supports_parallel_tool_calls_flag:
        kwargs.setdefault("parallel_tool_calls", False)
    return model.bind_tools(tools, **kwargs)


# --------------------------------------------------------------------------- usage

@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    cost: float | None = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {"calls": self.calls, "input_tokens": self.input_tokens,
                "cached_input_tokens": self.cached_input_tokens, "output_tokens": self.output_tokens,
                "total_tokens": self.total_tokens, "cost_usd": self.cost}


class UsageTracker(BaseCallbackHandler):
    """Accumulates token usage across every LLM call of a run, plus the most expensive call."""

    def __init__(self, prices: tuple[float, float, float] | None):
        self.prices = prices
        self.total = Usage(cost=0.0 if prices else None)
        self.max_call = Usage(cost=0.0 if prices else None)
        self._lock = threading.Lock()

    def on_llm_end(self, response: LLMResult, **_: Any) -> None:
        for generations in response.generations:
            for gen in generations:
                meta = getattr(getattr(gen, "message", None), "usage_metadata", None)
                if meta:
                    self._record(meta)

    def _record(self, meta: dict) -> None:
        cached = (meta.get("input_token_details") or {}).get("cache_read", 0) or 0
        call = Usage(1, meta.get("input_tokens", 0), cached, meta.get("output_tokens", 0), None)
        if self.prices:
            p_in, p_cached, p_out = self.prices
            call.cost = ((call.input_tokens - cached) * p_in + cached * p_cached + call.output_tokens * p_out) / 1e6
        with self._lock:
            t = self.total
            t.calls += 1
            t.input_tokens += call.input_tokens
            t.cached_input_tokens += cached
            t.output_tokens += call.output_tokens
            if t.cost is not None and call.cost is not None:
                t.cost += call.cost
            if call.total_tokens > self.max_call.total_tokens:
                self.max_call = call


# --------------------------------------------------------------------------- tool calls

def normalize_tool_calls(message: AIMessage) -> AIMessage:
    """Recover tool calls that open models emit as JSON text (``{"name": ..., "parameters": ...}``).

    Returns the message unchanged when it already has structured tool calls or the
    content is not such a JSON object. Otherwise returns a copy with ``tool_calls`` set.
    """
    if message.tool_calls or not isinstance(message.content, str):
        return message
    parsed = parse_json_tool_call(message.content)
    if parsed is None:
        return message
    name, args = parsed
    call_id = "call_" + "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(24))
    return message.model_copy(update={
        "content": "",
        "tool_calls": [{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
        "additional_kwargs": {**message.additional_kwargs, "tool_calls": [{
            "id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args, default=str)},
        }]},
    })


def parse_json_tool_call(content: str) -> tuple[str, dict] | None:
    text = content.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not (isinstance(parsed, dict) and "name" in parsed and isinstance(parsed.get("parameters"), dict)):
        return None
    args = {}
    for key, value in parsed["parameters"].items():
        # Open models often stringify dict/list arguments: "{'filepath': 'x.csv'}"
        if isinstance(value, str) and value[:1] + value[-1:] in ("{}", "[]", "()"):
            try:
                value = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                pass
        args[key] = value
    return parsed["name"], args
