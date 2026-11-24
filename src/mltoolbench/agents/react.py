"""ReAct: the paper's baseline. Plan once, then call one tool per turn until the
model stops calling tools."""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from mltoolbench.agents.base import Agent, RunContext
from mltoolbench.env import tools_for_stage
from mltoolbench.env.langchain import langchain_tools, merge_scratchpad
from mltoolbench.llm import bind_tools, normalize_tool_calls

SYSTEM_PROMPT = "You are a Data Scientist tasked with solving the Kaggle competition provided below. "
PLAN_PROMPT = """
            You are a Data Scientist tasked with solving the Kaggle competition provided below. Create a high level plan of what steps you will take to solve the problem.
            """
ONE_TOOL = "Only call one tool at a time. Multiple parallel tool calls are not allowed."


class ReActState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    scratchpad: Annotated[dict[str, Any], merge_scratchpad]
    num_tool_calls: int
    plan: str | None


class ReActAgent(Agent):
    name = "react"

    def __init__(self, plan: bool | None = None, recursion_limit: int = 250):
        """``plan=None`` follows the paper: plan first, except for Llama-family models."""
        self.plan = plan
        self.recursion_limit = recursion_limit

    def run(self, ctx: RunContext) -> dict[str, Any]:
        tools = langchain_tools(tools_for_stage(None))
        model = bind_tools(ctx.chat_model(), tools, ctx.llm)
        task = HumanMessage(ctx.prompt())
        use_plan = self.plan if self.plan is not None else "llama" not in ctx.llm.model.lower()

        def plan_node(state: ReActState) -> dict:
            response = model.invoke([SystemMessage(PLAN_PROMPT), task])
            ctx.log("plan", content=response.content)
            return {"plan": response.content}

        def agent_node(state: ReActState) -> dict:
            _log_tool_results(ctx, state["messages"])
            if use_plan:
                guide = (f"\n\nThe plan proposed by an agent to solve the problem is: {state['plan']}. "
                         f"Use this plan to guide your choice of tools to call. {ONE_TOOL}\n\n")
            else:
                guide = f"\n\n{ONE_TOOL}\n\n"
            response = normalize_tool_calls(model.invoke([SystemMessage(SYSTEM_PROMPT), task, HumanMessage(guide),
                                                          *state["messages"]]))
            _log_ai(ctx, response)
            return {"messages": [response]}

        graph = StateGraph(ReActState)
        graph.add_node("agent", agent_node)
        graph.add_node("tools", ToolNode(tools))
        if use_plan:
            graph.add_node("plan", plan_node)
            graph.set_entry_point("plan")
            graph.add_edge("plan", "agent")
        else:
            graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", _has_tool_calls, {True: "tools", False: END})
        graph.add_edge("tools", "agent")
        return _invoke(ctx, graph, {"messages": [], "scratchpad": {}, "num_tool_calls": 0}, self.recursion_limit)


# --------------------------------------------------------------------------- shared

def _has_tool_calls(state: ReActState) -> bool:
    last = state["messages"][-1]
    return isinstance(last, AIMessage) and bool(last.tool_calls)


def _invoke(ctx: RunContext, graph: StateGraph, init: dict, recursion_limit: int) -> dict[str, Any]:
    stopped = None
    final: dict = init
    try:  # stream so that the last state survives a recursion-limit stop
        for state in graph.compile().stream(init, {"recursion_limit": recursion_limit}, stream_mode="values"):
            final = state
    except GraphRecursionError:
        stopped = f"recursion limit {recursion_limit} reached"
        ctx.log("stopped", reason=stopped)
    return {"num_tool_calls": final.get("num_tool_calls", 0), "stopped": stopped,
            "scratchpad_keys": sorted(final.get("scratchpad", {}))}


def _log_ai(ctx: RunContext, message: AIMessage) -> None:
    calls = [{"tool": c["name"], "args": c["args"]} for c in message.tool_calls]
    ctx.log("llm", content=message.content, tool_calls=calls)


def _log_tool_results(ctx: RunContext, messages: list[BaseMessage]) -> None:
    results = []
    for msg in reversed(messages):  # the tool results since the last AI message
        if not isinstance(msg, ToolMessage):
            break
        results.append(msg)
    for msg in reversed(results):
        ctx.log("tool", tool=msg.name, status=getattr(msg, "status", "success"), content=msg.content)
