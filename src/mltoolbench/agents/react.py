"""ReAct baselines: a flat tool-calling loop, and a staged variant with tool masking.

* :class:`ReActAgent` — the paper's ReAct baseline: plan once, then call one tool
  per turn until the model stops calling tools.
* :class:`HierarchicalReActAgent` — an extra baseline (not in the paper): walks the
  10 pipeline stages in order, planning per stage and exposing only that stage's tools.
"""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from mltoolbench.agents.base import Agent, RunContext
from mltoolbench.env import stage_names, tools_for_stage
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
    stage: str
    stage_plans: dict[str, str]


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


# --------------------------------------------------------------------------- hierarchical

STAGE_PROMPTS = {
    "train_data_loading": "You are a Data Scientist in the Data Loading stage of solving a Kaggle challenge, using only the tools available to you. Load the train and test data and explore the data thoroughly. This stage ends when you have loaded the train data successfully.",
    "test_data_loading": "You are a Data Scientist in the Data Loading stage of solving a Kaggle challenge, using only the tools available to you. This stage ends when you have loaded the test data successfully.",
    "combine_train_test": "You are a Data Scientist in the Data Loading stage of solving a Kaggle challenge, using only the tools available to you. This stage ends when you have combined the train and test data into a single dataframe successfully, to be used for downstream Data Cleaning and Feature Engineering. This is to ensure same data preprocessing techniques are applied to both partitions.",
    "data_cleaning": "You are a Data Scientist in the Data Cleaning stage of solving a Kaggle challenge, using only the tools available to you. This stage ends when there are no missing values present in the data. This also includes the column corresponding to the target variable, that may have Nans in the test partition since the target variable is not present in the test partition. You are allowed to be innovative in filling the missing values based on your understanding of the data",
    "feature_engineering": "You are a Data Scientist in the Feature Engineering stage of solving a Kaggle challenge, using only the tools available to you. Create new features, or delete unimportant features or transform existing features as needed. You are not allowed to delete or modify features that indicate if the row in the data belongs to the train or test partition. You are also not allowed to augment the feature corresponding to the target variable. Use your understanding of the data to aid your decisions. This stage ends when you feel that the features are good enough for modeling, and categorical and numerical features have been properly encoded. After the end of this stage, all the features should be int or float. Ensure that the number of features in the transformed data is not exorbitantly large.",
    "split_train_test": "You are a Data Scientist in the Split Train Test stage of solving a Kaggle challenge, using only the tools available to you. Split the combined train and test data into train and test dataframes. This stage ends when the train and test dataframes are successfully split from the combined dataframe.",
    "train_data_to_features_target": "You are a Data Scientist in the Converting the Train Data to Features and Target stage of solving a Kaggle challenge, using only the tools available to you. Convert the train data into features and target. This stage ends when the train data is successfully converted into features and target, for making downstream modeling upon.",
    "test_data_to_features": "You are a Data Scientist in the Converting the Test Data to Features stage of solving a Kaggle challenge, using only the tools available to you. Convert the test data into features. This stage ends when the test data is successfully converted into features, for making downsream predictions upon.",
    "modeling": "You are a Data Scientist in the Modeling stage of solving a Kaggle challenge, using only the tools available to you. Train and tune models. You might need to experiment with different model choices and properly tune your hyperparameters to get good performance. Use the provided evaluation tools to evaluate your trained models if needed. This stage ends when the agent has successfully created a model, that it considers to be the best",
    "create_submission_dataframe": "You are a Data Scientist in the Create Submission stage of solving a Kaggle challenge, using only the tools available to you. Make predictions on the test data, and create a submission dataframe that contains the predictions in the requested format.  This stage ends when the submission dataframe in the correct format is created.",
}


class HierarchicalReActAgent(Agent):
    name = "hierarchical_react"

    def __init__(self, recursion_limit: int = 500):
        self.recursion_limit = recursion_limit

    def run(self, ctx: RunContext) -> dict[str, Any]:
        stages = stage_names()
        task = HumanMessage(ctx.prompt())
        stage_tools = {s: langchain_tools(tools_for_stage(s)) for s in stages}
        base = ctx.chat_model()
        models = {s: bind_tools(base, stage_tools[s], ctx.llm) for s in stages}

        def plan_node(state: ReActState) -> dict:
            stage = state["stage"]
            ask = (f"{STAGE_PROMPTS[stage]} Propose a step by step plan for the current stage of the problem i.e "
                   f"{stage}, that will help in clearing the current stage of the problem. Only propose the plan for "
                   f"the current stage you are at. Do not include any other stage's tasks in the plan.")
            response = models[stage].invoke([HumanMessage(ask), task])
            ctx.log("plan", stage=stage, content=response.content)
            return {"stage_plans": {**state.get("stage_plans", {}), stage: response.content}}

        def agent_node(state: ReActState) -> dict:
            stage = state["stage"]
            _log_tool_results(ctx, state["messages"])
            guide = (f"\n\nThe plan to solve the current stage i.e {stage} of the problem is: "
                     f"{state['stage_plans'][stage]}. Use the plan to guide your choice of tools to call\n\n")
            response = normalize_tool_calls(models[stage].invoke(
                [SystemMessage(STAGE_PROMPTS[stage]), task, HumanMessage(guide), *state["messages"]]))
            _log_ai(ctx, response, stage=stage)
            return {"messages": [response]}

        def advance(state: ReActState) -> dict:
            nxt = stages[stages.index(state["stage"]) + 1]
            ctx.log("stage_transition", from_stage=state["stage"], to_stage=nxt)
            return {"stage": nxt}

        def after_agent(state: ReActState) -> str:
            if _has_tool_calls(state):
                return f"{state['stage']}/tools"
            return END if state["stage"] == stages[-1] else "advance"

        graph = StateGraph(ReActState)
        graph.add_node("plan", plan_node)
        graph.add_node("advance", advance)
        for s in stages:
            graph.add_node(f"{s}/agent", agent_node)
            graph.add_node(f"{s}/tools", ToolNode(stage_tools[s]))
            graph.add_edge(f"{s}/tools", f"{s}/agent")
            graph.add_conditional_edges(f"{s}/agent", after_agent, [f"{s}/tools", "advance", END])
        graph.add_conditional_edges("plan", lambda st: f"{st['stage']}/agent", [f"{s}/agent" for s in stages])
        graph.add_edge("advance", "plan")
        graph.set_entry_point("plan")
        init = {"messages": [], "scratchpad": {}, "num_tool_calls": 0, "stage": stages[0], "stage_plans": {}}
        return _invoke(ctx, graph, init, self.recursion_limit)


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


def _log_ai(ctx: RunContext, message: AIMessage, **extra: Any) -> None:
    calls = [{"tool": c["name"], "args": c["args"]} for c in message.tool_calls]
    ctx.log("llm", content=message.content, tool_calls=calls, **extra)


def _log_tool_results(ctx: RunContext, messages: list[BaseMessage]) -> None:
    results = []
    for msg in reversed(messages):  # the tool results since the last AI message
        if not isinstance(msg, ToolMessage):
            break
        results.append(msg)
    for msg in reversed(results):
        ctx.log("tool", tool=msg.name, status=getattr(msg, "status", "success"), content=msg.content)
