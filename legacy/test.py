from typing import Annotated, Sequence, NotRequired, Any, Union, Dict
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage # The foundational class for all message types in LangGraph
from langchain_core.messages import ToolMessage # Passes data back to LLM after it calls a tool such as the content and the tool_call_id
from langchain_core.messages import SystemMessage # Message for providing instructions to the LLM
from langchain_openai import ChatOpenAI
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import pandas as pd
from functools import wraps
import inspect
import os, sys, ipdb


# LangSmith configuration
os.environ['LANGSMITH_TRACING'] = 'true'
os.environ['LANGSMITH_ENDPOINT'] = "https://api.smith.langchain.com"
os.environ['LANGSMITH_PROJECT'] = "mle"

# ------------------------------
# Agent State Definition
# ------------------------------

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    scratchpad: Dict[str, Any]

# ------------------------------
# Decorator to inject scratchpad variables
# ------------------------------

def scratchpad_tool(func):
    # @wraps(func)
    def wrapper(
        state: Annotated[AgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        bindings: Dict[str, str],
        output: str,
        func_kwargs: Dict[str, Any]
    ):
        print(func_kwargs)
        scratchpad = state.get("scratchpad", {})
        func_args = {
            arg_name: scratchpad.get(scratchpad_key)
            for arg_name, scratchpad_key in bindings.items()
        }
        func_args.update(func_kwargs)
        # print(func_args)
        result = func(**func_args)

        return Command(update={
            "scratchpad": {output: result} if(state.get('scratchpad') is None) else {**state.get('scratchpad'), output: result},
            "messages": [ToolMessage(f"Applied {func.__name__} with docstring: {func.__doc__}, annotations: {func.__annotations__}, and stored result in scratchpad as '{output}'", tool_call_id=tool_call_id)]
        })

    # Construct extended docstring
    extra_doc = """
        This tool reads arguments from the scratchpad using `bindings`, passes them to the function,
        and stores the result in the scratchpad under the given `output` key.

        Additional parameters injected by the framework:

        Parameters
        ----------
        bindings : dict
            A mapping of parameter names (e.g. "df1") to scratchpad keys (e.g. "users").

        output : str
            The name of the scratchpad key to store the result under.

        func_kwargs : dict
            keyword arguments to pass to the function.

        Returns
        -------
        Command
            A LangGraph Command object that updates the agent's state:
            - Stores the result in `scratchpad[output]`
            - Sends a ToolMessage acknowledging the update
        """
    old_doc_prefix = "\n\n The function has the following parameters it expects: \n\n"
    wrapper.__doc__ = (extra_doc.strip() + "\n\n" + old_doc_prefix.strip() + (func.__doc__ or "")).strip()

    return wrapper


@tool
@scratchpad_tool
def join_dataframes(df1: pd.DataFrame, df2: pd.DataFrame, how: str = "inner"):
    """
    Join two DataFrames from the scratchpad.

    Parameters
    ----------
    df1: DataFrame
        The first DataFrame to join.
    df2: DataFrame
        The second DataFrame to join.
    how: str
        Join method (e.g., 'inner', 'left').

    Returns
    -------
    DataFrame
        The joined DataFrame.
    """
    return pd.merge(df1, df2, left_on="id", right_on="user_id", how=how)

# Debug: Print function details
# print("=== join_dataframes function details ===")
# print(f"Function name: {join_dataframes.__name__}")
# print(f"Function doc: {join_dataframes.__doc__}")
# print(f"Function signature: {inspect.signature(join_dataframes)}")
# print(f"Function annotations: {join_dataframes.__annotations__}")
# print(f"Function type: {type(join_dataframes)}")
# print(f"Has __wrapped__: {hasattr(join_dataframes, '__wrapped__')}")
# if hasattr(join_dataframes, '__wrapped__'):
#     print(f"Wrapped function: {join_dataframes.__wrapped__}")
# print("=======================================")

# sys.exit()
# ------------------------------
# Tool Definition
# ------------------------------

# def join_dataframes(state: Annotated[AgentState, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId], bindings: Dict[str, str], output: str, how="inner"):
#     """
#     Join two DataFrames from the scratchpad.

#     Parameters
#     ----------
#     bindings: dict
#         Maps parameter names (e.g., df1, df2) to scratchpad keys.

#     how: str
#         Join method (e.g., 'inner', 'left').

#     output: str
#         Name to store result under in the scratchpad.
#     """

#     func_args = {
#             arg_name: scratchpad.get(scratchpad_key)
#             for arg_name, scratchpad_key in bindings.items()
#         }
#     res = pd.merge(**func_args, left_on="id", right_on="user_id", how=how)
#     return Command(update={
#             "scratchpad": {output: res},
#             "messages": [ToolMessage(f"Stored result in scratchpad as '{output}'", tool_call_id=tool_call_id)]
#         })




# Let's inspect some of the attributes associated with the tool.
# print(join_dataframes.name)
# print(join_dataframes.description)
# print(join_dataframes.args)
# sys.exit()

# original_func = join_dataframes.func
# print(original_func.__name__)                # Function name
# print(original_func.__doc__)                 # Docstring
# print(inspect.signature(original_func))      # True function signature
# print(original_func.__annotations__)   
# print('######################## \n\n')

# print(join_dataframes.__doc__)
# print(inspect.signature(join_dataframes))   
# print(join_dataframes.__annotations__)
# print(join_dataframes.__name__)
# print('########################')
# sys.exit()

# ------------------------------
# LLM + Tool Setup
# ------------------------------

tools = [join_dataframes]
model = ChatOpenAI(model="gpt-4o",
                    base_url='https://pluto-prod-hawang-llm-proxy-9qtfav-0-4000.colligo.dev',
                    api_key=os.environ['OPENAI_API_KEY']).bind_tools(tools)


# ------------------------------
# Graph Definition
# ------------------------------

def model_call(state:AgentState) -> AgentState:
    system_prompt = SystemMessage(content=
        "You are a Data Scientist. Answer the user's question based on the data provided."
    )
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


def should_continue(state: AgentState): 
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls: 
        return "end"
    else:
        return "continue"
    

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)


tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

graph.add_edge("tools", "our_agent")

app = graph.compile()


# builder = StateGraph(AgentState)

# builder.add_node("agent", llm)
# builder.add_node("tools", ToolNode(tools))
# builder.set_entry_point("agent")
# builder.add_edge("agent", "tools")
# builder.add_edge("tools", "agent")
# builder.add_conditional_edges("agent", lambda x: END if x["messages"][-1].content == "done" else "tools")

# graph = builder.compile()

# ------------------------------
# Initial Data Setup
# ------------------------------

users = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
orders = pd.DataFrame({"user_id": [1, 1, 2], "amount": [100, 150, 200]})

state = {
    "messages": [
        {"role": "user", "content": "Join the users and orders data and store the result as 'joined_dataframe' using a left join. The data is in the scratchpad under keys 'users' and 'orders'."}
    ],
    "scratchpad": {
        "users": users,
        "orders": orders
    }
}

# ------------------------------
# Run the Agent
# ------------------------------

# for s in graph.stream(state):
#     print(s)

# response = app.invoke(state, config={"recursion_limit": 50})

# print(response['messages'][-1].content)

current_state = state

while True:
    print(f"State ID: {id(current_state)}, Scratchpad ID: {id(current_state['scratchpad'])}, Messages ID: {id(current_state['messages'])} \n\n")

    user_input = input("\nYou: ")
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        break

    # Add the new user message to the ongoing state
    current_state["messages"].append(HumanMessage(content=user_input))
    
    # Run the app
    response = app.invoke(current_state)

    # Update the full state
    current_state = response

    # Show assistant message
    print(f"\n\n\n {response} \n\n\n")
    print(f"Assistant: {response['messages'][-1].content}")