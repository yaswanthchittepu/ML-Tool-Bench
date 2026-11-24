from typing import Annotated, Sequence, NotRequired, Any, Union, Dict
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage # The foundational class for all message types in LangGraph
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langmem.short_term import SummarizationNode, RunningSummary, summarize_messages
import os, sys, ipdb    
import pandas as pd
import pickle

# ------------------------------------------------------------
# Agent State Definition
# ------------------------------------------------------------

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    scratchpad: NotRequired[Dict[str, Any]]
    plan: Dict[str, Any]
    summary: NotRequired[RunningSummary]
    num_tool_calls: NotRequired[int]

# ------------------------------
# Decorator to inject scratchpad variables
# ------------------------------

# This decorator can access multiple objects in the scratchpad, and pass them to the function, using the bindings parameter.
# Can also set the result of the function to a key in the scratchpad, using the output parameter.
# As of now this decorator only supports writing a single object to the scratchpad, but can read multiple objects from the scratchpad.
# split_combined_into_train_test in feature_engineering is a problem, because it returns a tuple, and the decorator only supports writing a single object to the scratchpad.
# So we add a special case for this function and ignore the output parameter.
# Similar exception for split_data in loading
def make_get_and_set_tool(func):
    def wrapper(
        state: Annotated[AgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        bindings: Dict[str, str],
        output: str,
        func_kwargs: Dict[str, Any] = {},
    ):
        scratchpad = state.get("scratchpad", {})
        func_args = {
            arg_name: scratchpad.get(scratchpad_key)
            for arg_name, scratchpad_key in bindings.items()
        }
        func_args.update(func_kwargs)
        result = func(**func_args)
        
        if(isinstance(result, tuple)):
            if(len(result) == 1):
                result = result[0]
            else:
                if(func.__name__ == "split_combined_into_train_test"):
                    return Command(update={
                        "num_tool_calls": state.get('num_tool_calls', 0) + 1,
                        "scratchpad": {'train_df': result[0], 'test_df': result[1]} if(state.get('scratchpad') is None) else {**state.get('scratchpad'), 'train_df': result[0], 'test_df': result[1]},
                        "messages": [ToolMessage(f"Applied {func.__name__} with docstring: {func.__doc__}, annotations: {func.__annotations__}, and stored results in scratchpad as 'train_df' and 'test_df'. The mapping between the function parameters and the scratchpad keys accessed is {bindings}. Number of Tools Called: {state.get('num_tool_calls', 0)}", tool_call_id=tool_call_id)]
                    })
                elif(func.__name__ == "convert_dataframe_to_features_target"):
                    if(result[1] is None):
                        return Command(update={
                            "num_tool_calls": state.get('num_tool_calls', 0) + 1,
                            "scratchpad": {'X_test': result[0]} if(state.get('scratchpad') is None) else {**state.get('scratchpad'), 'X_test': result[0]},
                            "messages": [ToolMessage(f"Applied {func.__name__} with docstring: {func.__doc__}, annotations: {func.__annotations__}, and stored results in scratchpad as 'X_test'. The mapping between the function parameters and the scratchpad keys accessed is {bindings}. Number of Tools Called: {state.get('num_tool_calls', 0)}", tool_call_id=tool_call_id)]
                        })
                    else:   
                        return Command(update={
                            "num_tool_calls": state.get('num_tool_calls', 0) + 1,
                            "scratchpad": {'X_train': result[0], 'Y_train': result[1]} if(state.get('scratchpad') is None) else {**state.get('scratchpad'), 'X_train': result[0], 'Y_train': result[1]},
                            "messages": [ToolMessage(f"Applied {func.__name__} with docstring: {func.__doc__}, annotations: {func.__annotations__}, and stored results in scratchpad as 'X_train' and 'Y_train'. The mapping between the function parameters and the scratchpad keys accessed is {bindings}. Number of Tools Called: {state.get('num_tool_calls', 0)}", tool_call_id=tool_call_id)]
                        })
                elif(func.__name__ == "split_data"):
                    return Command(update={
                        "num_tool_calls": state.get('num_tool_calls', 0) + 1,
                        "scratchpad": {'X_train': result[0], 'X_val': result[1], 'Y_train': result[2], 'Y_val': result[3]} if(state.get('scratchpad') is None) else {**state.get('scratchpad'), 'X_train': result[0], 'X_val': result[1], 'Y_train': result[2], 'Y_val': result[3]},
                        "messages": [ToolMessage(f"Applied {func.__name__} with docstring: {func.__doc__}, annotations: {func.__annotations__}, and stored results in scratchpad as 'X_train', 'X_val', 'Y_train', and 'Y_val'. The mapping between the function parameters and the scratchpad keys accessed is {bindings}. Number of Tools Called: {state.get('num_tool_calls', 0)}", tool_call_id=tool_call_id)]
                    })
                elif('tune' in func.__name__):
                    return Command(update={
                        "num_tool_calls": state.get('num_tool_calls', 0) + 1,
                        "scratchpad": {'best_estimator': result[0], 'best_params': result[1], 'best_cv_score': result[2]} if(state.get('scratchpad') is None) else {**state.get('scratchpad'), 'best_estimator': result[0], 'best_params': result[1], 'best_cv_score': result[2]},
                        "messages": [ToolMessage(f"Applied {func.__name__} with docstring: {func.__doc__}, annotations: {func.__annotations__}, and stored results in scratchpad as 'best_estimator', 'best_params', and 'best_cv_score'. The Best params and CV score for this method are {result[1]} and {result[2]} respectively. The mapping between the function parameters and the scratchpad keys accessed is {bindings}. Number of Tools Called: {state.get('num_tool_calls', 0)}", tool_call_id=tool_call_id)]
                    })
                else:
                    raise ValueError(f"Function {func.__name__} returned a tuple, but it is not supported by the decorator.")

        return Command(update={
            "num_tool_calls": state.get('num_tool_calls', 0) + 1,
            "scratchpad": {output: result} if(state.get('scratchpad') is None) else {**state.get('scratchpad'), output: result},
            "messages": [ToolMessage(f"Applied {func.__name__} with docstring: {func.__doc__}, annotations: {func.__annotations__}, and stored result in scratchpad as '{output}'. The mapping between the function parameters and the scratchpad keys accessed is {bindings}>. Number of Tools Called: {state.get('num_tool_calls', 0)}", tool_call_id=tool_call_id)]
        })

    # Construct extended docstring
    extra_doc = """
        This tool reads arguments from the scratchpad using `bindings`, passes them to the internal function,
        and stores the result in the scratchpad under the given `output` key.

        Additional parameters injected by the framework:

        Parameters
        ----------
        bindings : dict
            A mapping of parameter names (e.g. "df1") to scratchpad keys (e.g. "users").

        output : str
            The name of the scratchpad key to store the result under.

        func_kwargs : dict
            keyword arguments to pass to the function. This is a dictionary of the arguments that the function expects.
            {
                'arg1': ...,
                'arg2': ...,
                ...
            }
            Default is an empty dictionary.
        Returns
        -------
        Command
            A LangGraph Command object that updates the agent's state:
            - Stores the result in `scratchpad[output]`
            - Sends a ToolMessage acknowledging the update
        """

    old_doc_prefix = "\n\n The internal function to which the objects in the scratchpad listed in `bindings`, and func_kwargs are passed has the following parameters it expects: \n\n"

    wrapper.__doc__ = (extra_doc.strip() + "\n\n" + old_doc_prefix.strip() + (func.__doc__ or "")).strip()
    wrapper.__name__ = func.__name__

    return wrapper

# This decorator can read multiple objects from the scratchpad, and pass them to the function, using the bindings parameter.
# This deorator does not write anything to the scratchpad. Instead it appends the result to the ToolMessage.
# The result has to be a simple data type, like a string, number, or boolean.
def make_get_tool(func):
    def wrapper(
        state: Annotated[AgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        bindings: Dict[str, str],
        func_kwargs: Dict[str, Any] = {}
    ):
        scratchpad = state.get("scratchpad", {})
        func_args = {
            arg_name: scratchpad.get(scratchpad_key)
            for arg_name, scratchpad_key in bindings.items()
        }
        func_args.update(func_kwargs)
        result = func(**func_args)

        return Command(update={
            "num_tool_calls": state.get('num_tool_calls', 0) + 1,
            "messages": [ToolMessage(f"Applied {func.__name__} with docstring: {func.__doc__}, annotations: {func.__annotations__} and result is {result}. The mapping between the function parameters and the scratchpad keys accessed is {bindings}. Number of Tools Called: {state.get('num_tool_calls', 0)}", tool_call_id=tool_call_id)]
        })

    # Construct extended docstring
    extra_doc = """
        This tool reads arguments from the scratchpad using `bindings`, passes them to the internal function.

        Additional parameters injected by the framework:

        Parameters
        ----------
        bindings : dict
            A mapping of parameter names (e.g. "df1") to scratchpad keys (e.g. "users").

        func_kwargs : dict
            keyword arguments to pass to the function. This is a dictionary of the arguments that the function expects.
            {
                'arg1': ...,
                'arg2': ...,
                ...
            }
            Default is an empty dictionary.

        Returns
        -------
        Command
            A LangGraph Command object that updates the agent's state:
            - Sends a ToolMessage acknowledging the successful tool call
        """

    old_doc_prefix = "\n\n The internal function to which the objects in the scratchpad listed in `bindings`, and func_kwargs are passed has the following parameters it expects: \n\n"

    wrapper.__doc__ = (extra_doc.strip() + "\n\n" + old_doc_prefix.strip() + (func.__doc__ or "")).strip()
    wrapper.__name__ = func.__name__
    return wrapper

# This decorator does not read anything from the scratchpad
# It passes the func_kwargs to the function, and sets the result of the function to the 'output' key in the scratchpad.
# Supports setting only 1 object to the scratchpad, for now.
def make_set_tool(func):
    def wrapper(
        state: Annotated[AgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        output: str,
        func_kwargs: Dict[str, Any] = {}
    ):
        
        result = func(**func_kwargs)

        if(isinstance(result, tuple)):
            assert len(result) == 1, "The make_set_tool decorator only supports a single return value"
            result = result[0]

        return Command(update={
            "num_tool_calls": state.get('num_tool_calls', 0) + 1,
            "scratchpad": {output: result} if(state.get('scratchpad') is None) else {**state.get('scratchpad'), output: result},
            "messages": [ToolMessage(f"Applied {func.__name__} with docstring: {func.__doc__}, annotations: {func.__annotations__} and stored result in scratchpad as '{output}'. Number of Tools Called: {state.get('num_tool_calls', 0)}", tool_call_id=tool_call_id)]
        })

    # Construct extended docstring
    extra_doc = """
        This tool passes the func_kwargs to the internal function and stores the result in the scratchpad under the given `output` key.

        Additional parameters injected by the framework:

        Parameters
        ----------

        output : str
            The name of the scratchpad key to store the result under.

        func_kwargs : dict
            keyword arguments to pass to the function. This is a dictionary of the arguments that the function expects.
            {
                'arg1': ...,
                'arg2': ...,
                ...
            }
            Default is an empty dictionary.
        Returns
        -------
        Command
            A LangGraph Command object that updates the agent's state:
            - Stores the result in `scratchpad[output]`
            - Sends a ToolMessage acknowledging the update
        """

    old_doc_prefix = "\n\n The internal function to which func_kwargs are passed has the following parameters it expects: \n\n"
    wrapper.__doc__ = (extra_doc.strip() + "\n\n" + old_doc_prefix.strip() + (func.__doc__ or "")).strip()
    wrapper.__name__ = func.__name__
    return wrapper


# This decorator can access a single object in the scratchpad, and pass them to the function, using the bindings parameter.
# The result is stored in the scratchpad in the same place it was accessed from i.e overrides the object in the scratchpad.
def override_tool(func):
    def wrapper(
        state: Annotated[AgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        bindings: Dict[str, str],
        func_kwargs: Dict[str, Any] = {},
    ):
        assert len(bindings) == 1, "The override_tool decorator only supports a single binding"
        scratchpad = state.get("scratchpad", {})
        func_args = {
            arg_name: scratchpad.get(scratchpad_key)
            for arg_name, scratchpad_key in bindings.items()
        }
        func_args.update(func_kwargs)
        result = func(**func_args)
        output = list(bindings.values())[0]
        
        if(isinstance(result, tuple)):
            assert len(result) == 1, "The override_tool decorator only supports a single return value"
            result = result[0]

        return Command(update={
            "num_tool_calls": state.get('num_tool_calls', 0) + 1,
            "scratchpad": {output: result} if(state.get('scratchpad') is None) else {**state.get('scratchpad'), output: result},
            "messages": [ToolMessage(f"Applied {func.__name__} with docstring: {func.__doc__}, annotations: {func.__annotations__}, and stored result in scratchpad as '{output}'. The mapping between the function parameters and the scratchpad keys accessed is {bindings}. Number of Tools Called: {state.get('num_tool_calls', 0)}", tool_call_id=tool_call_id)]
        })

    # Construct extended docstring
    extra_doc = """
        This tool reads arguments from the scratchpad using `bindings`, passes them to the internal function,
        and stores the result in the scratchpad under the given `output` key.

        Additional parameters injected by the framework:

        Parameters
        ----------
        bindings : dict
            A mapping of parameter names (e.g. "df1") to scratchpad keys (e.g. "users").

        func_kwargs : dict
            keyword arguments to pass to the function. This is a dictionary of the arguments that the function expects.
            {
                'arg1': ...,
                'arg2': ...,
                ...
            }
            Default is an empty dictionary.
        Returns
        -------
        Command
            A LangGraph Command object that updates the agent's state:
            - Sends a ToolMessage acknowledging the update
        """

    old_doc_prefix = "\n\n The internal function to which the objects in the scratchpad listed in `bindings`, and func_kwargs are passed has the following parameters it expects: \n\n"
    wrapper.__doc__ = (extra_doc.strip() + "\n\n" + old_doc_prefix.strip() + (func.__doc__ or "")).strip()
    wrapper.__name__ = func.__name__

    return wrapper




if __name__ == "__main__":

    from dotenv import load_dotenv
    load_dotenv()

    # print(read_table.name, read_table.description)
    # print(read_table.__name__, '\n', read_table.__doc__, '\n', read_table.__annotations__)
    # sys.exit()

    @tool
    @make_get_and_set_tool
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

    @tool
    @make_set_tool
    def read_table(table_name: str):
        """
        Read a table.

        Parameters
        ----------
        table_name: str
            The name of the table to read.
            The name of the table has to be either 'users' or 'orders'.

        Returns
        -------
        DataFrame: pd.DataFrame
            The table.
        """
        users = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
        orders = pd.DataFrame({"user_id": [1, 1, 2], "amount": [100, 150, 200]})
        return users if table_name == "users" else orders

    @tool
    @make_get_tool
    def summarize_data(df: pd.DataFrame):
        """
        Summarize the data.

        Parameters
        ----------
        df: DataFrame
            The DataFrame to summarize.

        Returns
        -------
        str
            The summary of the data.
        """

        return df.describe().to_string() + "\n\n" + df.head().to_string()


    tools = [join_dataframes, read_table, summarize_data]
    model = ChatOpenAI(model=os.environ['OPENAI_MODEL'],
                        base_url=os.environ['OPENAI_BASE_URL'],
                        api_key=os.environ['OPENAI_API_KEY']).bind_tools(tools, parallel_tool_calls=False)


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
        
        # Check for GPT-style tool_calls
        if last_message.tool_calls: 
            return "continue"
        
        # Check for Llama JSON format in content
        if hasattr(last_message, 'content') and last_message.content:
            try:
                import json
                content = last_message.content.strip()
                if content.startswith('{') and content.endswith('}'):
                    parsed = json.loads(content)
                    if isinstance(parsed, dict) and 'name' in parsed and 'parameters' in parsed:
                        return "continue"
            except (json.JSONDecodeError, AttributeError):
                pass
        
        return "end"
        

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


    current_state = {"messages": []}

    while True:
        
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