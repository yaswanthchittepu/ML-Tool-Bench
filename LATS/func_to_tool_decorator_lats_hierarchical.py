from typing import Annotated, Sequence, NotRequired, Any, Union, Dict, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
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
from cls import Reflection, Node, TreeState, TreeState_Verifiable_Rewards, Node_Verifiable_Rewards
import cls_hierarchical as cls_h

Node = cls_h.Node_Verifiable_Rewards
TreeState = cls_h.MultiStageTreeState

# ------------------------------------------------------------
# Agent State Definition
# ------------------------------------------------------------

# class Reflection(BaseModel):
#     reflections: str = Field(
#         description="The critique and reflections on the sufficiency, superfluency,"
#         " and general quality of the response"
#     )
#     score: int = Field(
#         description="Score from 0-10 on the quality of the candidate response.",
#         gte=0,
#         lte=10,
#     )
#     found_solution: bool = Field(
#         description="Whether the response has fully solved the question or task."
#     )

#     def as_message(self):
#         return HumanMessage(
#             content=f"Reasoning: {self.reflections}\nScore: {self.score}"
#         )

#     @property
#     def normalized_score(self) -> float:
#         return self.score / 10.0


# class Node:
#     # We add tree_node as an attribute for quick access to scratchpad and potentially other useful information in the TreeNode
#     def __init__(
#         self,
#         messages: list[BaseMessage],
#         # reflection: Reflection,
#         parent: Optional["Node"] = None,
#         tree_node: Optional["TreeState"] = None
#     ):
#         self.messages = messages
#         self.parent = parent
#         self.tree_node = tree_node
#         self.children = []
#         self.value = 0
#         self.visits = 0
#         # self.reflection = reflection
#         self.depth = parent.depth + 1 if parent is not None else 1
#         # self._is_solved = reflection.found_solution if reflection else False
#         # if self._is_solved:
#         #     self._mark_tree_as_solved()
#         # self.backpropagate(reflection.normalized_score)

#     def __repr__(self) -> str:
#         return (
#             f"<Node value={self.value}, visits={self.visits},"
#             # f" solution={self.messages} reflection={self.reflection}/>"
#         )

#     @property
#     def is_solved(self):
#         """If any solutions exist, we can end the search."""
#         return self._is_solved

#     @property
#     def is_terminal(self):
#         return not self.children

#     @property
#     def best_child_score(self):
#         """Return the child with the highest value."""
#         if not self.children:
#             return None
#         return max(self.children, key=lambda child: int(child.is_solved) * child.value)

#     @property
#     def height(self) -> int:
#         """Check for how far we've rolled out the tree."""
#         if self.children:
#             return 1 + max([child.height for child in self.children])
#         return 1

#     def upper_confidence_bound(self, exploration_weight=1.0):
#         """Return the UCT score. This helps balance exploration vs. exploitation of a branch."""
#         if self.parent is None:
#             raise ValueError("Cannot obtain UCT from root node")
#         if self.visits == 0:
#             return self.value
#         # Encourages exploitation of high-value trajectories
#         average_reward = self.value / self.visits
#         # Encourages exploration of less-visited trajectories
#         exploration_term = math.sqrt(math.log(self.parent.visits) / self.visits)
#         return average_reward + exploration_weight * exploration_term

#     def backpropagate(self, reward: float):
#         """Update the score of this node and its parents."""
#         node = self
#         while node:
#             node.visits += 1
#             node.value = (node.value * (node.visits - 1) + reward) / node.visits
#             node = node.parent

#     # def get_messages(self, include_reflections: bool = True):
#     #     if include_reflections:
#     #         return self.messages + [self.reflection.as_message()]
#     #     return self.messages

#     def get_trajectory(self, include_reflections: bool = True) -> list[BaseMessage]:
#         """Get messages representing this search branch."""
#         messages = []
#         node = self
#         while node:
#             messages.extend(
#                 node.get_messages(include_reflections=include_reflections)[::-1]
#             )
#             node = node.parent
#         # Reverse the final back-tracked trajectory to return in the correct order
#         return messages[::-1]  # root solution, reflection, child 1, ...

#     def _get_all_children(self):
#         all_nodes = []
#         nodes = deque()
#         nodes.append(self)
#         while nodes:
#             node = nodes.popleft()
#             all_nodes.extend(node.children)
#             for n in node.children:
#                 nodes.append(n)
#         return all_nodes

#     def get_best_solution(self):
#         """Return the best solution from within the current sub-tree."""
#         all_nodes = [self] + self._get_all_children()
#         best_node = max(
#             all_nodes,
#             # We filter out all non-terminal, non-solution trajectories
#             key=lambda node: int(node.is_terminal and node.is_solved) * node.value,
#         )
#         return best_node

#     def _mark_tree_as_solved(self):
#         parent = self.parent
#         while parent:
#             parent._is_solved = True
#             parent = parent.parent


# class TreeState(TypedDict):
#     messages: Annotated[Sequence[BaseMessage], add_messages]
#     root: NotRequired[Node]
#     input: NotRequired[str]
#     scratchpad: NotRequired[Dict[str, Any]]
#     # plan: NotRequired[str]
#     # summary: NotRequired[RunningSummary]
#     num_tool_calls: NotRequired[int]

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
        # state: Annotated[AgentState, InjectedState],
        state: Annotated[TreeState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        bindings: Dict[str, str],
        output: str,
        func_kwargs: Dict[str, Any] = {},
    ):
        scratchpad = state.get("scratchpad", {})
        if('predict_target' != func.__name__):
            func_args = {
                arg_name: scratchpad.get(scratchpad_key)
                for arg_name, scratchpad_key in bindings.items()
            }
            func_args.update(func_kwargs)
            result = func(**func_args)
        else:
            func_args = {
                arg_name: scratchpad.get(scratchpad_key) if(arg_name != 'model') else scratchpad.get('best_estimator')
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
                        # "scratchpad": {'train_df': result[0], 'test_df': result[1]} if(state.get('scratchpad') is None) else {**state.get('scratchpad'), 'train_df': result[0], 'test_df': result[1]},
                        "scratchpad": {'train_df': result[0], 'test_df': result[1]},
                        "messages": [ToolMessage(f"Applied {func.__name__} with docstring: {func.__doc__}, annotations: {func.__annotations__}, and stored results in scratchpad as 'train_df' and 'test_df'. The mapping between the function parameters and the scratchpad keys accessed is {bindings}. Number of Tools Called: {state.get('num_tool_calls', 0)}", tool_call_id=tool_call_id)]
                        # "messages": [ToolMessage(f"Applied {func.__name__}, and stored results in scratchpad as 'train_df' and 'test_df'. The mapping between the function parameters and the scratchpad keys accessed is {bindings}. Number of Tools Called: {state.get('num_tool_calls', 0)}", tool_call_id=tool_call_id)]
                    })
                    # , {"messages": [ToolMessage(f"Applied {func.__name__} with docstring: {func.__doc__}, annotations: {func.__annotations__}, and stored results in scratchpad as 'train_df' and 'test_df'. The mapping between the function parameters and the scratchpad keys accessed is {bindings}. Number of Tools Called: {state.get('num_tool_calls', 0)}", tool_call_id=tool_call_id)]}
                elif(func.__name__ == "convert_dataframe_to_features_target"):
                    if(result[1] is None):
                        return Command(update={
                            "num_tool_calls": state.get('num_tool_calls', 0) + 1,
                            "scratchpad": {'X_test': result[0]},
                            "messages": [ToolMessage(f"Applied {func.__name__} with docstring: {func.__doc__}, annotations: {func.__annotations__}, and stored results in scratchpad as 'X_test'. The mapping between the function parameters and the scratchpad keys accessed is {bindings}. Number of Tools Called: {state.get('num_tool_calls', 0)}", tool_call_id=tool_call_id)]
                            # "messages": [ToolMessage(f"Applied {func.__name__}, and stored results in scratchpad as 'X_test'. The mapping between the function parameters and the scratchpad keys accessed is {bindings}. Number of Tools Called: {state.get('num_tool_calls', 0)}", tool_call_id=tool_call_id)]
                        })
                    else:   
                        return Command(update={
                            "num_tool_calls": state.get('num_tool_calls', 0) + 1,
                            "scratchpad": {'X_train': result[0], 'Y_train': result[1]},
                            "messages": [ToolMessage(f"Applied {func.__name__} with docstring: {func.__doc__}, annotations: {func.__annotations__}, and stored results in scratchpad as 'X_train' and 'Y_train'. The mapping between the function parameters and the scratchpad keys accessed is {bindings}. Number of Tools Called: {state.get('num_tool_calls', 0)}", tool_call_id=tool_call_id)]
                            # "messages": [ToolMessage(f"Applied {func.__name__}, and stored results in scratchpad as 'X_train' and 'Y_train'. The mapping between the function parameters and the scratchpad keys accessed is {bindings}. Number of Tools Called: {state.get('num_tool_calls', 0)}", tool_call_id=tool_call_id)]
                        })
                elif(func.__name__ == "split_data"):
                    return Command(update={
                        "num_tool_calls": state.get('num_tool_calls', 0) + 1,
                        # "scratchpad": {'X_train': result[0], 'X_test': result[1], 'Y_train': result[2], 'Y_test': result[3]} if(state.get('scratchpad') is None) else {**state.get('scratchpad'), 'X_train': result[0], 'X_test': result[1], 'Y_train': result[2], 'Y_test': result[3]},
                        "scratchpad": {'X_train': result[0], 'X_test': result[1], 'Y_train': result[2], 'Y_test': result[3]},
                        "messages": [ToolMessage(f"Applied {func.__name__} with docstring: {func.__doc__}, annotations: {func.__annotations__}, and stored results in scratchpad as 'X_train', 'X_test', 'Y_train', and 'Y_test'. The mapping between the function parameters and the scratchpad keys accessed is {bindings}. Number of Tools Called: {state.get('num_tool_calls', 0)}", tool_call_id=tool_call_id)]
                        # "messages": [ToolMessage(f"Applied {func.__name__}, and stored results in scratchpad as 'X_train', 'X_test', 'Y_train', and 'Y_test'. The mapping between the function parameters and the scratchpad keys accessed is {bindings}. Number of Tools Called: {state.get('num_tool_calls', 0)}", tool_call_id=tool_call_id)]
                    })
                    # , {"messages": [ToolMessage(f"Applied {func.__name__} with docstring: {func.__doc__}, annotations: {func.__annotations__}, and stored results in scratchpad as 'X_train', 'X_test', 'Y_train', and 'Y_test'. The mapping between the function parameters and the scratchpad keys accessed is {bindings}. Number of Tools Called: {state.get('num_tool_calls', 0)}", tool_call_id=tool_call_id)]}
                elif('tune' in func.__name__):
                    return Command(update={
                        "num_tool_calls": state.get('num_tool_calls', 0) + 1,
                        "scratchpad": {'best_estimator': result[0], 'best_params': result[1], 'best_cv_score': result[2]},
                        "messages": [ToolMessage(f"Applied {func.__name__} with docstring: {func.__doc__}, annotations: {func.__annotations__}, and stored results in scratchpad as 'best_estimator', 'best_params', and 'best_cv_score'. The Best params and CV score for this method are {result[1]} and {result[2]} respectively. The mapping between the function parameters and the scratchpad keys accessed is {bindings}. Number of Tools Called: {state.get('num_tool_calls', 0)}", tool_call_id=tool_call_id)]
                        # "messages": [ToolMessage(f"Applied {func.__name__}, and stored results in scratchpad as 'best_estimator', 'best_params', and 'best_cv_score'. The Best params and CV score for this method are {result[1]} and {result[2]} respectively. The mapping between the function parameters and the scratchpad keys accessed is {bindings}. Number of Tools Called: {state.get('num_tool_calls', 0)}", tool_call_id=tool_call_id)]
                    })
                else:
                    raise ValueError(f"Function {func.__name__} returned a tuple, but it is not supported by the decorator.")

        return Command(update={
            "num_tool_calls": state.get('num_tool_calls', 0) + 1,
            # "scratchpad": {output: result} if(state.get('scratchpad') is None) else {**state.get('scratchpad'), output: result},
            "scratchpad": {output: result},
            "messages": [ToolMessage(f"Applied {func.__name__} with docstring: {func.__doc__}, annotations: {func.__annotations__}, and stored result in scratchpad as '{output}'. The mapping between the function parameters and the scratchpad keys accessed is {bindings}>. Number of Tools Called: {state.get('num_tool_calls', 0)}", tool_call_id=tool_call_id)]
            # "messages": [ToolMessage(f"Applied {func.__name__}, and stored result in scratchpad as '{output}'. The mapping between the function parameters and the scratchpad keys accessed is {bindings}>. Number of Tools Called: {state.get('num_tool_calls', 0)}", tool_call_id=tool_call_id)]
        })
        # , {"messages": [ToolMessage(f"Applied {func.__name__} with docstring: {func.__doc__}, annotations: {func.__annotations__}, and stored result in scratchpad as '{output}'. The mapping between the function parameters and the scratchpad keys accessed is {bindings}>. Number of Tools Called: {state.get('num_tool_calls', 0)}", tool_call_id=tool_call_id)]}

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
        state: Annotated[TreeState, InjectedState],
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
            "messages": [ToolMessage(f"Applied {func.__name__} with docstring: {func.__doc__}, annotations: {func.__annotations__} and result is {result}. \
                                        The mapping between the function parameters and the scratchpad keys accessed is {bindings}. \
                                        Number of Tools Called: {state.get('num_tool_calls', 0)}", tool_call_id=tool_call_id)]
                                        # "messages": [ToolMessage(f"Applied {func.__name__} and result is {result}. The mapping between the function parameters and the scratchpad keys accessed is {bindings}. Number of Tools Called: {state.get('num_tool_calls', 0)}", tool_call_id=tool_call_id)]
        })
        # , {"messages": [ToolMessage(f"Applied {func.__name__} with docstring: {func.__doc__}, annotations: {func.__annotations__} and result is {result}. The mapping between the function parameters and the scratchpad keys accessed is {bindings}. Number of Tools Called: {state.get('num_tool_calls', 0)}", tool_call_id=tool_call_id)]}

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
        state: Annotated[TreeState, InjectedState],
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
            # "scratchpad": {output: result} if(state.get('scratchpad') is None) else {**state.get('scratchpad'), output: result},
            "scratchpad": {output: result},
            "messages": [ToolMessage(f"Applied {func.__name__} with docstring: {func.__doc__}, annotations: {func.__annotations__} and stored result in scratchpad as '{output}'. Number of Tools Called: {state.get('num_tool_calls', 0)}", tool_call_id=tool_call_id)]
            # "messages": [ToolMessage(f"Applied {func.__name__} and stored result in scratchpad as '{output}'. Number of Tools Called: {state.get('num_tool_calls', 0)}", tool_call_id=tool_call_id)]
        })
        # , {"messages": [ToolMessage(f"Applied {func.__name__} with docstring: {func.__doc__}, annotations: {func.__annotations__} and stored result in scratchpad as '{output}'. Number of Tools Called: {state.get('num_tool_calls', 0)}", tool_call_id=tool_call_id)]}

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
        state: Annotated[TreeState, InjectedState],
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
            # "scratchpad": {output: result} if(state.get('scratchpad') is None) else {**state.get('scratchpad'), output: result},
            "scratchpad": {output: result},
            "messages": [ToolMessage(f"Applied {func.__name__} with docstring: {func.__doc__}, annotations: {func.__annotations__}, and stored result in scratchpad as '{output}'. The mapping between the function parameters and the scratchpad keys accessed is {bindings}. Number of Tools Called: {state.get('num_tool_calls', 0)}", tool_call_id=tool_call_id)]
            # "messages": [ToolMessage(f"Applied {func.__name__}, and stored result in scratchpad as '{output}'. The mapping between the function parameters and the scratchpad keys accessed is {bindings}. Number of Tools Called: {state.get('num_tool_calls', 0)}", tool_call_id=tool_call_id)]
        })
        # , {"messages": [ToolMessage(f"Applied {func.__name__} with docstring: {func.__doc__}, annotations: {func.__annotations__}, and stored result in scratchpad as '{output}'. The mapping between the function parameters and the scratchpad keys accessed is {bindings}. Number of Tools Called: {state.get('num_tool_calls', 0)}", tool_call_id=tool_call_id)]}

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