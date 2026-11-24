from copy import deepcopy
import math
import uuid
import sys, os, json
import warnings

# Get the absolute path of the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import feature_engineering as fe
import hyperparameter_tuning as hpt
import evaluation as ev
import mle_utils as ut
import modeling as md
import loading as ld
from func_to_tool_decorator_lats_rollout_verifiable import make_get_and_set_tool, make_get_tool, make_set_tool, override_tool
from challenges import CHALLENGES

from cls import Reflection, Node, TreeState, TreeState_Verifiable_Rewards_Rollout, Node_Verifiable_Rewards_Rollout, verifiable_rewards, verifiable_rewards_outcome_based, verifiable_rewards_outcome_based_rollout, verifiable_rewards_rollout

from collections import deque, defaultdict
from typing import Optional
from typing_extensions import TypedDict
from typing import Literal
from typing import Annotated, Sequence, NotRequired, Any, Union, Dict

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage, AIMessage # The foundational class for all message types in LangGraph
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langmem.short_term import SummarizationNode, RunningSummary, summarize_messages

from langgraph.graph import END, StateGraph, START
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langgraph.prebuilt import ToolNode
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import chain as as_runnable
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import RunnableConfig
from langchain.callbacks import get_openai_callback

from pydantic import BaseModel, Field

import pandas as pd
import numpy as np
import time

import ipdb
from dotenv import load_dotenv
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()


def extract_score_from_reward(reward_dict):
    """
    Extract numeric score from reward dictionary.
    For verifiable rewards, we sum all reward values.
    """
    if isinstance(reward_dict, dict):
        return sum(reward_dict.values())
    elif isinstance(reward_dict, (int, float)):
        return float(reward_dict)
    else:
        return 0.0

def save_lats_verifiable_rewards_trajectory(solution_node, search_stats, save_directory, agent_id):
    """
    Save LATS verifiable rewards trajectory by traversing from solution node back to root.
    This captures the complete search path that led to the best solution.
    """

    trajectory_data = {
        'agent_type': 'LATS_Verifiable_Rewards',
        'agent_id': agent_id,
        'timestamp': datetime.now().isoformat(),
        'search_statistics': search_stats,
        'solution_node_info': {
            'depth': solution_node.depth,
            'visits': solution_node.visits,
            'value': solution_node.value,
            'total_reward': extract_score_from_reward(solution_node.get_trajectory_reward()),
            'has_scratchpad_data': bool(solution_node.scratchpad)
        }
    }
    
    # Get the complete trajectory from solution back to root
    trajectory_messages = solution_node.get_trajectory(include_reflections=False)  # No reflections in rewards version
    
    # Parse trajectory into structured steps
    trajectory_steps = []
    step_number = 1
    
    for i, message in enumerate(trajectory_messages):
        step_info = {
            'step_number': step_number,
            'message_type': type(message).__name__,
            'timestamp': datetime.now().isoformat()
        }
        
        # Handle different message types
        if hasattr(message, 'tool_calls') and message.tool_calls:
            step_info.update({
                'step_type': 'tool_selection',
                'action': 'selected_tools_for_execution',
                'tools_selected': [tc['name'] for tc in message.tool_calls],
                'tool_calls_detail': message.tool_calls,
                'content': message.content
            })
        elif hasattr(message, 'content'):
            if(isinstance(message, ToolMessage)):
            # if 'tool' in str(type(message)).lower():
                step_info.update({
                    'step_type': 'tool_result',
                    'action': 'received_tool_output',
                    'content_preview': str(message.content)[:300] + "..." if len(str(message.content)) > 300 else str(message.content),
                    'content_length': len(str(message.content))
                })
            elif isinstance(message, AIMessage):
                step_info.update({
                    'step_type': 'ai_reasoning',
                    'action': 'generated_ai_reasoning',
                    'content_preview': message.content[:300] + "..." if len(message.content) > 300 else message.content,
                    'content_length': len(message.content)
                })
            else:
                step_info.update({
                    'step_type': 'reward_feedback',
                    'action': 'generated_reward_feedback',
                    'content_preview': message.content[:300] + "..." if len(message.content) > 300 else message.content,
                    'content_length': len(message.content)
                })
        
        trajectory_steps.append(step_info)
        step_number += 1
    
    # Add reward information along the trajectory
    trajectory_rewards = solution_node.get_trajectory_reward()
    trajectory_data['trajectory_rewards'] = trajectory_rewards
    trajectory_data['trajectory_steps'] = trajectory_steps
    trajectory_data['total_steps'] = len(trajectory_steps)
    
    # Save to file
    reward_type = "outcome" if os.environ.get('MCTS_OUTCOME_REWARD') in ['true', 'True'] else "shaped"
    feedback_type = "feedback" if os.environ.get('USE_FEEDBACK') in ['true', 'True'] else "no_feedback"
    trajectory_file = os.path.join(save_directory, f"lats_verifiable_rewards_{reward_type}_{feedback_type}_trajectory_{agent_id}.json")
    
    with open(trajectory_file, 'w') as f:
        json.dump(trajectory_data, f, indent=2)
    
    return trajectory_file, trajectory_data

# MCTS ROLLOUT CONFIGURATION:
# This file implements limited depth rollouts for MCTS instead of direct state evaluation.
# 
# Environment Variables:
# - MCTS_USE_ROLLOUTS: 'true'/'false' - Enable rollout-based evaluation (default: 'false')
# - MCTS_ROLLOUT_DEPTH: int - Number of steps to simulate in rollout (default: 3)
# - MCTS_ROLLOUT_POLICY: 'random'/'greedy' - Policy for action selection during rollout (default: 'random')
#
# Usage:
#   export MCTS_USE_ROLLOUTS=true
#   export MCTS_ROLLOUT_DEPTH=5
#   export MCTS_ROLLOUT_POLICY=random
#   python lats_ds_verifiable_rewards_simulate.py


if __name__ == "__main__":


    allowed_functions = set(['read_data', 'create_conditional_feature', 'concatenate_train_test', 'fillna_with_median', 'fillna_with_mode',
        'extract_string_pattern', 'split_string_column', 'create_group_aggregation', 'drop_feature', 'cast_columns',
        'normalize_all_numerical_columns', 'encode_all_categorical_columns', 'split_combined_into_train_test',
        'drop_feature', 'get_features', 'tune_catboost_classifier', 'tune_random_forest_classifier',
        'tune_lightgbm_classifier', 'tune_logistic_classifier', 'tune_xgboost_classifier', 'predict_target', 
        'rename_feature', 'concatenate_dataframes'
    ])

    # Load the decorator mapping from JSON file
    with open("function_to_decorator_map.json", "r") as f:
        decorator_map = json.load(f)
    
    # Apply decorators to functions
    print("\n" + "="*80)
    print("APPLYING DECORATORS TO FUNCTIONS")
    print("="*80)
    
    # Module mapping
    module_map = {
        'feature_engineering': fe,
        'loading': ld,
        'modeling': md,
        'hyperparameter_tuning': hpt,
        'evaluation': ev,
        'mle_utils': ut
    }
    
    # Decorator function mapping
    decorator_func_map = {
        'make_get_and_set_tool': make_get_and_set_tool,
        'make_get_tool': make_get_tool,
        'make_set_tool': make_set_tool,
        'override_tool': override_tool
    }
    
    # Apply decorators
    decorated_functions = {}
    for module_name, functions in decorator_map.items():
        module = module_map[module_name]
        decorated_functions[module_name] = {}
        
        for func_name, decorator_name in functions.items():
            # if(func_name not in allowed_functions):
            #     continue
            if(func_name == 'save_dataframe_to_csv'):
                continue
            try:
                # Get the original function
                original_func = getattr(module, func_name)
                
                # Get the decorator function
                decorator_func = decorator_func_map[decorator_name]
                
                # Apply the decorator
                decorated_func = decorator_func(original_func)
                
                # Store the decorated function
                decorated_functions[module_name][func_name] = decorated_func
                
                print(f"✓ Applied {decorator_name} to {module_name}.{func_name}")
                
            except Exception as e:
                print(f"✗ Failed to apply {decorator_name} to {module_name}.{func_name}: {e}")
    
    print(f"\nSuccessfully decorated {sum(len(funcs) for funcs in decorated_functions.values())} functions")

    # Merge all decorated functions into a single dictionary
    all_decorated_functions = {}
    for module_dict in decorated_functions.values():
        all_decorated_functions.update(module_dict)


    # Apply @tool decorator to all decorated functions
    print("\n" + "="*80)
    print("APPLYING @TOOL DECORATOR TO ALL FUNCTIONS")
    print("="*80)
    
    tool_functions = {}
    for func_name, decorated_func in all_decorated_functions.items():
        try:
            # Apply @tool decorator
            tool_func = tool(decorated_func)
            tool_functions[func_name] = tool_func
            print(f"✓ Applied @tool to {func_name}")
        except Exception as e:
            print(f"✗ Failed to apply @tool to {func_name}: {e}")
    
    print(f"\nSuccessfully created {len(tool_functions)} tool functions")

    tools = list(tool_functions.values())

    bind_kwargs = {'parallel_tool_calls': False} if('o3' not in os.environ['OPENAI_MODEL']) else {}
    llm = ChatOpenAI(model=os.environ['OPENAI_MODEL'],
                        base_url=os.environ['OPENAI_BASE_URL'],
                        api_key=os.environ['OPENAI_API_KEY'],
                        temperature=float(os.environ['TEMPERATURE'])
                    ).bind_tools(tools, 
                            **bind_kwargs
                        )
    tool_node = ToolNode(tools=tools)

    def parse_llama_json(state):
        messages = state["messages"]
        last_message = messages[-1]

        # Check if we need to convert Llama JSON format to tool_calls
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            # Try to parse Llama JSON format
            if hasattr(last_message, 'content') and last_message.content:
                try:
                    import json
                    import string
                    import random
                    import ast
                    
                    content = last_message.content.strip()
                    
                    # Handle multiple tool calls separated by semicolons
                    if ';' in content:
                        # Split by semicolon and clean up each part
                        tool_call_strings = [part.strip() for part in content.split(';')]
                    else:
                        # Single tool call
                        tool_call_strings = [content] if content.startswith('{') and content.endswith('}') else []
                    
                    parsed_tool_calls = []
                    additional_kwargs_tool_calls = []
                    
                    for tool_call_string in tool_call_strings:
                        if tool_call_string.startswith('{') and tool_call_string.endswith('}'):
                            try:
                                # Fix malformed JSON by balancing braces
                                open_braces = tool_call_string.count('{')
                                close_braces = tool_call_string.count('}')
                                if open_braces > close_braces:
                                    # Add missing closing braces
                                    tool_call_string += '}' * (open_braces - close_braces)
                                    print(f"Fixed malformed JSON by adding {open_braces - close_braces} closing braces")
                                
                                parsed = json.loads(tool_call_string)
                                # Fix common JSON escaping issues from LLM output
                                # Replace invalid \' with valid \" in JSON strings
                                # cleaned_string = tool_call_string.replace("\\'", "'")
                                # parsed = json.loads(cleaned_string)
                                if isinstance(parsed, dict) and 'name' in parsed and 'parameters' in parsed:
                                    # Fix parameters that are string representations of Python objects
                                    parameters = parsed['parameters']
                                    
                                    for key, value in parameters.items():
                                        if isinstance(value, str):
                                            # Check if it looks like a dict/list representation
                                            if (value.startswith('{') and value.endswith('}')) or \
                                               (value.startswith('[') and value.endswith(']')) or \
                                               (value.startswith('(') and value.endswith(')')):
                                                try:
                                                    # Parse the string representation back to a Python object
                                                    parameters[key] = ast.literal_eval(value)
                                                except (ValueError, SyntaxError) as e:
                                                    print(f"Failed to parse {key} string '{value}': {e}")
                                    
                                    # Generate GPT-style call ID (call_ + 24 random characters)
                                    call_id = 'call_' + ''.join(random.choices(string.ascii_letters + string.digits, k=24))
                                    
                                    # Add to tool_calls list
                                    parsed_tool_calls.append({
                                        'name': parsed['name'],
                                        'args': parameters,
                                        'id': call_id,
                                        'type': 'tool_call'
                                    })
                                    
                                    # Add to additional_kwargs format
                                    additional_kwargs_tool_calls.append({
                                        'id': call_id,
                                        'function': {
                                            'arguments': json.dumps(parameters),
                                            'name': parsed['name']
                                        },
                                        'type': 'function'
                                    })
                            except json.JSONDecodeError as e:
                                # import ipdb; ipdb.set_trace()
                                print(f"Failed to parse individual tool call: {tool_call_string}, error: {e}")
                                continue
                    
                    # If we successfully parsed any tool calls, update the message
                    if parsed_tool_calls:
                        # Modify the existing message in-place to match GPT format
                        last_message.content = ""  # Clear content like GPT does
                        last_message.tool_calls = parsed_tool_calls
                        
                        # Ensure additional_kwargs has tool_calls for compatibility
                        if not hasattr(last_message, 'additional_kwargs'):
                            last_message.additional_kwargs = {}
                        last_message.additional_kwargs['tool_calls'] = additional_kwargs_tool_calls
                            
                except (json.JSONDecodeError, AttributeError) as e:
                    print(f"Failed to parse Llama JSON format: {e}")
        return state

    interaction_info = defaultdict(int)
    worst_case_per_query_info = defaultdict(int)

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a Data Scientist tasked with solving the Kaggle competition provided below, with the tools available to you. Propose tool candidates that would help solving the problem at the current stage. Your Job is to only call tool candidates at the current step. Any interaction apart from this is not allowed.",
                # Also provide the rationale for your choice of tool candidates. You are allowed to propose at most 5 tool candidates. You can also propose less than 5 or even a single tool candidate if you think there are no other valid options at this stage.
            ),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="messages", optional=True),
        ]
    )


    initial_answer_chain = prompt_template | llm.bind_tools(tools=tools).with_config(
        run_name="GenerateInitialCandidate"
    )


    parser = JsonOutputToolsParser(return_id=True)

    ### IN our case, this has to be a pass through function
    def generate_initial_response(state: TreeState_Verifiable_Rewards_Rollout) -> dict:
        return state

    
    def generate_candidates(messages: ChatPromptValue, config: RunnableConfig):
        n = config["configurable"].get("N", 5)
        bound_kwargs = llm.bind_tools(tools=tools).kwargs
        # chat_result = llm.generate(
        #     [messages.to_messages()],
        #     n=n,
        #     callbacks=config["callbacks"],
        #     run_name="GenerateCandidates",
        #     **bound_kwargs,
        # )
        chat_result = llm.generate(
            [messages.to_messages()],
            n=1,
            callbacks=config["callbacks"],
            run_name="GenerateCandidates",
            **bound_kwargs,
        )
        return [gen.message for gen in chat_result.generations[0]]


    # expansion_chain = prompt_template | generate_candidates

    expansion_chain = prompt_template | llm.bind_tools(tools=tools).with_config(
        run_name="GenerateCandidates"
    )

    # res = expansion_chain.invoke({"input": "Write a research report on lithium pollution."})
    # print(res)

    def try_expansion_with_reducing_context(state, messages, config):
        """Try expansion with progressively reducing context on errors."""
        # context_sizes = [None, 20, 15, 10, 5]  # None means full context
        # Calculate dynamic context sizes
        msg_len = len(messages)
        global worst_case_per_query_info
        if msg_len > 0:
            msg_lengths = [len(msg.content) if hasattr(msg, 'content') and msg.content else 0 for msg in messages]
            total_chars = sum(msg_lengths)
            avg_length = total_chars / msg_len
            max_length = max(msg_lengths)
            min_length = min(msg_lengths)
            
            print(f"\n=== MESSAGE STATISTICS ===")
            print(f"Total messages: {msg_len}")
            print(f"Total characters: {total_chars:,}")
            print(f"Average message length: {avg_length:.1f} chars")
            print(f"Longest message: {max_length:,} chars")
            print(f"Shortest message: {min_length:,} chars")
            print(f"===========================\n")
        if msg_len <= 10:
            context_sizes = [None, 5, 1]  # Just use full context if <= 10 messages
        else:
            # Start with nearest multiple of 10 less than msg_len
            start_size = (msg_len // 10) * 10
            if start_size == msg_len:
                start_size = msg_len - 10
            
            # Generate decreasing sizes by 10, minimum 10
            context_sizes = [None]  # Start with full context
            current_size = start_size
            while current_size >= 10:
                context_sizes.append(current_size)
                current_size -= 10
            context_sizes.extend([5, 1])
        
        info = defaultdict(int)
        for i, context_size in enumerate(context_sizes):
            try:
                if context_size is None:
                    # Use full context
                    context_messages = messages
                    print(f"Attempt {i+1}: Using full context ({len(messages)} messages)")
                else:
                    # Use last N messages
                    # ipdb.set_trace()
                    context_messages = messages[-context_size:] if len(messages) > context_size else messages
                    print(f"Attempt {i+1}: Using last {context_size} messages ({len(context_messages)} messages)")
                
                with get_openai_callback() as cb:
                    new_candidates = expansion_chain.invoke(
                        {"input": state["input"], "messages": context_messages}, config
                    )
                    info['total_tokens'] += cb.total_tokens
                    info['prompt_tokens'] += cb.prompt_tokens
                    info['completion_tokens'] += cb.completion_tokens
                    info['total_cost'] += cb.total_cost
                    if(cb.total_tokens > worst_case_per_query_info['total_tokens']):
                        worst_case_per_query_info['total_tokens'] = cb.total_tokens
                        worst_case_per_query_info['prompt_tokens'] = cb.prompt_tokens
                        worst_case_per_query_info['prompt_tokens_cached'] = cb.prompt_tokens_cached
                        worst_case_per_query_info['completion_tokens'] = cb.completion_tokens
                        worst_case_per_query_info['total_cost'] = cb.total_cost
                
                print(f"Success on attempt {i+1}")
                return new_candidates, info
                
            except Exception as e:
                print(f"Attempt {i+1} failed: {e}")
                if i == len(context_sizes) - 1:  # Last attempt
                    print("All attempts failed, re-raising last exception")
                    raise e
                else:
                    print(f"Retrying with smaller context...")
                    continue

    # This is the select function for the tree
    def select(root: Node_Verifiable_Rewards_Rollout) -> dict:
        """Starting from the root node a child node is selected at each tree level until a leaf node is reached."""

        if not root.children:
            return root

        node = root
        while node.children:
            max_child = max(node.children, key=lambda child: child.upper_confidence_bound(exploration_weight=int(os.environ.get('EXPLORATION_WEIGHT', '1'))))
            node = max_child

        return node


    def expand(state: TreeState_Verifiable_Rewards_Rollout, config: RunnableConfig) -> dict:
        """Starting from the "best" node in the tree, generate N candidates for the next step."""
        
        # ipdb.set_trace()
        global interaction_info, all_decorated_functions
        
        root = state["root"]
        best_candidate: Node_Verifiable_Rewards_Rollout = select(root)
        # Combine all messages from root to best_candidate
        messages = best_candidate.get_trajectory()
        # Combine all scratchpads from root to best_candidate
        scratchpad = best_candidate.get_trajectory_scratchpad()
        # Accrue rewards along trajectory
        traj_rewards = best_candidate.get_trajectory_reward()
        # Get stages solved along trajectory
        traj_stages_solved = best_candidate.get_trajectory_stages_solved()
        # Generate N candidates from the single child candidate
        # new_candidates = expansion_chain.invoke(
        #     {"input": state["input"], "messages": messages}, config
        # )
        # Dont expand an already solved node. solved nodes are terminal nodes
        # if(best_candidate._is_solved):
        #     print(f"Best candidate is already solved. Skipping expansion.")
        #     return state
        try:
            new_candidates, info = try_expansion_with_reducing_context(state, messages, config)
            interaction_info['total_tokens'] += info['total_tokens']
            interaction_info['prompt_tokens'] += info['prompt_tokens']
            interaction_info['completion_tokens'] += info['completion_tokens']
            interaction_info['total_cost'] += info['total_cost']
        except Exception as e:
            print(f"All expansion attempts failed: {e}")
            raise e
        # parsed = parser.batch(new_candidates)
        # Need to make input to parser a list and take the first element
        if('llama' in os.environ['OPENAI_MODEL']):
            new_candidates = parse_llama_json({'messages': [new_candidates]})['messages'][0]
        parsed = parser.batch([new_candidates])
        print(f"\n {parsed} \n")

        flattened = [
            (i, tool_call)
            for i, tool_call in enumerate(parsed[0])
        ]
        tool_docs = [
            all_decorated_functions[tool_call["type"]].__doc__ if tool_call.get("type", None) is not None and tool_call["type"] in all_decorated_functions else ""
            for i, tool_call in flattened
        ]
        tool_responses = [
            (
                i,
                tool_node.invoke(
                    # Change the func_to_tool decarotor to only return the new objects in the scratchpad (shouldnt have the old ones)
                    {   
                        **state,
                        # We override the scratchpad and messages here
                        # scrathpad is cumulative from root to best_candidate
                        # scratchpad is used by the tool to access objects (Eg: dataframes, models, etc in our case)
                        # messages is just an AI message with tool info and args
                        "scratchpad": scratchpad,
                        "messages": [
                            AIMessage(
                                content="",
                                tool_calls=[
                                    {
                                        "name": tool_call["type"],
                                        "args": tool_call["args"],
                                        "id": tool_call["id"],
                                    }
                                ],
                            )
                        ]
                    }
                ),
            )
            for i, tool_call in flattened
        ]
        # If successful, use the Commmand to update the state
        # We only deal with scratchpad updates here
        # All the following updates can be merged in a single loop for efficiency ## Future work
        new_node_scratchpads = defaultdict(dict)
        collected_responses = defaultdict(list)
        for i, result in tool_responses:
            # Skip if this is an error
            if isinstance(result, dict) and any(
                isinstance(m, ToolMessage) and getattr(m, "status", None) == "error"
                for m in result.get("messages", [])
            ):
                collected_responses[i].append(result["messages"][0])
                print(f"Tool call {i} failed, skipping.")
                continue

            # If it's a Command, apply the update
            if isinstance(result, list) and all(isinstance(res, Command) for res in result):
                update = result[0].update
                if "scratchpad" in update:
                    new_node_scratchpads[i] = update["scratchpad"]
                if "messages" in update:
                    collected_responses[i].append(update["messages"][0])
            else:
                warnings.warn(f"Unexpected result type for tool call {i}: {type(result)} \n\n Result: {result}")

        output_messages = []

        def split_ai_message_by_tool_calls(ai_message: AIMessage) -> list[AIMessage]:
            tool_calls = ai_message.tool_calls
            if not tool_calls:
                return [ai_message]  # No tool calls, return original message

            split_messages = []
            for tool_call in tool_calls:
                new_message = AIMessage(
                    content=ai_message.content,
                    tool_calls=[tool_call],
                    additional_kwargs={"tool_calls": [tool_call]},
                    response_metadata=ai_message.response_metadata,
                )
                split_messages.append(new_message)
            return split_messages

        new_candidates_split = split_ai_message_by_tool_calls(new_candidates)
        # since new_candidates is not a list, I was seeing the candidates as just 'content' and not a Message class. So fixing this
        for i, candidate in enumerate(new_candidates_split):
            output_messages.append([candidate] + collected_responses[i])


        # Choose between rollout-based evaluation and direct state evaluation
        use_rollouts = int(os.environ.get('MCTS_ROLLOUT_DEPTH', '0')) > 0
        
        if use_rollouts:
            # Use rollout-based evaluation
            # ipdb.set_trace()
            rollout_depth = int(os.environ.get('MCTS_ROLLOUT_DEPTH', '0'))  # Default to 3 steps
            rollout_policy = os.environ.get('MCTS_ROLLOUT_POLICY', 'random')  # random or greedy
            print(f"Using rollout evaluation with depth {rollout_depth} and {rollout_policy} policy")
            
            all_candidate_rewards = []
            all_candidate_human_feedback_msgs = []
            all_candidate_stage_solved = []
            
            for i, output_msg in enumerate(output_messages):
                tdoc_i = [tool_docs[i] if(i < len(tool_docs)) else ""]
                try:
                    # Get candidate state info
                    candidate_messages = output_msg
                    candidate_scratchpad = new_node_scratchpads.get(i, {})
                    candidate_traj_rewards = traj_rewards
                    candidate_traj_stages_solved = traj_stages_solved
                    # Perform rollout to get reward
                    # Here candidate_messages is the a combination of AIMessage and ToolMessage for the current candidate
                    # candidate_scratchpad is the scratchpad for the current candidate
                    # candidate_traj_rewards is the cumulative reward from the root up to and NOT including the current candidate
                    # candidate_traj_stages_solved is the list of stages solved from the root up to and NOT including the current candidate
                    candidate_rewards, candidate_human_feedback_msgs, candidate_stage_solved = rollout(state, messages, scratchpad, candidate_messages, candidate_scratchpad, candidate_traj_rewards, candidate_traj_stages_solved, tdoc_i, problem_type=os.environ['PROBLEM_TYPE'], rollout_depth=rollout_depth, policy=rollout_policy)
                    all_candidate_rewards.append(candidate_rewards[0])
                    all_candidate_human_feedback_msgs.append(candidate_human_feedback_msgs[0])
                    all_candidate_stage_solved.append(candidate_stage_solved[0])
                    
                except Exception as e:
                    print(f"Rollout failed for candidate {i}: {e}. Falling back to direct evaluation.")
                    # Fallback to direct evaluation for this candidate
                    if(os.environ['MCTS_OUTCOME_REWARD'] in ['true', 'True']):
                        fallback_rewards, fallback_msgs, fallback_stage_solved = verifiable_rewards_outcome_based_rollout([output_msg], traj_rewards, {i: new_node_scratchpads.get(i, {})}, traj_stages_solved, tdoc_i, problem_type=os.environ['PROBLEM_TYPE'])
                        print(f"Using outcome based rewards")
                    else:
                        fallback_rewards, fallback_msgs, fallback_stage_solved = verifiable_rewards_rollout([output_msg], traj_rewards, {i: new_node_scratchpads.get(i, {})}, traj_stages_solved, tdoc_i, problem_type=os.environ['PROBLEM_TYPE'])
                        print(f"Using shaped rewards")
                    all_candidate_rewards.append(fallback_rewards[0] if fallback_rewards else 0.0)
                    all_candidate_human_feedback_msgs.append(fallback_msgs[0] if fallback_msgs else HumanMessage(content="Fallback evaluation"))
                    all_candidate_stage_solved.append(fallback_stage_solved[0] if fallback_stage_solved else None)
        else:
            # Use original direct state evaluation
            print("Using direct state evaluation (original behavior)")
            if(os.environ['MCTS_OUTCOME_REWARD'] in ['true', 'True']):
                all_candidate_rewards, all_candidate_human_feedback_msgs, all_candidate_stage_solved = verifiable_rewards_outcome_based_rollout(output_messages, traj_rewards, new_node_scratchpads, traj_stages_solved, tool_docs, problem_type=os.environ['PROBLEM_TYPE'])
                print(f"Using outcome based rewards")
            else:
                all_candidate_rewards, all_candidate_human_feedback_msgs, all_candidate_stage_solved = verifiable_rewards_rollout(output_messages, traj_rewards, new_node_scratchpads, traj_stages_solved, tool_docs, problem_type=os.environ['PROBLEM_TYPE'])
                print(f"Using shaped rewards")
        
        # ipdb.set_trace()
        # Grow tree
        if(os.environ['USE_FEEDBACK'] in ['true', 'True']):
            child_nodes = [
                Node_Verifiable_Rewards_Rollout(
                    cand+[feedback_msg], 
                    parent=best_candidate, 
                    reward=reward, 
                    scratchpad=new_node_scratchpads.get(i, {}), 
                    stage_solved=stage_solved, 
                    rollout_depth=int(os.environ.get('MCTS_ROLLOUT_DEPTH', '0')), 
                    rollout_policy=os.environ.get('MCTS_ROLLOUT_POLICY', 'random')
                )
                for i, (cand, reward, feedback_msg, stage_solved) in enumerate(zip(output_messages, all_candidate_rewards, all_candidate_human_feedback_msgs, all_candidate_stage_solved))
            ]
            print(f"Using rewards with feedback")
        else:
            child_nodes = [
                Node_Verifiable_Rewards_Rollout(cand, parent=best_candidate, reward=reward, scratchpad=new_node_scratchpads.get(i, {}), stage_solved=stage_solved, rollout_depth=int(os.environ.get('MCTS_ROLLOUT_DEPTH', '0')), rollout_policy=os.environ.get('MCTS_ROLLOUT_POLICY', 'random'))
                for i, (cand, reward, stage_solved) in enumerate(zip(output_messages, all_candidate_rewards, all_candidate_stage_solved))
            ]
            print(f"Using rewards without feedback")
        best_candidate.children.extend(child_nodes)
        # We have already extended the tree directly, so we just return the state
        return state

    def rollout(state: TreeState_Verifiable_Rewards_Rollout, all_messages, all_scratchpad, candidate_messages, candidate_scratchpad, candidate_traj_rewards, candidate_traj_stages_solved, tool_doc, problem_type, rollout_depth: int = 3, policy: str = 'random') -> float:
        """
        Perform a limited depth rollout from a candidate state.
        
        Args:
            state: Current tree state
            candidate_messages: Messages for this candidate
            candidate_scratchpad: Scratchpad for this candidate  
            candidate_traj_rewards: Trajectory rewards for this candidate
            rollout_depth: Number of steps to simulate forward
            policy: Policy for action selection ('random' or 'greedy')
            
        Returns:
            Final evaluation score for this rollout
        """

        # Here candidate_messages is the a combination of AIMessage and ToolMessage for the current candidate
        # candidate_scratchpad is the scratchpad for the current candidate
        # candidate_traj_rewards is the cumulative reward from the root up to and NOT including the current candidate
        # candidate_traj_stages_solved is the list of stages solved from the root up to and NOT including the current candidate
        # Create a temporary node for rollout simulation
        global interaction_info, all_decorated_functions
        # ipdb.set_trace()
        all_messages = all_messages.copy()
        all_scratchpad = all_scratchpad.copy()
        current_messages = candidate_messages.copy()
        current_scratchpad = candidate_scratchpad.copy()
        current_traj_rewards = candidate_traj_rewards
        current_traj_stages_solved = candidate_traj_stages_solved.copy()
        
        # The reward here is a unitary list of reward recieved by the current candidate
        # human feedback message is a unitary list of human feedback messages for the current candidate
        # stages solved is a unitary list of the name of the stage solved for the current candidate, None if not solved
        if(os.environ['MCTS_OUTCOME_REWARD'] in ['true', 'True']):
            candidate_rewards, candidate_human_feedback_msgs, candidate_stages_solved = verifiable_rewards_outcome_based_rollout([current_messages], current_traj_rewards, [current_scratchpad], current_traj_stages_solved, tool_doc, problem_type=problem_type)
            print(f"Using outcome based rewards")
        else:
            candidate_rewards, candidate_human_feedback_msgs, candidate_stages_solved = verifiable_rewards_rollout([current_messages], current_traj_rewards, [current_scratchpad], current_traj_stages_solved, tool_doc, problem_type=problem_type)
            print(f"Using shaped rewards")
        rollout_reward = candidate_rewards[0]

        rollout_config = RunnableConfig(
            callbacks=[],
            configurable={"N": 1}  # Generate only 1 candidate per rollout step
        )
        
        if(os.environ['USE_FEEDBACK'] in ['true', 'True']):
            all_messages.extend(current_messages+candidate_human_feedback_msgs)
        else:
            all_messages.extend(current_messages)
        all_scratchpad.update(current_scratchpad)
        if(candidate_stages_solved[0] is not None and candidate_stages_solved[0] not in current_traj_stages_solved):
            current_traj_stages_solved.append(candidate_stages_solved[0])
        current_traj_rewards += candidate_rewards[0]
        entry_node_stage_solved = candidate_stages_solved
        entry_node_human_feedback_msgs = candidate_human_feedback_msgs

        print(f"Starting rollout with depth {rollout_depth}")
        
        for step in range(rollout_depth):
            try:

                # Generate candidates for next step using same expansion logic
                new_candidates, info = try_expansion_with_reducing_context(
                    state, all_messages, rollout_config
                )
                interaction_info['total_tokens'] += info['total_tokens']
                interaction_info['prompt_tokens'] += info['prompt_tokens']
                interaction_info['completion_tokens'] += info['completion_tokens']
                interaction_info['total_cost'] += info['total_cost']
                
                # Parse the candidates
                if('llama' in os.environ['OPENAI_MODEL']):
                    new_candidates = parse_llama_json({'messages': [new_candidates]})['messages'][0]
                parsed = parser.batch([new_candidates])
                print(f"\n {parsed} \n")
                if not parsed or not parsed[0]:
                    break  # No valid actions available
                
                # Action selection based on policy
                if policy == 'random':
                    import random
                    selected_action = random.choice(parsed[0])
                elif policy == 'greedy':
                    # For greedy, we could use a simple heuristic or just take the first
                    # In a more sophisticated implementation, you'd evaluate each action
                    selected_action = parsed[0][0]  # Take first for now
                else:
                    # Default to random
                    import random
                    selected_action = random.choice(parsed[0])

                flattened = [
                    (i, tool_call)
                    for i, tool_call in enumerate([selected_action])
                ]
                tool_docs_rollout = [
                    all_decorated_functions[tool_call["type"]].__doc__ if tool_call.get("type", None) is not None and tool_call["type"] in all_decorated_functions else ""
                    for i, tool_call in flattened
                ]
                selected_tool_id = selected_action['id']
                tool_responses = [
                    (
                        i,
                        tool_node.invoke(
                            # Change the func_to_tool decarotor to only return the new objects in the scratchpad (shouldnt have the old ones)
                            {   
                                **state,
                                # We override the scratchpad and messages here
                                # scrathpad is cumulative from root to best_candidate
                                # scratchpad is used by the tool to access objects (Eg: dataframes, models, etc in our case)
                                # messages is just an AI message with tool info and args
                                "scratchpad": all_scratchpad,
                                "messages": [
                                    AIMessage(
                                        content="",
                                        tool_calls=[
                                            {
                                                "name": tool_call["type"],
                                                "args": tool_call["args"],
                                                "id": tool_call["id"],
                                            }
                                        ],
                                    )
                                ]
                            }
                        ),
                    )
                    for i, tool_call in flattened
                ]
                
                new_node_scratchpads = defaultdict(dict)
                collected_responses = defaultdict(list)
                for i, result in tool_responses:
                    # Skip if this is an error
                    if isinstance(result, dict) and any(
                        isinstance(m, ToolMessage) and getattr(m, "status", None) == "error"
                        for m in result.get("messages", [])
                    ):
                        collected_responses[i].append(result["messages"][0])
                        print(f"Tool call {i} failed, skipping.")
                        continue

                    # If it's a Command, apply the update
                    if isinstance(result, list) and all(isinstance(res, Command) for res in result):
                        update = result[0].update
                        if "scratchpad" in update:
                            new_node_scratchpads[i] = update["scratchpad"]
                        if "messages" in update:
                            collected_responses[i].append(update["messages"][0])
                    else:
                        warnings.warn(f"Unexpected result type for tool call {i}: {type(result)} \n\n Result: {result}")

                output_messages = []

                def split_ai_message_by_tool_calls(ai_message: AIMessage) -> list[AIMessage]:
                    tool_calls = ai_message.tool_calls
                    if not tool_calls:
                        return [ai_message]  # No tool calls, return original message

                    split_messages = []
                    for tool_call in tool_calls:
                        if(tool_call['id'] == selected_tool_id):
                            new_message = AIMessage(
                                content=ai_message.content,
                                tool_calls=[tool_call],
                                additional_kwargs={"tool_calls": [tool_call]},
                                response_metadata=ai_message.response_metadata,
                            )
                            split_messages.append(new_message)
                    return split_messages

                new_candidates_split = split_ai_message_by_tool_calls(new_candidates)
                # since new_candidates is not a list, I was seeing the candidates as just 'content' and not a Message class. So fixing this
                for i, candidate in enumerate(new_candidates_split):
                    output_messages.append([candidate] + collected_responses[i])
                
                if(os.environ['MCTS_OUTCOME_REWARD'] in ['true', 'True']):
                    candidate_rewards, candidate_human_feedback_msgs, candidate_stages_solved = verifiable_rewards_outcome_based_rollout(output_messages, current_traj_rewards, new_node_scratchpads, current_traj_stages_solved, tool_docs_rollout, problem_type=problem_type)
                    print(f"Using outcome based rewards")
                else:
                    candidate_rewards, candidate_human_feedback_msgs, candidate_stages_solved = verifiable_rewards_rollout(output_messages, current_traj_rewards, new_node_scratchpads, current_traj_stages_solved, tool_docs_rollout, problem_type=problem_type)
                    print(f"Using shaped rewards")

                # Compute Reward and update all_messages and all_scratchpad
                if(os.environ['USE_FEEDBACK'] in ['true', 'True']):
                    all_messages.extend(output_messages[0]+candidate_human_feedback_msgs)
                else:
                    all_messages.extend(output_messages[0])
                all_scratchpad.update(new_node_scratchpads[0])
                if(candidate_stages_solved[0] is not None and candidate_stages_solved[0] not in current_traj_stages_solved):
                    current_traj_stages_solved.append(candidate_stages_solved[0])
                current_traj_rewards += candidate_rewards[0]
                rollout_reward += candidate_rewards[0]
            except Exception as e:
                print(f"Rollout step {step} failed: {e}")
                break

        print(f"Finished rollout with depth {rollout_depth}")
        
        return [rollout_reward], entry_node_human_feedback_msgs, entry_node_stage_solved

    def should_loop(state: TreeState_Verifiable_Rewards_Rollout):
        """Determine whether to continue the tree search."""
        root = state["root"]
        # If there is a solved node, we stop the tree search
        solved_nodes = root.get_solved_nodes()
        if(len(solved_nodes) > 0):
            return END
        if root.is_solved:
            return END
        if root.height > int(os.environ.get('MAX_TREE_DEPTH', '50')):
            return END
        return "expand"


    builder = StateGraph(TreeState_Verifiable_Rewards_Rollout)
    builder.add_node("start", generate_initial_response)
    builder.add_node("expand", expand)
    builder.add_edge(START, "start")


    builder.add_conditional_edges(
        "start",
        # Either expand/rollout or finish
        should_loop,
        ["expand", END],
    )
    builder.add_conditional_edges(
        "expand",
        # Either continue to rollout or finish
        should_loop,
        ["expand", END],
    )

    graph = builder.compile()

    mid = uuid.uuid4()
    if(os.environ['USE_FEEDBACK'] in ['true', 'True']):
        if(os.environ['MCTS_OUTCOME_REWARD'] in ['true', 'True']):
            save_directory = f"submissions/{os.environ['COMPETITION_NAME']}/lats_outcome_rewards_feedback"
            question = CHALLENGES[os.environ['COMPETITION_NAME']].format(save_directory=save_directory, save_file_name=f"lats_outcome_rewards_feedback_agent_{mid}.csv")
        else:
            save_directory = f"submissions/{os.environ['COMPETITION_NAME']}/lats_shaped_rewards_feedback"
            question = CHALLENGES[os.environ['COMPETITION_NAME']].format(save_directory=save_directory, save_file_name=f"lats_shaped_rewards_feedback_agent_{mid}.csv")
    else:
        if(os.environ['MCTS_OUTCOME_REWARD'] in ['true', 'True']):
            save_directory = f"submissions/{os.environ['COMPETITION_NAME']}/lats_outcome_rewards_no_feedback"
            question = CHALLENGES[os.environ['COMPETITION_NAME']].format(save_directory=save_directory, save_file_name=f"lats_outcome_rewards_no_feedback_agent_{mid}.csv")
        else:
            save_directory = f"submissions/{os.environ['COMPETITION_NAME']}/lats_shaped_rewards_no_feedback"
            question = CHALLENGES[os.environ['COMPETITION_NAME']].format(save_directory=save_directory, save_file_name=f"lats_shaped_rewards_no_feedback_agent_{mid}.csv")
    os.makedirs(save_directory, exist_ok=True)
    os.makedirs(f"model_saves/{os.environ['COMPETITION_NAME']}/", exist_ok=True)
    # You are only allowed a maximum of 20 tool calls in a trajectory to solve the problem.

    # Most Importantly, save your predictions for the test data, as a csv, to the submissions folder. The csv should have the following columns: PassengerId - Id for each passenger in the test set.
    # Transported - The predicted value. For each passenger, predict either True or False. Name the file as react_agent_{model_name}.csv. Look at the sample_submission.csv file in the data folder for an example of the format the results should be in.

    last_step = None
    tree_node = {'input': question}
    root_node = Node_Verifiable_Rewards_Rollout(messages=[], tree_node=tree_node, stage_solved='init')
    tree_node['root'] = root_node
    
    # Record start time
    start_time = time.time()
    print(f"Starting LATS verifiable rewards rollout search at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    
    # for step in graph.stream({"input": question}):
    for step in graph.stream(tree_node, config={"recursion_limit": 1250 if(int(os.environ['MCTS_ROLLOUT_DEPTH']) > 0) else 750}):
        last_step = step
        step_name, step_state = next(iter(step.items()))
        print(step_name)
        print("rolled out: ", step_state["root"].height)
        print("---")
    
    # Record end time and calculate duration
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nSearch completed at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Total search time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Interaction info: {interaction_info}")
    print(f"Worst case per query info: {worst_case_per_query_info}")

    # ipdb.set_trace()
    solved_nodes = root_node.get_solved_nodes()
    
    # Get best node - either solution or highest valued leaf if no solution found
    if(len(solved_nodes) > 0):
        # Select best node: (has_dataframe) * trajectory_reward
        best_node = max(solved_nodes, key=lambda node: (
            int(any(hasattr(v, 'shape') and hasattr(v, 'columns') for v in node.scratchpad.values()) if node.scratchpad else False) * 
            node.get_trajectory_reward()
        ))
        node_type = "SOLUTION"
        print(f"✓ Best solution found with reward: {best_node.get_trajectory_reward()}")
    else:
        # No solution found, get highest valued leaf node
        def get_best_leaf_node_rollout(root):
            """Get the leaf node with highest reward, regardless of solution status"""
            all_nodes = [root] + root._get_all_children()
            leaf_nodes = [node for node in all_nodes if node.is_terminal]
            if not leaf_nodes:
                # If no leaf nodes, return the highest valued node
                leaf_nodes = all_nodes
            return max(leaf_nodes, key=lambda node: node.get_trajectory_reward())
        
        best_node = get_best_leaf_node_rollout(root_node)
        node_type = "BEST_ATTEMPT"
        print(f"⚠ No solution found, using best attempt node with reward: {best_node.get_trajectory_reward()}")
    
    # Save trajectory by traversing from best node back to root
    print("\n" + "="*70)
    print(f"SAVING LATS VERIFIABLE REWARDS ROLLOUT TRAJECTORY FROM {node_type} TO ROOT")
    print("="*70)
    
    reward_type = "outcome" if os.environ.get('MCTS_OUTCOME_REWARD') in ['true', 'True'] else "shaped"
    feedback_enabled = os.environ.get('USE_FEEDBACK') in ['true', 'True']
    
    search_statistics = {
        'total_search_time': total_time,
        'total_tokens': interaction_info['total_tokens'],
        'total_cost': interaction_info['total_cost'],
        'prompt_tokens': interaction_info['prompt_tokens'], 
        'completion_tokens': interaction_info['completion_tokens'],
        'tree_height': root_node.height,
        'num_solved_nodes': len(solved_nodes),
        'solution_found': len(solved_nodes) > 0,
        'node_type': node_type.lower(),
        'rollout_depth': int(os.environ.get('MCTS_ROLLOUT_DEPTH', '0')),
        'competition_name': os.environ.get('COMPETITION_NAME', 'unknown'),
        'model_used': os.environ.get('OPENAI_MODEL', 'unknown'),
        'reward_type': reward_type,
        'feedback_enabled': feedback_enabled
    }
    
    trajectory_file, trajectory_data = save_lats_verifiable_rewards_trajectory(
        solution_node=best_node,
        search_stats=search_statistics,
        save_directory=save_directory,
        agent_id=str(mid)
    )
    
    print(f"✓ LATS Verifiable Rewards Rollout trajectory saved to: {trajectory_file}")
    print(f"✓ Trajectory contains {trajectory_data['total_steps']} steps from {node_type.lower()} back to root")
    print(f"✓ Best node depth: {best_node.depth}")
    print(f"✓ Best node reward: {best_node.get_trajectory_reward()}")
    print(f"✓ Tree search height: {root_node.height}")
    print(f"✓ Rollout depth: {int(os.environ.get('MCTS_ROLLOUT_DEPTH', '0'))}")
    print(f"✓ Reward type: {reward_type}, Feedback: {feedback_enabled}")
    print(f"✓ Node type: {node_type}")
    
    # Only save submission if we actually found a solution
    if len(solved_nodes) > 0 and best_node.scratchpad:
        for v in best_node.scratchpad.values():
            submission_df = v
            # Generate unique filename with node ID
        if(os.environ['USE_FEEDBACK'] in ['true', 'True']):
            if(os.environ['MCTS_OUTCOME_REWARD'] in ['true', 'True']):
                unique_filename = os.path.join(save_directory, f"lats_outcome_rewards_feedback_agent_manual_{mid}.csv")
            else:
                unique_filename = os.path.join(save_directory, f"lats_shaped_rewards_feedback_agent_manual_{mid}.csv")
        else:
            if(os.environ['MCTS_OUTCOME_REWARD'] in ['true', 'True']):
                unique_filename = os.path.join(save_directory, f"lats_outcome_rewards_no_feedback_agent_manual_{mid}.csv")
            else:
                unique_filename = os.path.join(save_directory, f"lats_shaped_rewards_no_feedback_agent_manual_{mid}.csv")
        ut.save_dataframe_to_csv(submission_df, unique_filename)
        print(f"✓ Submission saved successfully as: {unique_filename}")
    elif len(solved_nodes) > 0:
        print("⚠ Solution found but no submission data in scratchpad")
    else:
        print("⚠ No solution found - no submission to save")
    
    print("="*70)
