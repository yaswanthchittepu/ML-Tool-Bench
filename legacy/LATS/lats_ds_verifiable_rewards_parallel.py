"""
Parallelized LATS Data Science Verifiable Rewards Agent for Kaggle Competitions

This module provides a parallelized version of LATS with verifiable rewards system specifically 
designed for data science tasks. It can run multiple independent LATS searches simultaneously, 
each exploring different solution paths using verifiable rewards instead of reflection-based evaluation.

Key Features:
- Multiprocessing-based parallelization for true concurrency
- Verifiable rewards system for objective evaluation
- Outcome-based and step-based reward mechanisms
- Scratchpad-based state management for dataframes, models, etc.
- Tool decorator system for data science functions
- Configurable search parameters and reward systems
- File-based success criteria (submission generation)

Usage:

1. Command line (overrides environment variables):
   python lats_ds_verifiable_rewards_parallel.py --competition spaceship_titanic --trajectories 8 --max_workers 4 --verbose
   python lats_ds_verifiable_rewards_parallel.py -c customer_churn_prediction -t 12 -w 6 --depth 25 --use_feedback --outcome_reward
   python lats_ds_verifiable_rewards_parallel.py --competition bank_marketing_campaign --no_feedback --step_reward --quiet
   
2. Environment Variables (used as defaults):
   export NUM_TRAJECTORIES=4        # Number of parallel LATS searches (default: 4)
   export MAX_WORKERS=4             # Max worker processes (default: cpu_count)
   export MAX_DEPTH=50              # Maximum tree depth (default: 50)
   export COMPETITION_NAME=spaceship_titanic  # Competition to solve
   export USE_FEEDBACK=true         # Whether to use feedback messages
   export MCTS_OUTCOME_REWARD=false # Whether to use outcome-based rewards
   
3. Direct execution:
   python lats_ds_verifiable_rewards_parallel.py

Output:
- Each successful trajectory generates submission files with verifiable rewards naming
- Success is determined by whether the expected submission file was created
- Best solution selection based on cumulative reward scores
"""

from copy import deepcopy
import math
import uuid
import sys, os, json
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import argparse
import pickle

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
from func_to_tool_decorator_lats_verifiable import make_get_and_set_tool, make_get_tool, make_set_tool, override_tool
from challenges import CHALLENGES

from cls import Reflection, Node, TreeState, TreeState_Verifiable_Rewards, Node_Verifiable_Rewards, verifiable_rewards, verifiable_rewards_outcome_based

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

import ipdb
from dotenv import load_dotenv
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

load_dotenv()


# class AgentState(TypedDict):
#     messages: Annotated[Sequence[BaseMessage], add_messages]
#     scratchpad: NotRequired[Dict[str, Any]]
#     plan: NotRequired[str]
#     summary: NotRequired[RunningSummary]
#     num_tool_calls: NotRequired[int]

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



def setup_ds_verifiable_rewards_components(competition_name: str, use_feedback: str, outcome_reward: str):
    """Set up all DS LATS verifiable rewards components for a single search trajectory."""
    from dotenv import load_dotenv
    load_dotenv()

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

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a Data Scientist tasked with solving the Kaggle competition provided below, with the tools available to you. Propose tool candidates that would help solving the problem at the current stage. Your Job is to only call tool candidates at the current step. Any interaction apart from this is not allowed.",
            ),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="messages", optional=True),
        ]
    )

    initial_answer_chain = prompt_template | llm.bind_tools(tools=tools).with_config(
        run_name="GenerateInitialCandidate"
    )

    parser = JsonOutputToolsParser(return_id=True)

    expansion_chain = prompt_template | llm.bind_tools(tools=tools).with_config(
        run_name="GenerateCandidates"
    )
    
    # Create submissions directory
    if(use_feedback in ['true', 'True']):
        if(outcome_reward in ['true', 'True']):
            save_directory = os.path.join(os.environ['SUBMISSIONS_DIR_PREFIX'], f"submissions/{competition_name}/lats_outcome_rewards_feedback")
        else:
            save_directory = os.path.join(os.environ['SUBMISSIONS_DIR_PREFIX'], f"submissions/{competition_name}/lats_shaped_rewards_feedback")
    else:
        if(outcome_reward in ['true', 'True']):
            save_directory = os.path.join(os.environ['SUBMISSIONS_DIR_PREFIX'], f"submissions/{competition_name}/lats_outcome_rewards_no_feedback")
        else:
            save_directory = os.path.join(os.environ['SUBMISSIONS_DIR_PREFIX'], f"submissions/{competition_name}/lats_shaped_rewards_no_feedback")
    os.makedirs(save_directory, exist_ok=True)
    os.makedirs(f"model_saves/{competition_name}/", exist_ok=True)

    return tools, tool_node, llm, initial_answer_chain, parser, expansion_chain, all_decorated_functions


def run_single_ds_verifiable_rewards_search(trajectory_id: int, competition_name: str, max_depth: int, use_feedback: str, outcome_reward: str, problem_type: str, verbose: bool = False):
    """
    Run a single DS LATS verifiable rewards search for the given competition.
    
    Args:
        trajectory_id: Unique identifier for this trajectory
        competition_name: Name of the competition to solve
        max_depth: Maximum depth of the search tree
        verbose: Whether to print detailed progress information
    
    Returns:
        Dictionary containing search results including trajectory results and rewards
    """
    
    # Generate unique ID for this search
    search_id = f"{trajectory_id}_{uuid.uuid4()}"
    interaction_info = defaultdict(int)
    worst_case_per_query_info = defaultdict(int)

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
    
    try:
        if verbose:
            print(f"\n" + "="*80)
            print(f"STARTING DS VERIFIABLE REWARDS LATS SEARCH {trajectory_id}")
            print("="*80)
            print("", flush=True)
        
        # Setup DS LATS components
        tools, tool_node, llm, initial_answer_chain, parser, expansion_chain, all_decorated_functions = setup_ds_verifiable_rewards_components(competition_name, use_feedback, outcome_reward)
        
        ### IN our case, this has to be a pass through function
        def generate_initial_response(state: TreeState_Verifiable_Rewards) -> dict:
            return state

        def generate_candidates(messages: ChatPromptValue, config: RunnableConfig):
            n = config["configurable"].get("N", 5)
            bound_kwargs = llm.bind_tools(tools=tools).kwargs
            chat_result = llm.generate(
                [messages.to_messages()],
                n=1,
                callbacks=config["callbacks"],
                run_name="GenerateCandidates",
                **bound_kwargs,
            )
            return [gen.message for gen in chat_result.generations[0]]

        def try_expansion_with_reducing_context(state, messages, config):
            """Try expansion with progressively reducing context on errors."""
            # Calculate dynamic context sizes
            msg_len = len(messages)
            nonlocal worst_case_per_query_info
            if msg_len > 0:
                msg_lengths = [len(msg.content) if hasattr(msg, 'content') and msg.content else 0 for msg in messages]
                total_chars = sum(msg_lengths)
                avg_length = total_chars / msg_len
                max_length = max(msg_lengths)
                min_length = min(msg_lengths)
                
                if verbose:
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
                        if verbose:
                            print(f"Attempt {i+1}: Using full context ({len(messages)} messages)")
                    else:
                        # Use last N messages
                        context_messages = messages[-context_size:] if len(messages) > context_size else messages
                        if verbose:
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
                    
                    if verbose:
                        print(f"Success on attempt {i+1}")
                    return new_candidates, info
                    
                except Exception as e:
                    if verbose:
                        print(f"Attempt {i+1} failed: {e}")
                    if i == len(context_sizes) - 1:  # Last attempt
                        if verbose:
                            print("All attempts failed, re-raising last exception")
                        raise e
                    else:
                        if verbose:
                            print(f"Retrying with smaller context...")
                        continue

        # This is the select function for the tree
        def select(root: Node_Verifiable_Rewards) -> Node_Verifiable_Rewards:
            """Starting from the root node a child node is selected at each tree level until a leaf node is reached."""

            if not root.children:
                return root

            node = root
            while node.children:
                max_child = max(node.children, key=lambda child: child.upper_confidence_bound(exploration_weight=int(os.environ.get('EXPLORATION_WEIGHT', '1'))))
                node = max_child

            return node

        def expand(state: TreeState_Verifiable_Rewards, config: RunnableConfig) -> dict:
            """Starting from the "best" node in the tree, generate N candidates for the next step."""
            
            nonlocal interaction_info, all_decorated_functions

            root = state["root"]
            best_candidate: Node_Verifiable_Rewards = select(root)
            # Combine all messages from root to best_candidate
            messages = best_candidate.get_trajectory()
            # Combine all scratchpads from root to best_candidate
            scratchpad = best_candidate.get_trajectory_scratchpad()
            # Accrue rewards along trajectory
            traj_rewards = best_candidate.get_trajectory_reward()
            # Generate N candidates from the single child candidate
            
            # Usage in your expand function:
            try:
                new_candidates, info = try_expansion_with_reducing_context(state, messages, config)
                interaction_info['total_tokens'] += info['total_tokens']
                interaction_info['prompt_tokens'] += info['prompt_tokens']
                interaction_info['completion_tokens'] += info['completion_tokens']
                interaction_info['total_cost'] += info['total_cost']
            except Exception as e:
                if verbose:
                    print(f"All expansion attempts failed: {e}")
                raise e
            
            # Need to make input to parser a list and take the first element
            if('llama' in os.environ['OPENAI_MODEL']):
                new_candidates = parse_llama_json({'messages': [new_candidates]})['messages'][0]
            parsed = parser.batch([new_candidates])
            if verbose:
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
                        {   
                            **state,
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
            new_node_scratchpads = defaultdict(dict)
            collected_responses = defaultdict(list)
            for i, result in tool_responses:
                # Skip if this is an error
                if isinstance(result, dict) and any(
                    isinstance(m, ToolMessage) and getattr(m, "status", None) == "error"
                    for m in result.get("messages", [])
                ):
                    collected_responses[i].append(result["messages"][0])
                    if verbose:
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

            # Define our own verifieable rewards here
            # output message is a list of AIMessages, ToolMessages
            # contains info about what tool was called woth what args, and then the response from the tool
            if(outcome_reward in ['true', 'True']):
                all_candidate_rewards, all_candidate_human_feedback_msgs = verifiable_rewards_outcome_based(output_messages, traj_rewards, new_node_scratchpads, tool_docs, problem_type=problem_type)
            else:
                all_candidate_rewards, all_candidate_human_feedback_msgs = verifiable_rewards(output_messages, traj_rewards, new_node_scratchpads, tool_docs, problem_type=problem_type)
            
            # Grow tree
            if(use_feedback in ['true', 'True']):
                child_nodes = [
                    Node_Verifiable_Rewards(cand+[feedback_msg], parent=best_candidate, reward=reward, scratchpad=new_node_scratchpads.get(i, {}))
                    for i, (cand, reward, feedback_msg) in enumerate(zip(output_messages, all_candidate_rewards, all_candidate_human_feedback_msgs))
                ]
                if verbose:
                    print(f"Using rewards with feedback")
            else:
                child_nodes = [
                    Node_Verifiable_Rewards(cand, parent=best_candidate, reward=reward, scratchpad=new_node_scratchpads.get(i, {}))
                    for i, (cand, reward) in enumerate(zip(output_messages, all_candidate_rewards))
                ]
                if verbose:
                    print(f"Using rewards without feedback")
            best_candidate.children.extend(child_nodes)
            # We have already extended the tree directly, so we just return the state
            return state

        def should_loop(state: TreeState_Verifiable_Rewards):
            """Determine whether to continue the tree search."""
            root = state["root"]
            if root.is_solved:
                return END
            if root.height > max_depth:
                return END
            return "expand"

        builder = StateGraph(TreeState_Verifiable_Rewards)
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

        mid = search_id

        if(use_feedback in ['true', 'True']):
            if(outcome_reward in ['true', 'True']):
                save_directory = os.path.join(os.environ['SUBMISSIONS_DIR_PREFIX'], f"submissions/{competition_name}/lats_outcome_rewards_feedback")
                question = CHALLENGES[competition_name].format(save_directory=save_directory, save_file_name=f"lats_outcome_rewards_feedback_agent_{mid}.csv")
            else:
                save_directory = os.path.join(os.environ['SUBMISSIONS_DIR_PREFIX'],f"submissions/{competition_name}/lats_shaped_rewards_feedback")
                question = CHALLENGES[competition_name].format(save_directory=save_directory, save_file_name=f"lats_shaped_rewards_feedback_agent_{mid}.csv")
        else:
            if(outcome_reward in ['true', 'True']):
                save_directory = os.path.join(os.environ['SUBMISSIONS_DIR_PREFIX'], f"submissions/{competition_name}/lats_outcome_rewards_no_feedback")
                question = CHALLENGES[competition_name].format(save_directory=save_directory, save_file_name=f"lats_outcome_rewards_no_feedback_agent_{mid}.csv")
            else:
                save_directory = os.path.join(os.environ['SUBMISSIONS_DIR_PREFIX'], f"submissions/{competition_name}/lats_shaped_rewards_no_feedback")
                question = CHALLENGES[competition_name].format(save_directory=save_directory, save_file_name=f"lats_shaped_rewards_no_feedback_agent_{mid}.csv")

        # if(os.environ['USE_FEEDBACK'] in ['true', 'True']):
        #     if(os.environ['MCTS_OUTCOME_REWARD'] in ['true', 'True']):
        #         save_directory = f"submissions/{os.environ['COMPETITION_NAME']}/lats_verifiable_rewards"
        #         question = CHALLENGES[competition_name].format(save_directory=save_directory, save_file_name=f"lats_outcome_rewards_agent_{mid}.csv")
        #     else:
        #         question = CHALLENGES[competition_name].format(save_directory=save_directory, save_file_name=f"lats_verifiable_rewards_agent_{mid}.csv")
        # else:
        #     if(os.environ['MCTS_OUTCOME_REWARD'] in ['true', 'True']):
        #         question = CHALLENGES[competition_name].format(save_directory=save_directory, save_file_name=f"lats_outcome_rewards_no_feedback_agent_{mid}.csv")
        #     else:
        #         question = CHALLENGES[competition_name].format(save_directory=save_directory, save_file_name=f"lats_verifiable_rewards_no_feedback_agent_{mid}.csv")

        start_time = time.time()
        last_step = None
        tree_node = {'input': question}
        root_node = Node_Verifiable_Rewards(messages=[], tree_node=tree_node)
        tree_node['root'] = root_node
        
        for step in graph.stream(tree_node, config={"recursion_limit": 750}):
            last_step = step
            step_name, step_state = next(iter(step.items()))
            if verbose:
                print(step_name)
                print("rolled out: ", step_state["root"].height)
                print("---")

        end_time = time.time()
        total_time = end_time - start_time
        
        # Process results
        solved_nodes = root_node.get_solved_nodes()
        
        # Get best node - either solution or highest valued leaf if no solution found
        if(len(solved_nodes) > 0):
            # Select best node: (has_dataframe) * trajectory_reward
            best_node = max(solved_nodes, key=lambda node: (
                int(any(hasattr(v, 'shape') and hasattr(v, 'columns') for v in node.scratchpad.values()) if node.scratchpad else False) * 
                sum(node.get_trajectory_reward().values())
            ))
            node_type = "SOLUTION"
            if verbose:
                print(f"✓ Best solution found with reward: {sum(best_node.get_trajectory_reward().values())}")
        else:
            # No solution found, get highest valued leaf node
            def get_best_leaf_node_verifiable(root):
                """Get the leaf node with highest total reward, regardless of solution status"""
                all_nodes = [root] + root._get_all_children()
                leaf_nodes = [node for node in all_nodes if node.is_terminal]
                if not leaf_nodes:
                    # If no leaf nodes, return the highest valued node
                    leaf_nodes = all_nodes
                return max(leaf_nodes, key=lambda node: extract_score_from_reward(node.get_trajectory_reward()))
            
            best_node = get_best_leaf_node_verifiable(root_node)
            node_type = "BEST_ATTEMPT"
            if verbose:
                print(f"⚠ No solution found, using best attempt node with total reward: {extract_score_from_reward(best_node.get_trajectory_reward()):.3f}")
        
        # Save trajectory by traversing from best node back to root
        print("\n" + "="*70)
        print(f"SAVING LATS VERIFIABLE REWARDS TRAJECTORY FROM {node_type} TO ROOT")
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
        
        print(f"✓ LATS Verifiable Rewards trajectory saved to: {trajectory_file}")
        print(f"✓ Trajectory contains {trajectory_data['total_steps']} steps from {node_type.lower()} back to root")
        print(f"✓ Best node depth: {best_node.depth}")
        print(f"✓ Best node total reward: {extract_score_from_reward(best_node.get_trajectory_reward()):.3f}")
        print(f"✓ Tree search height: {root_node.height}")
        print(f"✓ Reward type: {reward_type}, Feedback: {feedback_enabled}")
        print(f"✓ Node type: {node_type}")
        
        # Only save submission if we actually found a solution
        if len(solved_nodes) > 0 and best_node.scratchpad:
            for v in best_node.scratchpad.values():
                submission_df = v
                # Generate unique filename with node ID
            if(use_feedback in ['true', 'True']):
                if(outcome_reward in ['true', 'True']):
                    unique_filename = os.path.join(save_directory, f"lats_outcome_rewards_feedback_agent_manual_{mid}.csv")
                else:
                    unique_filename = os.path.join(save_directory, f"lats_shaped_rewards_feedback_agent_manual_{mid}.csv")
            else:
                if(outcome_reward in ['true', 'True']):
                    unique_filename = os.path.join(save_directory, f"lats_outcome_rewards_no_feedback_agent_manual_{mid}.csv")
                else:
                    unique_filename = os.path.join(save_directory, f"lats_shaped_rewards_no_feedback_agent_manual_{mid}.csv")
            ut.save_dataframe_to_csv(submission_df, unique_filename)
            if verbose:
                print(f"✓ Submission saved successfully as: {unique_filename}")
            
            file_exists = os.path.exists(unique_filename)
            
            result = {
                'trajectory_id': trajectory_id,
                'search_id': search_id,
                'execution_time': end_time - start_time,
                'best_reward': extract_score_from_reward(best_node.get_trajectory_reward()),
                'tree_height': root_node.height,
                'solved_nodes_count': len(solved_nodes),
                'success': file_exists,
                'error': None if file_exists else f"Output file not found: {unique_filename}",
                'output_file': unique_filename if file_exists else None,
                'interaction_info': interaction_info,
                'worst_case_per_query_info': worst_case_per_query_info,
            }
        elif len(solved_nodes) > 0:
            if verbose:
                print("⚠ Solution found but no submission data in scratchpad")
            result = {
                'trajectory_id': trajectory_id,
                'search_id': search_id,
                'execution_time': end_time - start_time,
                'best_reward': extract_score_from_reward(best_node.get_trajectory_reward()),
                'tree_height': root_node.height,
                'solved_nodes_count': len(solved_nodes),
                'success': False,
                'error': "Solution found but no submission data in scratchpad",
                'output_file': None,
                'interaction_info': interaction_info,
                'worst_case_per_query_info': worst_case_per_query_info,
            }
        else:
            if verbose:
                print("⚠ No solution found - no submission to save")
            result = {
                'trajectory_id': trajectory_id,
                'search_id': search_id,
                'execution_time': end_time - start_time,
                'best_reward': extract_score_from_reward(best_node.get_trajectory_reward()),
                'tree_height': root_node.height,
                'solved_nodes_count': 0,
                'success': False,
                'error': "No solution found",
                'output_file': None,
                'interaction_info': interaction_info,
                'worst_case_per_query_info': worst_case_per_query_info,
            }
        
        if verbose:
            status_symbol = "✓" if result['success'] else "⚠"
            if result['success']:
                status_msg = "found solution and created file"
            else:
                status_msg = f"completed with {node_type.lower()} (no solution found)"
            print(f"{status_symbol} DS Verifiable Rewards LATS Search {trajectory_id} {status_msg} in {result['execution_time']:.2f}s")
            print(f"Tree height: {result['tree_height']}, Solved nodes: {result['solved_nodes_count']}, Best reward: {result['best_reward']}, Node type: {node_type}")
            if result['output_file']:
                print(f"Output file: {result['output_file']}")
            print(f"\n\n Trajectory ID: {trajectory_id}, Worst case per query information: {result['worst_case_per_query_info']}, Total Cost: {result['interaction_info']['total_cost']} \n\n")
            print(f"Interaction information: {result['interaction_info']}")
        
        return result
        
    except Exception as e:
        result = {
            'trajectory_id': trajectory_id,
            'search_id': search_id,
            'success': False,
            'error': str(e)
        }
        
        if verbose:
            print(f"✗ DS Verifiable Rewards LATS Search {trajectory_id} failed: {e}")
        
        return result


def run_parallel_ds_verifiable_rewards_searches(competition_name: str, num_trajectories: int, max_workers: int, max_depth: int, use_feedback: str, outcome_reward: str, problem_type: str, verbose: bool = True):
    """
    Run multiple DS LATS verifiable rewards searches in parallel.
    """
    
    print(f"\n" + "="*80)
    print(f"RUNNING {num_trajectories} PARALLEL DS VERIFIABLE REWARDS LATS SEARCHES")
    print(f"Max workers: {max_workers}")
    print(f"Competition: {competition_name}")
    print(f"Max depth: {max_depth}")
    print("="*80, flush=True)
    
    start_time = time.time()
    results = []
    
    # Use ProcessPoolExecutor for true parallelism
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all search trajectories
        future_to_trajectory = {
            executor.submit(run_single_ds_verifiable_rewards_search, i, competition_name, max_depth, use_feedback, outcome_reward, problem_type, verbose): i 
            for i in range(num_trajectories)
        }
        
        print(f"All {num_trajectories} DS Verifiable Rewards LATS searches submitted to worker pool. Waiting for completion...")
        print("", flush=True)
        
        # Collect results as they complete
        completed_count = 0
        for future in as_completed(future_to_trajectory):
            trajectory_id = future_to_trajectory[future]
            completed_count += 1
            
            try:
                result = future.result()
                results.append(result)
                
                if verbose:
                    if result['success']:
                        print(f"✓ DS Verifiable Rewards Search {trajectory_id} FOUND SOLUTION - reward: {result.get('best_reward', 0)} ({completed_count}/{num_trajectories})")
                    else:
                        print(f"✗ DS Verifiable Rewards Search {trajectory_id} NO SOLUTION FOUND ({completed_count}/{num_trajectories})")
                    print("", flush=True)
                    
            except Exception as e:
                print(f"✗ DS Verifiable Rewards Search {trajectory_id} generated an exception: {e} ({completed_count}/{num_trajectories})")
                print("", flush=True)
                results.append({
                    'trajectory_id': trajectory_id,
                    'success': False,
                    'error': str(e)
                })
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Summary statistics
    successful_searches = [r for r in results if r['success']]
    failed_searches = [r for r in results if not r['success']]
    
    print(f"\n" + "="*80)
    print("PARALLEL DS VERIFIABLE REWARDS LATS EXECUTION SUMMARY")
    print("="*80)
    print(f"Total searches: {num_trajectories}")
    print(f"Successful: {len(successful_searches)}")
    print(f"Failed: {len(failed_searches)}")
    print(f"Total execution time: {total_time:.2f}s")
    
    # if successful_searches:
    #     avg_time = sum(r['execution_time'] for r in successful_searches) / len(successful_searches)
    #     avg_reward = sum(r['best_reward'] for r in successful_searches) / len(successful_searches)
    #     best_result = max(successful_searches, key=lambda x: x['best_reward'])
        
    #     print(f"Average search time: {avg_time:.2f}s")
    #     print(f"Average best reward: {avg_reward:.2f}")
    #     print(f"Best reward: {best_result['best_reward']} (Search {best_result['trajectory_id']})")
    #     print(f"Speedup factor: {avg_time * num_trajectories / total_time:.2f}x")
        
    #     print(f"\nBest solution found:")
    #     print(f"  Search ID: {best_result['trajectory_id']}")
    #     print(f"  Reward: {best_result['best_reward']}")
    #     print(f"  Tree height: {best_result['tree_height']}")
    #     print(f"  Solved nodes: {best_result['solved_nodes_count']}")
    #     print(f"  Output file: {best_result['output_file']}")
    
    # # Print failed searches
    # if failed_searches:
    #     print(f"\nFailed searches:")
    #     for result in failed_searches:
    #         print(f"  - Search {result['trajectory_id']}: {result['error']}")
    
    print(f"\n" + "="*80)
    print("PARALLEL DS VERIFIABLE REWARDS LATS EXECUTION COMPLETED")
    print("="*80)
    print("", flush=True)
    
    return results


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run parallel DS LATS verifiable rewards searches for Kaggle competitions')
    parser.add_argument('--competition', '-c', type=str, 
                       help='Competition name (overrides COMPETITION_NAME env var)')
    parser.add_argument('--trajectories', '-t', type=int,
                       help='Number of parallel DS LATS searches (overrides NUM_TRAJECTORIES env var)')
    parser.add_argument('--max_workers', '-w', type=int,
                       help='Maximum number of worker processes (overrides MAX_WORKERS env var)')
    parser.add_argument('--problem_type', '-p', type=str, choices=['binary-acc', 'binary-probs', 'regression', 'multiclass'],
                       help='Kaggle Challenge Problem type (overrides PROBLEM_TYPE env var)')
    parser.add_argument('--depth', '-d', type=int,
                       help='Maximum depth of search tree (overrides MAX_DEPTH env var)')
    parser.add_argument('--use_feedback', action='store_true',
                       help='Enable feedback messages (overrides USE_FEEDBACK env var)')
    parser.add_argument('--no_feedback', action='store_true',
                       help='Disable feedback messages (overrides USE_FEEDBACK env var)')
    parser.add_argument('--outcome_reward', action='store_true',
                       help='Enable outcome-based rewards (overrides MCTS_OUTCOME_REWARD env var)')
    parser.add_argument('--stage_reward', action='store_true',
                       help='Enable step-based rewards (overrides MCTS_OUTCOME_REWARD env var)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--quiet', action='store_true',
                       help='Disable verbose output')
    
    args = parser.parse_args()
    
    # Configuration - command line args override environment variables
    NUM_TRAJECTORIES = args.trajectories if args.trajectories is not None else int(os.environ.get('NUM_TRAJECTORIES', 4))
    MAX_WORKERS = args.max_workers if args.max_workers is not None else int(os.environ.get('MAX_WORKERS', mp.cpu_count()))
    MAX_DEPTH = args.depth if args.depth is not None else int(os.environ.get('MAX_DEPTH', 50))
    COMPETITION_NAME = args.competition if args.competition is not None else os.environ['COMPETITION_NAME']
    PROBLEM_TYPE = args.problem_type if args.problem_type is not None else os.environ['PROBLEM_TYPE']

    # Handle feedback flag
    if args.use_feedback:
        USE_FEEDBACK = 'true'
        feedback_source = 'CLI'
    elif args.no_feedback:
        USE_FEEDBACK = 'false'
        feedback_source = 'CLI'
    else:
        USE_FEEDBACK = os.environ.get('USE_FEEDBACK', 'true')
        feedback_source = 'ENV/DEFAULT'
    
    # Handle reward type flag
    if args.outcome_reward:
        MCTS_OUTCOME_REWARD = 'true'
        reward_source = 'CLI'
    elif args.stage_reward:
        MCTS_OUTCOME_REWARD = 'false'
        reward_source = 'CLI'
    else:
        MCTS_OUTCOME_REWARD = os.environ.get('MCTS_OUTCOME_REWARD', 'false')
        reward_source = 'ENV/DEFAULT'
    
    # Set environment variables for the worker processes
    os.environ['USE_FEEDBACK'] = USE_FEEDBACK
    os.environ['MCTS_OUTCOME_REWARD'] = MCTS_OUTCOME_REWARD
    os.environ['PROBLEM_TYPE'] = PROBLEM_TYPE

    # Handle verbose flag
    if args.quiet:
        VERBOSE = False
    elif args.verbose:
        VERBOSE = True
    else:
        VERBOSE = os.environ.get('VERBOSE', 'true').lower() == 'true'
    
    # Show configuration
    print(f"DS Verifiable Rewards LATS Configuration:")
    print(f"  COMPETITION_NAME: {COMPETITION_NAME} ({'CLI' if args.competition is not None else 'ENV/DEFAULT'})")
    print(f"  NUM_TRAJECTORIES: {NUM_TRAJECTORIES} ({'CLI' if args.trajectories is not None else 'ENV/DEFAULT'})")
    print(f"  MAX_WORKERS: {MAX_WORKERS} ({'CLI' if args.max_workers is not None else 'ENV/DEFAULT'})")
    print(f"  PROBLEM_TYPE: {PROBLEM_TYPE} ({'CLI' if args.problem_type is not None else 'ENV/DEFAULT'})")
    print(f"  MAX_DEPTH: {MAX_DEPTH} ({'CLI' if args.depth is not None else 'ENV/DEFAULT'})")
    print(f"  USE_FEEDBACK: {USE_FEEDBACK} ({feedback_source})")
    print(f"  MCTS_OUTCOME_REWARD: {MCTS_OUTCOME_REWARD} ({reward_source})")
    print(f"  VERBOSE: {VERBOSE} ({'CLI' if args.verbose or args.quiet else 'ENV/DEFAULT'})")
    print("", flush=True)
    
    # Run parallel DS LATS searches
    results = run_parallel_ds_verifiable_rewards_searches(
        competition_name=COMPETITION_NAME,
        num_trajectories=NUM_TRAJECTORIES,
        max_workers=MAX_WORKERS,
        max_depth=MAX_DEPTH,
        use_feedback=USE_FEEDBACK,
        outcome_reward=MCTS_OUTCOME_REWARD,
        problem_type=PROBLEM_TYPE,
        verbose=VERBOSE
    )
    
    # Save results to pickle file
    feedback_str = "feedback" if USE_FEEDBACK.lower() == 'true' else "no_feedback"
    reward_str = "outcome" if MCTS_OUTCOME_REWARD.lower() == 'true' else "shaped"
    log_directory = os.path.join(os.environ['SUBMISSIONS_DIR_PREFIX'], f"logs/{COMPETITION_NAME}/lats_{reward_str}_rewards_{feedback_str}")
    os.makedirs(log_directory, exist_ok=True)
    with open(os.path.join(log_directory, f"lats_{reward_str}_rewards_{feedback_str}_{NUM_TRAJECTORIES}_results.pkl"), 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to pickle file")
    
    print(f"\nParallel DS Verifiable Rewards LATS execution completed with {len(results)} searches.")