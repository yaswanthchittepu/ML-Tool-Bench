"""
Parallelized LATS Data Science Reflection Agent for Kaggle Competitions

This module provides a parallelized version of LATS specifically designed for data science tasks.
It can run multiple independent LATS searches simultaneously, each exploring different solution
paths for Kaggle competitions using data science tools and maintaining scratchpad state.

Key Features:
- Multiprocessing-based parallelization for true concurrency
- Monte Carlo Tree Search with data science tool integration
- Scratchpad-based state management for dataframes, models, etc.
- Reflection-based trajectory evaluation for ML tasks
- Tool decorator system for data science functions
- Configurable search parameters and competition selection
- File-based success criteria (submission generation)

Usage:

1. Command line (overrides environment variables):
   python lats_ds_reflection_parallel.py --competition spaceship_titanic --trajectories 8 --max_workers 4 --verbose
   python lats_ds_reflection_parallel.py -c customer_churn_prediction -t 12 -w 6 --depth 10
   
2. Environment Variables (used as defaults):
   export NUM_TRAJECTORIES=4        # Number of parallel LATS searches (default: 4)
   export MAX_WORKERS=4             # Max worker processes (default: cpu_count)
   export MAX_DEPTH=30              # Maximum tree depth (default: 30)
   export COMPETITION_NAME=spaceship_titanic  # Competition to solve
   
3. Direct execution:
   python lats_ds_reflection_parallel.py

4. Programmatic usage:
   from lats_ds_reflection_parallel import run_parallel_ds_lats_searches
   
   results = run_parallel_ds_lats_searches(
       competition_name='spaceship_titanic',
       num_trajectories=8,
       max_workers=4,
       verbose=True
   )

Output:
- Each successful trajectory generates a unique CSV file: lats_reflection_agent_{trajectory_id}_{uuid}.csv
- Success is determined by whether the expected submission file was created
- Comparison of solution quality and tree exploration across searches
- Best overall solution and trajectory analysis

Performance:
- Scales roughly linearly with available CPU cores
- Each LATS search is completely independent with its own tool state
- Memory usage scales with number of parallel processes and tree complexity
"""

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
from func_to_tool_decorator_lats import make_get_and_set_tool, make_get_tool, make_set_tool, override_tool
from challenges import CHALLENGES

from cls import Reflection, Node, TreeState

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
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()


# class AgentState(TypedDict):
#     messages: Annotated[Sequence[BaseMessage], add_messages]
#     scratchpad: NotRequired[Dict[str, Any]]
#     plan: NotRequired[str]
#     summary: NotRequired[RunningSummary]
#     num_tool_calls: NotRequired[int]

def extract_score_from_reflection(content):
    """
    Extract numeric score from reflection content.
    Handles patterns like 'Score: 3', 'Score: 7.5', etc.
    """
    import re
    
    # Look for "Score: X" pattern (case insensitive)
    score_match = re.search(r'score:\s*(\d+(?:\.\d+)?)', content, re.IGNORECASE)
    
    if score_match:
        try:
            return float(score_match.group(1))
        except ValueError:
            return None
    
    # Fallback: look for just a number at the end after newline
    fallback_match = re.search(r'\n(\d+(?:\.\d+)?)$', content.strip())
    if fallback_match:
        try:
            return float(fallback_match.group(1))
        except ValueError:
            return None
    
    return None

def save_lats_trajectory(solution_node, search_stats, save_directory, agent_id):
    """
    Save LATS trajectory by traversing from solution node back to root.
    This captures the complete search path that led to the best solution.
    """
    trajectory_data = {
        'agent_type': 'LATS',
        'agent_id': agent_id,
        'timestamp': datetime.now().isoformat(),
        'search_statistics': search_stats,
        'solution_node_info': {
            'depth': solution_node.depth,
            'visits': solution_node.visits,
            'value': solution_node.value,
            'is_solved': solution_node.is_solved
        }
    }
    
    # Get the complete trajectory from solution back to root
    trajectory_messages = solution_node.get_trajectory(include_reflections=True)
    
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
                # Extract numeric score from reflection content
                extracted_score = extract_score_from_reflection(message.content)
                
                step_info.update({
                    'step_type': 'reflection',
                    'action': 'llm_reflection',
                    'content_preview': message.content[:300] + "..." if len(message.content) > 300 else message.content,
                    'content_length': len(message.content),
                    'extracted_score': extracted_score if extracted_score is not None else None,
                    'full_reflection_content': message.content
                })
        
        trajectory_steps.append(step_info)
        step_number += 1
    
    trajectory_data['trajectory_steps'] = trajectory_steps
    trajectory_data['total_steps'] = len(trajectory_steps)
    
    # Save to file
    trajectory_file = os.path.join(save_directory, f"lats_trajectory_{agent_id}.json")
    with open(trajectory_file, 'w') as f:
        json.dump(trajectory_data, f, indent=2)
    
    return trajectory_file, trajectory_data


def setup_ds_lats_components(competition_name: str):
    """Set up all DS LATS components for a single search trajectory."""
    from dotenv import load_dotenv
    load_dotenv()
    
    # Load the decorator mapping from JSON file
    with open("function_to_decorator_map.json", "r") as f:
        decorator_map = json.load(f)
    
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

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a Data Science judge, who evaluates the goodness of tool calling trajectories to solve Machine Learning tasks on Kaggle. Reflect and grade the agents trajectory plan for the provided challenge. The trajectories should be aimed towards solving the challenge i.e generating a trained model and a valid submission file. Keep your reflections concise and to the point.",
            ),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="candidate"),
        ]
    )

    reflection_llm_chain = (
        prompt
        | llm.bind_tools(tools=[Reflection], tool_choice="Reflection").with_config(
            run_name="Reflection"
        )
        | PydanticToolsParser(tools=[Reflection])
    )

    @as_runnable
    def reflection_chain(inputs) -> Reflection:
        tool_choices = reflection_llm_chain.invoke(inputs)
        reflection = tool_choices[0]
        if not isinstance(inputs["candidate"][-1], AIMessage):
            reflection.found_solution = False
        return reflection

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a Data Scientist tasked with solving the Kaggle competition provided below, with the tools available to you. Propose tool candidates that would help solving the problem at the current stage.",
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
    save_directory = os.path.join(os.environ['SUBMISSIONS_DIR_PREFIX'], f"submissions/{competition_name}/lats_reflection")
    os.makedirs(save_directory, exist_ok=True)
    os.makedirs(f"model_saves/{competition_name}/", exist_ok=True)
    
    return tools, tool_node, llm, reflection_chain, initial_answer_chain, parser, expansion_chain


def run_single_ds_lats_search(trajectory_id: int, competition_name: str, max_depth: int = 30, verbose: bool = False):
    """
    Run a single DS LATS search for the given competition.
    
    Args:
        trajectory_id: Unique identifier for this trajectory
        competition_name: Name of the competition to solve
        max_depth: Maximum depth of the search tree
        verbose: Whether to print detailed progress information
    
    Returns:
        Dictionary containing search results including:
        - trajectory_id: ID of the trajectory
        - search_id: Unique search ID
        - execution_time: Time taken to complete the search
        - final_solution: Best solution trajectory found
        - tree_height: Final height of the search tree
        - total_nodes: Total number of nodes explored
        - solution_score: Quality score of the final solution
        - found_solution: Whether a validated solution was found
        - success: Whether the expected submission file was created
        - error: Error message if search failed
        - output_file: Path to the created output file (if successful)
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
            print(f"STARTING DS LATS SEARCH {trajectory_id}")
            print("="*80)
            print("", flush=True)
        
        # Setup DS LATS components
        tools, tool_node, llm, reflection_chain, initial_answer_chain, parser, expansion_chain = setup_ds_lats_components(competition_name)
        
        # Generate competition question with unique ID
        mid = search_id
        save_directory = os.path.join(os.environ['SUBMISSIONS_DIR_PREFIX'], f"submissions/{competition_name}/lats_reflection")
        question = CHALLENGES[competition_name].format(save_directory=save_directory, save_file_name=f"lats_reflection_agent_{mid}.csv")
        
        def generate_initial_response(state: TreeState) -> dict:
            """Generate the initial candidate response - pass through for DS LATS."""
            return state
        
        def select(root: Node) -> Node:
            """Starting from the root node a child node is selected at each tree level until a leaf node is reached."""
            if not root.children:
                return root

            node = root
            while node.children:
                max_child = max(node.children, key=lambda child: child.upper_confidence_bound(exploration_weight=int(os.environ.get('EXPLORATION_WEIGHT', '1'))))
                node = max_child

            return node

        def expand(state: TreeState, config: RunnableConfig) -> dict:
            """Starting from the "best" node in the tree, generate N candidates for the next step."""
            
            nonlocal interaction_info

            root = state["root"]
            best_candidate: Node = select(root)
            # Combine all messages from root to best_candidate
            messages = best_candidate.get_trajectory()
            # Combine all scratchpads from root to best_candidate
            scratchpad = best_candidate.get_trajectory_scratchpad()
            # Generate N candidates from the single child candidate

            with get_openai_callback() as cb:
                new_candidates = expansion_chain.invoke(
                    {"input": state["input"], "messages": messages}, config
                )
                interaction_info['total_tokens'] += cb.total_tokens
                interaction_info['prompt_tokens'] += cb.prompt_tokens
                interaction_info['completion_tokens'] += cb.completion_tokens
                interaction_info['total_cost'] += cb.total_cost
                if(cb.total_tokens > worst_case_per_query_info['total_tokens']):
                    worst_case_per_query_info['total_tokens'] = cb.total_tokens
                    worst_case_per_query_info['prompt_tokens'] = cb.prompt_tokens
                    worst_case_per_query_info['prompt_tokens_cached'] = cb.prompt_tokens_cached
                    worst_case_per_query_info['completion_tokens'] = cb.completion_tokens
                    worst_case_per_query_info['total_cost'] = cb.total_cost
            
            # Need to make input to parser a list and take the first element
            if('llama' in os.environ['OPENAI_MODEL']):
                new_candidates = parse_llama_json({'messages': [new_candidates]})['messages'][0]
            parsed = parser.batch([new_candidates])
            if verbose:
                print(f"Parsed tool calls: {len(parsed[0]) if parsed else 0}")
            
            # Extract tool calls
            flattened = [
                (i, tool_call)
                for i, tool_call in enumerate(parsed[0])
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
            
            if verbose:
                print(f"Executed {len(tool_responses)} tool calls")
            
            # Process tool responses and update scratchpads
            new_node_scratchpads = {}
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
                    if verbose:
                        warnings.warn(f"Unexpected result type for tool call {i}: {type(result)}")

            def split_ai_message_by_tool_calls(ai_message: AIMessage) -> list[AIMessage]:
                tool_calls = ai_message.tool_calls
                if not tool_calls:
                    return [ai_message]

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
            output_messages = []
            for i, candidate in enumerate(new_candidates_split):
                output_messages.append([candidate] + collected_responses[i])

            # Reflect on each candidate with retry logic
            max_retries = 5
            reflections = None
            
            for attempt in range(max_retries):
                try:
                    with get_openai_callback() as cb:
                        reflections = reflection_chain.batch(
                            [{"input": state["input"], "candidate": messages+msges} for msges in output_messages],
                            config,
                        )
                        interaction_info['total_tokens'] += cb.total_tokens
                        interaction_info['prompt_tokens'] += cb.prompt_tokens
                        interaction_info['completion_tokens'] += cb.completion_tokens
                        interaction_info['total_cost'] += cb.total_cost
                        if(cb.total_tokens > worst_case_per_query_info['total_tokens']):
                            worst_case_per_query_info['total_tokens'] = cb.total_tokens
                            worst_case_per_query_info['prompt_tokens'] = cb.prompt_tokens
                            worst_case_per_query_info['prompt_tokens_cached'] = cb.prompt_tokens_cached
                            worst_case_per_query_info['completion_tokens'] = cb.completion_tokens
                            worst_case_per_query_info['total_cost'] = cb.total_cost
                    break
                except Exception as e:
                    if verbose:
                        print(f"Error in reflection (attempt {attempt + 1}/{max_retries}): {e}")
                    if attempt == max_retries - 1:
                        raise e
                    continue
            
            # Grow tree
            child_nodes = [
                Node(cand, parent=best_candidate, reflection=reflection, scratchpad=new_node_scratchpads.get(i, {}))
                for i, (cand, reflection) in enumerate(zip(output_messages, reflections))
            ]
            best_candidate.children.extend(child_nodes)
            
            return state

        def should_loop(state: TreeState):
            """Determine whether to continue the tree search."""
            root = state["root"]
            if root.is_solved:
                return END
            if root.height > max_depth:
                return END
            return "expand"

        # Build the search graph
        builder = StateGraph(TreeState)
        builder.add_node("start", generate_initial_response)
        builder.add_node("expand", expand)
        builder.add_edge(START, "start")

        builder.add_conditional_edges(
            "start",
            should_loop,
            ["expand", END],
        )
        builder.add_conditional_edges(
            "expand",
            should_loop,
            ["expand", END],
        )

        graph = builder.compile()

        # Run the DS LATS search
        start_time = time.time()
        last_step = None
        tree_node = {'input': question}
        root_node = Node(messages=[], tree_node=tree_node)
        tree_node['root'] = root_node
        
        for step in graph.stream(tree_node, config={"recursion_limit": 500}):
            last_step = step
            step_name, step_state = next(iter(step.items()))
            print(f"Step: {step_name}, Tree height: {step_state['root'].height}")
        
        end_time = time.time()
        total_time = end_time - start_time

        # Extract results
        try:
            root_node = last_step["expand"]["root"]
            solution_node = root_node.get_best_solution()
            best_trajectory = solution_node.get_trajectory(include_reflections=False)

            if solution_node and solution_node.is_solved:
                best_node = solution_node
                node_type = "SOLUTION"
                print(f"✓ Found solution node with value: {best_node.value:.3f}")
            else:
                # No solution found, get highest valued leaf node
                def get_best_leaf_node(root):
                    """Get the leaf node with highest value, regardless of solution status"""
                    all_nodes = [root] + root._get_all_children()
                    leaf_nodes = [node for node in all_nodes if node.is_terminal]
                    if not leaf_nodes:
                        # If no leaf nodes, return the highest valued node
                        leaf_nodes = all_nodes
                    return max(leaf_nodes, key=lambda node: node.value)
                
                best_node = get_best_leaf_node(root_node)
                node_type = "BEST_ATTEMPT"
                print(f"⚠ No solution found, using best attempt node with value: {best_node.value:.3f}")

            # Save trajectory by traversing from best node back to root
            print("\n" + "="*60)
            print(f"SAVING LATS TRAJECTORY FROM {node_type} TO ROOT")
            print("="*60)
            
            search_statistics = {
                'total_search_time': total_time,
                'total_tokens': interaction_info['total_tokens'],
                'total_cost': interaction_info['total_cost'],
                'prompt_tokens': interaction_info['prompt_tokens'], 
                'completion_tokens': interaction_info['completion_tokens'],
                'tree_height': root_node.height,
                'solution_found': solution_node and solution_node.is_solved,
                'node_type': node_type.lower(),
                'competition_name': os.environ.get('COMPETITION_NAME', 'unknown'),
                'model_used': os.environ.get('OPENAI_MODEL', 'unknown')
            }
            
            trajectory_file, trajectory_data = save_lats_trajectory(
                solution_node=best_node,
                search_stats=search_statistics,
                save_directory=save_directory,
                agent_id=str(mid)
            )
            
            print(f"✓ LATS trajectory saved to: {trajectory_file}")
            print(f"✓ Trajectory contains {trajectory_data['total_steps']} steps from {node_type.lower()} back to root")
            print(f"✓ Best node depth: {best_node.depth}")
            print(f"✓ Best node value: {best_node.value:.3f}")
            print(f"✓ Tree search height: {root_node.height}")
            print(f"✓ Node type: {node_type}")
            

            # search_statistics = {
            #     'total_search_time': total_time,
            #     'total_tokens': interaction_info['total_tokens'],
            #     'total_cost': interaction_info['total_cost'],
            #     'prompt_tokens': interaction_info['prompt_tokens'], 
            #     'completion_tokens': interaction_info['completion_tokens'],
            #     'tree_height': last_step["expand"]["root"].height,
            #     'competition_name': os.environ.get('COMPETITION_NAME', 'unknown'),
            #     'model_used': os.environ.get('OPENAI_MODEL', 'unknown')
            # }

            # trajectory_file, trajectory_data = save_lats_trajectory(
            #     solution_node=solution_node,
            #     search_stats=search_statistics,
            #     save_directory=save_directory,
            #     agent_id=str(mid)
            # )

            # Only save submission if we actually found a solution
            if solution_node and solution_node.is_solved and solution_node.scratchpad:
                submission_df = list(solution_node.scratchpad.values())[0]
                save_path = os.path.join(save_directory, f"lats_reflection_agent_manual_{mid}.csv")
                ut.save_dataframe_to_csv(submission_df, save_path)
                print(f"✓ Submission saved successfully as: {save_path}")
            elif solution_node and solution_node.is_solved:
                print("⚠ Solution found but no submission data in scratchpad")
            else:
                print("⚠ No solution found - no submission to save")

            # if(solution_node.scratchpad):
            #     submission_df = list(solution_node.scratchpad.values())[0]
            #     save_path = os.path.join(save_directory, f"lats_reflection_agent_manual_{mid}.csv")
            #     ut.save_dataframe_to_csv(submission_df, save_path)
            #     print(f"Submission saved successfully as: {save_path}")
            # else:
            #     print("No submission data found in best node scratchpad")
            
            # Count total nodes in tree
            def count_nodes(node):
                count = 1
                for child in node.children:
                    count += count_nodes(child)
                return count
            
            total_nodes = count_nodes(root_node)
            
            # Check if the expected output file was created
            expected_file_path = f"submissions/{competition_name}/lats_reflection/lats_reflection_agent_manual_{mid}.csv"
            file_exists = os.path.exists(expected_file_path)
            
            result = {
                'trajectory_id': trajectory_id,
                'search_id': search_id,
                'execution_time': end_time - start_time,
                'final_solution': best_trajectory[-1].content if best_trajectory else None,
                'tree_height': root_node.height,
                'total_nodes': total_nodes,
                'solution_score': solution_node.reflection.score if solution_node.reflection else 0,
                'success': file_exists,
                'error': None if file_exists else f"Expected output file not found: {expected_file_path}",
                'output_file': expected_file_path if file_exists else None,
                'interaction_info': interaction_info,
                'worst_case_per_query_info': worst_case_per_query_info,
            }
        except Exception as e:
            result = {
                'trajectory_id': trajectory_id,
                'search_id': search_id,
                'execution_time': end_time - start_time,
                'final_solution': None,
                'tree_height': 0,
                'total_nodes': 0,
                'solution_score': 0,
                'success': False,
                'error': "No valid search result obtained",
                'output_file': None,
                **interaction_info
            }
        
        if verbose:
            status_symbol = "✓" if result['success'] else "⚠"
            status_msg = "found solution and created file" if result['success'] else "completed but no output file"
            print(f"{status_symbol} DS LATS Search {trajectory_id} {status_msg} in {result['execution_time']:.2f}s")
            print(f"Tree height: {result['tree_height']}, Nodes explored: {result['total_nodes']}, Score: {result['solution_score']}")
            if result['output_file']:
                print(f"Output file: {result['output_file']}")
            print(f"\n\n Trajctory ID: {trajectory_id}, Worst case per query information: {result['worst_case_per_query_info']}, Total Cost: {result['interaction_info']['total_cost']} \n\n")
        
        return result
        
    except Exception as e:
        result = {
            'trajectory_id': trajectory_id,
            'search_id': search_id,
            'success': False,
            'error': str(e)
        }
        
        if verbose:
            print(f"✗ DS LATS Search {trajectory_id} failed: {e}")
        
        return result


def run_parallel_ds_lats_searches(competition_name: str, num_trajectories: int, max_workers: int = None, max_depth: int = 30, verbose: bool = True):
    """
    Run multiple DS LATS searches in parallel.
    
    Args:
        competition_name: Name of the competition to solve (must be in CHALLENGES)
        num_trajectories: Number of parallel DS LATS searches to run
        max_workers: Maximum number of worker processes (defaults to min(num_trajectories, cpu_count))
        max_depth: Maximum depth of each search tree
        verbose: Whether to print detailed progress information
    
    Returns:
        List of result dictionaries, one for each search trajectory
    
    Example:
        results = run_parallel_ds_lats_searches(
            competition_name='spaceship_titanic',
            num_trajectories=8,
            max_workers=4,
            verbose=True
        )
    """
    if max_workers is None:
        max_workers = min(num_trajectories, mp.cpu_count())
    
    print(f"\n" + "="*80)
    print(f"RUNNING {num_trajectories} PARALLEL DS LATS SEARCHES")
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
            executor.submit(run_single_ds_lats_search, i, competition_name, max_depth, verbose): i 
            for i in range(num_trajectories)
        }
        
        print(f"All {num_trajectories} DS LATS searches submitted to worker pool. Waiting for completion...")
        print("", flush=True)
        
        # Collect results as they complete
        completed_count = 0
        try:
            for future in as_completed(future_to_trajectory, timeout=1000):
                trajectory_id = future_to_trajectory[future]
                completed_count += 1
                
                try:
                    result = future.result(timeout=1800)
                    results.append(result)
                    
                    if verbose:
                        if result['success']:
                            print(f"✓ DS LATS Search {trajectory_id} FOUND SOLUTION - score: {result['solution_score']} ({completed_count}/{num_trajectories})")
                        else:
                            print(f"✗ DS LATS Search {trajectory_id} NO SOLUTION FOUND ({completed_count}/{num_trajectories})")
                        print("", flush=True)
                        
                except TimeoutError:
                    print(f"✗ DS LATS Search {trajectory_id} TIMED OUT ({completed_count}/{num_trajectories})")
                    print("", flush=True)
                    results.append({
                        'trajectory_id': trajectory_id,
                        'success': False,
                        'error': 'Task timed out'
                    })
                except Exception as e:
                    print(f"✗ DS LATS Search {trajectory_id} generated an exception: {e} ({completed_count}/{num_trajectories})")
                    print("", flush=True)
                    results.append({
                        'trajectory_id': trajectory_id,
                        'success': False,
                        'error': str(e)
                    })
        except TimeoutError:
            print(f"✗ Overall timeout reached. {completed_count}/{num_trajectories} tasks completed.")
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Summary statistics
    successful_searches = [r for r in results if r['success']]
    failed_searches = [r for r in results if not r['success']]
    
    print(f"\n" + "="*80)
    print("PARALLEL DS LATS EXECUTION SUMMARY")
    print("="*80)
    print(f"Total searches: {num_trajectories}")
    print(f"Successful: {len(successful_searches)}")
    print(f"Failed: {len(failed_searches)}")
    print(f"Total execution time: {total_time:.2f}s")
    
    if successful_searches:
        avg_time = sum(r['execution_time'] for r in successful_searches) / len(successful_searches)
        avg_score = sum(r['solution_score'] for r in successful_searches) / len(successful_searches)
        best_result = max(successful_searches, key=lambda x: x['solution_score'])
        
        print(f"Average search time: {avg_time:.2f}s")
        print(f"Average solution score: {avg_score:.2f}")
        print(f"Best solution score: {best_result['solution_score']} (Search {best_result['trajectory_id']})")
        print(f"Speedup factor: {avg_time * num_trajectories / total_time:.2f}x")
        
        print(f"\nBest solution found:")
        print(f"  Search ID: {best_result['trajectory_id']}")
        print(f"  Score: {best_result['solution_score']}")
        print(f"  Tree height: {best_result['tree_height']}")
        print(f"  Nodes explored: {best_result['total_nodes']}")
        print(f"  Output file: {best_result['output_file']}")
        print(f"  Solution preview: {best_result['final_solution'][:200]}..." if best_result['final_solution'] else "No solution text")
    
    # Print successful trajectory files
    # files_created = [r for r in results if r.get('output_file')]
    # if files_created:
    #     print(f"\nGenerated output files:")
    #     for result in files_created:
    #         print(f"  - {result['output_file']}")
    
    # # Print failed searches
    # if failed_searches:
    #     print(f"\nFailed searches:")
    #     for result in failed_searches:
    #         print(f"  - Search {result['trajectory_id']}: {result['error']}")
    
    print(f"\n" + "="*80)
    print("PARALLEL DS LATS EXECUTION COMPLETED")
    print("="*80)
    print("", flush=True)
    
    return results


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run parallel DS LATS searches for Kaggle competitions')
    parser.add_argument('--competition', '-c', type=str, 
                       help='Competition name (overrides COMPETITION_NAME env var)')
    parser.add_argument('--trajectories', '-t', type=int,
                       help='Number of parallel DS LATS searches (overrides NUM_TRAJECTORIES env var)')
    parser.add_argument('--max_workers', '-w', type=int,
                       help='Maximum number of worker processes (overrides MAX_WORKERS env var)')
    parser.add_argument('--depth', '-d', type=int,
                       help='Maximum depth of search tree (overrides MAX_DEPTH env var)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--quiet', action='store_true',
                       help='Disable verbose output')
    
    args = parser.parse_args()
    
    # Configuration - command line args override environment variables
    NUM_TRAJECTORIES = args.trajectories if args.trajectories is not None else int(os.environ.get('NUM_TRAJECTORIES', 4))
    MAX_WORKERS = args.max_workers if args.max_workers is not None else int(os.environ.get('MAX_WORKERS', mp.cpu_count()))
    MAX_DEPTH = args.depth if args.depth is not None else int(os.environ.get('MAX_TREE_DEPTH', 30))
    COMPETITION_NAME = args.competition if args.competition is not None else os.environ['COMPETITION_NAME']
    
    # Handle verbose flag
    if args.quiet:
        VERBOSE = False
    elif args.verbose:
        VERBOSE = True
    else:
        VERBOSE = os.environ.get('VERBOSE', 'true').lower() == 'true'
    
    # Show configuration
    print(f"DS LATS Configuration:")
    print(f"  COMPETITION_NAME: {COMPETITION_NAME} ({'CLI' if args.competition is not None else 'ENV/DEFAULT'})")
    print(f"  NUM_TRAJECTORIES: {NUM_TRAJECTORIES} ({'CLI' if args.trajectories is not None else 'ENV/DEFAULT'})")
    print(f"  MAX_WORKERS: {MAX_WORKERS} ({'CLI' if args.max_workers is not None else 'ENV/DEFAULT'})")
    print(f"  MAX_DEPTH: {MAX_DEPTH} ({'CLI' if args.depth is not None else 'ENV/DEFAULT'})")
    print(f"  VERBOSE: {VERBOSE} ({'CLI' if args.verbose or args.quiet else 'ENV/DEFAULT'})")
    print("", flush=True)
    
    # Run parallel DS LATS searches
    results = run_parallel_ds_lats_searches(
        competition_name=COMPETITION_NAME,
        num_trajectories=NUM_TRAJECTORIES,
        max_workers=MAX_WORKERS,
        max_depth=MAX_DEPTH,
        verbose=VERBOSE
    )
    
    # Save results to pickle file
    log_directory = os.path.join(os.environ['SUBMISSIONS_DIR_PREFIX'], f"logs/{COMPETITION_NAME}/lats_reflection")
    os.makedirs(log_directory, exist_ok=True)
    with open(os.path.join(log_directory, f"lats_reflection_{MAX_DEPTH}_depth_{NUM_TRAJECTORIES}_results.pkl"), 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to pickle file")
    
    print(f"\nParallel DS LATS execution completed with {len(results)} searches.")