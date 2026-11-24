"""
Parallelized Data Science Agent for Kaggle Competitions

This module provides a parallelized version of the ReAct-based data science agent
that can run multiple trajectories simultaneously for improved exploration and
potentially better solutions.

Key Features:
- Multiprocessing-based parallelization for true concurrency
- Configurable number of parallel trajectories
- Unique file outputs for each trajectory
- Comprehensive result tracking and statistics
- Error handling and recovery

Usage:

1. Command line (overrides environment variables):
   python react_ds_agent_parallel.py --competition spaceship_titanic --trajectories 8 --workers 4 --verbose
   python react_ds_agent_parallel.py -c customer_churn_prediction -t 12 -w 6 -q
   
2. Environment Variables (used as defaults):
   export NUM_TRAJECTORIES=8        # Number of parallel trajectories (default: 4)
   export MAX_WORKERS=4             # Max worker processes (default: cpu_count)
   export COMPETITION_NAME=spaceship_titanic  # Competition to solve
   export VERBOSE=true              # Enable detailed logging
   
3. Direct execution:
   python react_ds_agent_parallel.py

4. Programmatic usage:
   from react_ds_agent_parallel import run_parallel_trajectories
   
   results = run_parallel_trajectories(
       num_trajectories=8,
       competition_name='spaceship_titanic',
       max_workers=4,
       verbose=True
   )

Output:
- Each successful trajectory generates a unique CSV file: submissions/{competition_name}/react/react_agent_{trajectory_id}_{uuid}.csv
- Success is determined by whether the expected output file was actually created
- Summary statistics including success rate, execution times, and speedup factor
- Detailed error reporting for failed trajectories and trajectories that didn't create output files

Performance:
- Scales roughly linearly with available CPU cores
- Each trajectory is completely independent
- Memory usage scales with number of parallel processes
"""

from collections import defaultdict
import feature_engineering as fe
import hyperparameter_tuning as hpt
import evaluation as ev
import mle_utils as ut
import modeling as md
import loading as ld
from func_to_tool_decorator import make_get_and_set_tool, make_get_tool, make_set_tool, override_tool
from challenges import CHALLENGES

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
from langchain.callbacks import get_openai_callback

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import sys, ipdb, os, json
import inspect
import uuid
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from functools import partial
import argparse
import pickle
from datetime import datetime

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    scratchpad: NotRequired[Dict[str, Any]]
    plan: NotRequired[str]
    summary: NotRequired[RunningSummary]
    num_tool_calls: NotRequired[int]
    trajectory: NotRequired[list]

def setup_tools_and_model(competition_name: str):
    """Set up all decorated functions and model for a trajectory."""
    load_dotenv()
    
    # Load the decorator mapping from JSON file
    with open("function_to_decorator_map.json", "r") as f:
        decorator_map = json.load(f)
    
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

                print(f"✓ Successfully Applied {decorator_name} to {module_name}.{func_name}")
                
            except Exception as e:
                print(f"✗ Failed to apply {decorator_name} to {module_name}.{func_name}: {e}")

    print(f"\nSuccessfully decorated {sum(len(funcs) for funcs in decorated_functions.values())} functions")

    # Merge all decorated functions into a single dictionary
    all_decorated_functions = {}
    for module_dict in decorated_functions.values():
        all_decorated_functions.update(module_dict)

    # Apply @tool decorator to all decorated functions
    tool_functions = {}
    for func_name, decorated_func in all_decorated_functions.items():
        try:
            # Apply @tool decorator
            tool_func = tool(decorated_func)
            tool_functions[func_name] = tool_func
            print(f"✓ Successfully Applied @tool to {func_name}")
        except Exception as e:
            print(f"✗ Failed to apply @tool to {func_name}: {e}")
    
    print(f"\nSuccessfully created {len(tool_functions)} tool functions")

    # ------------------------------------------------------------
    # Model Definition and Binding Tools to Model
    # ------------------------------------------------------------

    tools = list(tool_functions.values())

    # bind_kwargs = {'parallel_tool_calls': False} if('o3' not in os.environ['OPENAI_MODEL']) else {}
    bind_kwargs = {'parallel_tool_calls': False} if('o3' not in os.environ['OPENAI_MODEL'] and 'claude' not in os.environ['OPENAI_MODEL']) else {}
    model = ChatOpenAI(model=os.environ['OPENAI_MODEL'],
                        base_url=os.environ['OPENAI_BASE_URL'],
                        api_key=os.environ['OPENAI_API_KEY'],
                        temperature=float(os.environ['TEMPERATURE'])
                    ).bind_tools(tools, 
                            **bind_kwargs
                        )
    summarization_model = ChatOpenAI(model=os.environ['OPENAI_MODEL'],
                            base_url=os.environ['OPENAI_BASE_URL'],
                            api_key=os.environ['OPENAI_API_KEY'],
                            temperature=float(os.environ['TEMPERATURE'])
                        )

    save_directory = os.path.join(os.environ['SUBMISSIONS_DIR_PREFIX'], f"submissions/{competition_name}/react")
    os.makedirs(save_directory, exist_ok=True)
    os.makedirs(f"model_saves/{competition_name}/", exist_ok=True)
    
    return tools, model, summarization_model


def run_single_trajectory(trajectory_id: int, competition_name: str, verbose: bool = False):
    """
    Run a single trajectory for the data science agent.
    
    Args:
        trajectory_id: Unique identifier for this trajectory
        competition_name: Name of the competition to solve
        verbose: Whether to print detailed progress information
    
    Returns:
        Dictionary containing trajectory results including:
        - trajectory_id: ID of the trajectory
        - mid: Unique model ID used for file naming
        - execution_time: Time taken to complete the trajectory
        - final_response: Final agent response
        - num_messages: Number of messages in the conversation
        - success: Whether the trajectory created the expected output file
        - error: Error message if trajectory failed or didn't create output file
        - output_file: Path to the created output file (if successful)
    """

    

    # Generate unique ID for this trajectory
    mid = f"{trajectory_id}_{uuid.uuid4()}"
    interaction_info = defaultdict(int)
    worst_case_per_query_info = defaultdict(int)

    # Trajectory logging utilities
    def log_trajectory_step(state: AgentState, step_type: str, step_data: dict) -> AgentState:
        """Add a step to the trajectory log"""
        if 'trajectory' not in state:
            state['trajectory'] = []
        
        trajectory_step = {
            'step_number': len(state['trajectory']) + 1,
            'timestamp': datetime.now().isoformat(),
            'step_type': step_type,
            **step_data
        }
        
        # return Command(update={
        #     "trajectory": state.get('trajectory').append(trajectory_step)
        # })
        state['trajectory'].append(trajectory_step)
        print(f"[TRAJECTORY] Step {trajectory_step['step_number']}: {step_type}")
        return state

    def save_trajectory(trajectory: list, save_directory: str, agent_id: str):
        """Save the complete trajectory to a JSON file"""
        trajectory_file = os.path.join(save_directory, f"trajectory_{agent_id}.json")
        
        # Add summary statistics
        trajectory_summary = {
            'total_steps': len(trajectory),
            'step_types': {step['step_type']: sum(1 for s in trajectory if s['step_type'] == step['step_type']) 
                        for step in trajectory},
            'start_time': trajectory[0]['timestamp'] if trajectory else None,
            'end_time': trajectory[-1]['timestamp'] if trajectory else None,
            'trajectory': trajectory
        }
        
        with open(trajectory_file, 'w') as f:
            json.dump(trajectory_summary, f, indent=2)
        
        print(f"[TRAJECTORY] Saved complete trajectory to {trajectory_file}")
        return trajectory_file

    try:
        if verbose:
            print(f"\n" + "="*80)
            print(f"STARTING TRAJECTORY {trajectory_id}")
            print("="*80)
        
        # Setup tools and model for this trajectory
        tools, model, summarization_model = setup_tools_and_model(competition_name)
        
        save_directory = os.path.join(os.environ['SUBMISSIONS_DIR_PREFIX'], f"submissions/{competition_name}/react")
        competition_description = HumanMessage(content=CHALLENGES[competition_name].format(save_directory=save_directory, save_file_name=f"react_agent_{mid}.csv"))
        
        def plan_call(state: AgentState) -> AgentState:
            """Create a high-level plan for the entire ML pipeline."""
            if(state.get("plan") is not None):
                return state
            nonlocal worst_case_per_query_info
            plan_prompt = SystemMessage(content="""
            You are a Data Scientist tasked with solving the Kaggle competition provided below. Create a high level plan of what steps you will take to solve the problem.
            """)
            with get_openai_callback() as cb:
                response = model.invoke([plan_prompt] + [competition_description])
                if(cb.total_tokens > worst_case_per_query_info['total_tokens']):
                    worst_case_per_query_info['total_tokens'] = cb.total_tokens
                    worst_case_per_query_info['prompt_tokens'] = cb.prompt_tokens
                    worst_case_per_query_info['prompt_tokens_cached'] = cb.prompt_tokens_cached
                    worst_case_per_query_info['completion_tokens'] = cb.completion_tokens
                    worst_case_per_query_info['total_cost'] = cb.total_cost
            # Log the planning step completion
            state = log_trajectory_step(state, 'plan_completion', {
                'action': 'plan_generated',
                'plan_content': response.content,
                'tokens_used': cb.total_tokens,
                'cost': cb.total_cost
            })
            state['plan'] = response.content
            return state

        # def model_call(state:AgentState) -> AgentState:
        #     system_prompt = SystemMessage(content=
        #         "You are a Data Scientist tasked with solving the Kaggle competition provided below. "
        #     )
        #     response = model.invoke([system_prompt] + [competition_description] + [f"\n\nThe plan proopsed by an agent to solve the problem is: {state['plan']}. Use this plan to guide your choice of tools to call\n\n"] + state["messages"])
        #     return {"messages": [response]}

        def model_call(state:AgentState) -> AgentState:
            system_prompt = SystemMessage(content=
                "You are a Data Scientist tasked with solving the Kaggle competition provided below. "
            )
            nonlocal worst_case_per_query_info
            with get_openai_callback() as cb:
                if('llama' in os.environ['OPENAI_MODEL']):
                    response = model.invoke([system_prompt] + [competition_description] + [f"\n\nOnly call one tool at a time. Multiple parallel tool calls are not allowed.\n\n"] + state["messages"])
                else:
                    response = model.invoke([system_prompt] + [competition_description] + [f"\n\nThe plan proposed by an agent to solve the problem is: {state['plan']}. Use this plan to guide your choice of tools to call. Only call one tool at a time. Multiple parallel tool calls are not allowed.\n\n"] + state["messages"])
                if(cb.total_tokens > worst_case_per_query_info['total_tokens']):
                    worst_case_per_query_info['total_tokens'] = cb.total_tokens
                    worst_case_per_query_info['prompt_tokens'] = cb.prompt_tokens
                    worst_case_per_query_info['prompt_tokens_cached'] = cb.prompt_tokens_cached
                    worst_case_per_query_info['completion_tokens'] = cb.completion_tokens
                    worst_case_per_query_info['total_cost'] = cb.total_cost
            return {"messages": [response]}

        # def should_continue(state: AgentState): 
        #     messages = state["messages"]
        #     last_message = messages[-1]
        #     if not last_message.tool_calls: 
        #         return "end"
        #     else:
        #         return "continue"

        def should_continue(state: AgentState): 
            messages = state["messages"]
            last_message = messages[-1]
            
            # ipdb.set_trace()
            # Check for GPT-style tool calls
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "continue"
            
            # Check for Llama-style tool calls (JSON in content)
            if hasattr(last_message, 'content') and last_message.content:
                try:
                    import json
                    # Try to parse the content as JSON
                    content = last_message.content.strip()
                    if content.startswith('{') and content.endswith('}'):
                        parsed = json.loads(content)
                        # Check if it looks like a tool call
                        if isinstance(parsed, dict) and 'name' in parsed and 'parameters' in parsed:
                            return "continue"
                except (json.JSONDecodeError, AttributeError):
                    pass
            
            return "end"
        
        # ------------------------------
        # Graph Definition
        # ------------------------------
        graph = StateGraph(AgentState)

        # Custom ToolNode wrapper to handle both Llama and GPT formats
        def custom_tool_node(state: AgentState) -> AgentState:
            """Custom tool node that handles both Llama JSON format and GPT tool_calls format"""
            messages = state["messages"]
            last_message = messages[-1]

            # Log tool execution initiation
            tool_execution_info = {'action': 'tool_execution_started'}

            # Check if we need to convert Llama JSON format to tool_calls
            if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
                # Try to parse Llama JSON format
                if hasattr(last_message, 'content') and last_message.content:
                    try:
                        import json
                        import string
                        import random
                        content = last_message.content.strip()
                        if content.startswith('{') and content.endswith('}'):
                            parsed = json.loads(content)
                            if isinstance(parsed, dict) and 'name' in parsed and 'parameters' in parsed:
                                tool_execution_info['format_conversion'] = 'llama_to_gpt'
                                tool_execution_info['original_content'] = content
                                # Fix parameters that are string representations of Python objects
                                parameters = parsed['parameters']
                                import ast
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
                                
                                # Modify the existing message in-place to match GPT format
                                last_message.content = ""  # Clear content like GPT does
                                last_message.tool_calls = [{
                                    'name': parsed['name'],
                                    'args': parameters,
                                    'id': call_id,
                                    'type': 'tool_call'
                                }]
                                # Ensure additional_kwargs has tool_calls for compatibility
                                if not hasattr(last_message, 'additional_kwargs'):
                                    last_message.additional_kwargs = {}
                                last_message.additional_kwargs['tool_calls'] = [{
                                    'id': call_id,
                                    'function': {
                                        'arguments': json.dumps(parameters),
                                        'name': parsed['name']
                                    },
                                    'type': 'function'
                                }]
                                
                    except (json.JSONDecodeError, AttributeError) as e:
                        print(f"Failed to parse Llama JSON format: {e}")
            
                # Extract tool call information for logging
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                tool_execution_info['tools_to_execute'] = []
                for tool_call in last_message.tool_calls:
                    tool_execution_info['tools_to_execute'].append({
                        'tool_name': tool_call['name'],
                        'tool_args': tool_call['args'],
                        'tool_id': tool_call['id']
                    })
            
            state = log_trajectory_step(state, 'tool_execution_initiation', tool_execution_info)

            # Use standard ToolNode
            standard_tool_node = ToolNode(tools=tools)
            result = standard_tool_node.invoke(state)

            # If it's a Command, apply the update. This is always the case when the tool call succeeded. 
            # If the tool call failed, result is a dictionary with a single key 'messages' containing a single ToolMessage.
            # The else block handles the case where the tool call failed.
            if isinstance(result, list) and all(isinstance(res, Command) for res in result):
                update = result[0].update
                for k in update.keys():
                    state[k] = update[k]
            else:
                if(isinstance(result, dict) and 'messages' in result and len(result['messages']) == 1 and isinstance(result['messages'][0], ToolMessage)):
                    state['messages'].append(result['messages'][0])
                else:
                    raise ValueError(f"Unexpected result type for tool call: {type(result)} \n\n Result: {result}")

            # Log tool execution completion
            tool_completion_info = {
                'action': 'tool_execution_completed',
                # 'new_message_count': len(result[0].update['messages']),
            }
        
            # Extract tool results from the new messages
        
            tool_results = []
            msg = state["messages"][-1]
            if hasattr(msg, 'content'):
                # Truncate very long tool results for logging
                content = str(msg.content)
                if len(content) > 1000:
                    content = content[:1000] + "... [truncated]"
                
                tool_results.append({
                    'message_type': type(msg).__name__,
                    'content_preview': content,
                    'content_length': len(str(msg.content))
                })
            
            tool_completion_info['tool_results'] = tool_results
            
            state = log_trajectory_step(state, 'tool_execution_completion', tool_completion_info)
            
            return state

        # Add plan node
        if('llama' in os.environ['OPENAI_MODEL']):
            graph.add_node("our_agent", model_call)
            graph.add_node("tools", custom_tool_node)

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
        else:
            graph.add_node("plan", plan_call)
            graph.add_node("our_agent", model_call)
            graph.add_node("tools", custom_tool_node)

            graph.set_entry_point("plan")
            
            # Add edge from plan to agent
            graph.add_edge("plan", "our_agent")

            graph.add_conditional_edges(
                "our_agent",
                should_continue,
                {
                    "continue": "tools",
                    "end": END,
                },
            )

            graph.add_edge("tools", "our_agent")
        
        # # Add plan node
        # graph.add_node("plan", plan_call)
        # graph.add_node("our_agent", model_call)

        # tool_node = ToolNode(tools=tools)
        # graph.add_node("tools", tool_node)

        # graph.set_entry_point("plan")
        
        # # Add edge from plan to agent
        # graph.add_edge("plan", "our_agent")

        # graph.add_conditional_edges(
        #     "our_agent",
        #     should_continue,
        #     {
        #         "continue": "tools",
        #         "end": END,
        #     },
        # )

        # graph.add_edge("tools", "our_agent")

        app = graph.compile()

        # Run the trajectory
        current_state = {"messages": []}
        start_time = time.time()
        with get_openai_callback() as cb:
            response = app.invoke(current_state, config={"recursion_limit": 250})
            interaction_info['total_tokens'] += cb.total_tokens
            interaction_info['prompt_tokens'] += cb.prompt_tokens
            interaction_info['completion_tokens'] += cb.completion_tokens
            interaction_info['total_cost'] += cb.total_cost
        end_time = time.time()
        total_time = end_time - start_time
        
        # Check if the expected output file was created
        expected_file_path = f"submissions/{competition_name}/react/react_agent_{mid}.csv"
        file_exists = os.path.exists(expected_file_path)

        # Save trajectory
        if 'trajectory' in response and response['trajectory']:
            # Log final execution summary
            log_trajectory_step(response, 'execution_summary', {
                'action': 'agent_execution_completed',
                'total_execution_time': total_time,
                'total_tokens': interaction_info['total_tokens'],
                'total_cost': interaction_info['total_cost'],
                'final_message_count': len(response.get('messages', [])),
                'competition_name': os.environ.get('COMPETITION_NAME', 'unknown')
            })
            
            trajectory_file = save_trajectory(
                response['trajectory'], 
                save_directory, 
                str(mid)
            )
            print(f"✓ Agent trajectory saved to: {trajectory_file}")
        else:
            print("⚠ No trajectory data found to save")
        
        result = {
            'trajectory_id': trajectory_id,
            'mid': mid,
            'execution_time': end_time - start_time,
            'final_response': response['messages'][-1].content if response.get('messages') else None,
            'num_messages': len(response.get('messages', [])),
            'success': file_exists,
            'error': None if file_exists else f"Expected output file not found: {expected_file_path}",
            'output_file': expected_file_path if file_exists else None,
            'interaction_info': interaction_info,
            'worst_case_per_query_info': worst_case_per_query_info,
        }
        
        if verbose:
            status_symbol = "✓" if file_exists else "⚠"
            status_msg = "completed successfully" if file_exists else "completed but no output file found"
            print(f"{status_symbol} Trajectory {trajectory_id} {status_msg} in {result['execution_time']:.2f}s")
            if file_exists:
                print(f"Output file: {expected_file_path}")
            else:
                print(f"Expected file missing: {expected_file_path}")
            print(f"Final response: {result['final_response'][:200]}..." if result['final_response'] else "No final response")
            print(f"\n\n Trajctory ID: {trajectory_id}, Worst case per query information: {result['worst_case_per_query_info']}, Total Cost: {result['interaction_info']['total_cost']} \n\n")
        
        return result
        
    except Exception as e:
        return {
            'trajectory_id': trajectory_id,
            'mid': mid,
            'success': False
        }


def run_parallel_trajectories(num_trajectories: int, competition_name: str, max_workers: int = None, verbose: bool = True):
    """
    Run multiple trajectories in parallel using multiprocessing.
    
    Args:
        num_trajectories: Number of parallel trajectories to run
        competition_name: Name of the competition to solve (must be in CHALLENGES)
        max_workers: Maximum number of worker processes (defaults to min(num_trajectories, cpu_count))
        verbose: Whether to print detailed progress information
    
    Returns:
        List of result dictionaries, one for each trajectory
    
    Example:
        results = run_parallel_trajectories(
            num_trajectories=8,
            competition_name='spaceship_titanic',
            max_workers=4,
            verbose=True
        )
    """
    if max_workers is None:
        max_workers = min(num_trajectories, mp.cpu_count())
    
    print(f"\n" + "="*80)
    print(f"RUNNING {num_trajectories} PARALLEL TRAJECTORIES")
    print(f"Max workers: {max_workers}")
    print(f"Competition: {competition_name}")
    print("="*80, flush=True)
    
    start_time = time.time()
    results = []
    
    # Use ProcessPoolExecutor for true parallelism
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all trajectories
        future_to_trajectory = {
            executor.submit(run_single_trajectory, i, competition_name, verbose): i 
            for i in range(num_trajectories)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_trajectory):
            trajectory_id = future_to_trajectory[future]
            try:
                result = future.result()
                results.append(result)
                
                if verbose:
                    if result['success']:
                        print(f"✓ Trajectory {trajectory_id} FINISHED SUCCESSFULLY - output file created")
                    # elif result.get('output_file'):
                    #     print(f"⚠ Trajectory {trajectory_id} finished with errors but output file exists") 
                    else:
                        print(f"✗ Trajectory {trajectory_id} NOT FINISHED SUCCESSFULLY")
                    
            except Exception as e:
                print(f"✗ Trajectory {trajectory_id} generated an exception: {e}")
                results.append({
                    'trajectory_id': trajectory_id,
                    'success': False,
                    'error': str(e)
                })
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Summary statistics
    successful_trajectories = [r for r in results if r['success']]
    failed_trajectories = [r for r in results if not r['success']]
    
    print(f"\n" + "="*80)
    print("PARALLEL EXECUTION SUMMARY")
    print("="*80)
    print(f"Total trajectories: {num_trajectories}")
    print(f"Successful: {len(successful_trajectories)}")
    print(f"Failed: {len(failed_trajectories)}")
    print(f"Successful Trajectory Names: {[r['mid'] for r in successful_trajectories]}")
    print(f"Total execution time: {total_time:.2f}s")
    
    # if successful_trajectories:
    #     avg_time = sum(r['execution_time'] for r in successful_trajectories) / len(successful_trajectories)
    #     print(f"Average trajectory time: {avg_time:.2f}s")
    #     print(f"Speedup factor: {avg_time * num_trajectories / total_time:.2f}x")
    
    # # Print failed trajectories
    # if failed_trajectories:
    #     print(f"\nFailed trajectories:")
    #     for result in failed_trajectories:
    #         print(f"  - Trajectory {result['trajectory_id']}: {result['error']}")
    
    # # Print successful trajectory files
    # files_created = [r for r in results if r.get('output_file')]
    # if files_created:
    #     print(f"\nGenerated output files:")
    #     for result in files_created:
    #         print(f"  - {result['output_file']}")
    
    # # Report trajectories that completed but didn't create files
    # completed_no_file = [r for r in results if not r.get('output_file') and r.get('error') and 'Expected output file not found' in r['error']]
    # if completed_no_file:
    #     print(f"\nTrajectories that completed but didn't create output files:")
    #     for result in completed_no_file:
    #         print(f"  - Trajectory {result['trajectory_id']}: Execution completed but no CSV file generated")
    
    return results


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run parallel data science agent trajectories')
    parser.add_argument('--competition', '-c', type=str, 
                       help='Competition name (overrides COMPETITION_NAME env var)')
    parser.add_argument('--trajectories', '-t', type=int,
                       help='Number of parallel trajectories (overrides NUM_TRAJECTORIES env var)')
    parser.add_argument('--max_workers', '-w', type=int,
                       help='Maximum number of worker processes (overrides MAX_WORKERS env var)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Disable verbose output')
    
    args = parser.parse_args()
    
    # Configuration - command line args override environment variables
    NUM_TRAJECTORIES = args.trajectories if args.trajectories is not None else int(os.environ.get('NUM_TRAJECTORIES', 4))
    MAX_WORKERS = args.max_workers if args.max_workers is not None else int(os.environ.get('MAX_WORKERS', mp.cpu_count()))
    COMPETITION_NAME = args.competition if args.competition is not None else os.environ['COMPETITION_NAME']
    
    # Handle verbose flag
    if args.quiet:
        VERBOSE = False
    elif args.verbose:
        VERBOSE = True
    else:
        VERBOSE = os.environ.get('VERBOSE', 'true').lower() == 'true'
    
    # Show configuration
    print(f"Configuration:")
    print(f"  NUM_TRAJECTORIES: {NUM_TRAJECTORIES} ({'CLI' if args.trajectories is not None else 'ENV/DEFAULT'})")
    print(f"  MAX_WORKERS: {MAX_WORKERS} ({'CLI' if args.max_workers is not None else 'ENV/DEFAULT'})")
    print(f"  COMPETITION_NAME: {COMPETITION_NAME} ({'CLI' if args.competition is not None else 'ENV/DEFAULT'})")
    print(f"  VERBOSE: {VERBOSE} ({'CLI' if args.verbose or args.quiet else 'ENV/DEFAULT'})")
    print("", flush=True)
    
    # Run parallel trajectories
    results = run_parallel_trajectories(
        num_trajectories=NUM_TRAJECTORIES,
        competition_name=COMPETITION_NAME, 
        max_workers=MAX_WORKERS,
        verbose=VERBOSE
    )
    
    # Save results to pickle file
    log_directory = os.path.join(os.environ['SUBMISSIONS_DIR_PREFIX'], f"logs/{COMPETITION_NAME}/react")
    os.makedirs(log_directory, exist_ok=True)
    with open(os.path.join(log_directory, f"react_agent_{NUM_TRAJECTORIES}_trajectories_results.pkl"), 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to pickle file")
    
    print(f"\nParallel execution completed with {len(results)} trajectories.")
