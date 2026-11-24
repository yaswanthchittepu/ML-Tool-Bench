"""
Parallelized Hierarchical Data Science Agent for Kaggle Competitions

This module provides a parallelized version of the hierarchical ReAct-based data science agent
that can run multiple trajectories simultaneously. The hierarchical agent progresses through
distinct stages of the ML pipeline with stage-specific tools and models.

Key Features:
- Multiprocessing-based parallelization for true concurrency
- Stage-based progression: train_data_loading → test_data_loading → combine_train_test → 
  data_cleaning → feature_engineering → split_train_test → train_data_to_features_target → 
  test_data_to_features → modeling → create_submission_dataframe
- Stage-specific tools and models for focused execution
- Configurable number of parallel trajectories
- Unique file outputs for each trajectory
- Comprehensive result tracking and statistics
- Error handling and recovery

Usage:

1. Command line (overrides environment variables):
   python hierarchical_react_ds_agent_parallel.py --competition spaceship_titanic --trajectories 8 --max_workers 4 --verbose
   python hierarchical_react_ds_agent_parallel.py -c customer_churn_prediction -t 12 -w 6 -q
   
2. Environment Variables (used as defaults):
   export NUM_TRAJECTORIES=8        # Number of parallel trajectories (default: 4)
   export MAX_WORKERS=4             # Max worker processes (default: cpu_count)
   export COMPETITION_NAME=spaceship_titanic  # Competition to solve
   export VERBOSE=true              # Enable detailed logging
   
3. Direct execution:
   python hierarchical_react_ds_agent_parallel.py

4. Programmatic usage:
   from hierarchical_react_ds_agent_parallel import run_parallel_hierarchical_trajectories
   
   results = run_parallel_hierarchical_trajectories(
       num_trajectories=8,
       competition_name='spaceship_titanic',
       max_workers=4,
       verbose=True
   )

Output:
- Each successful trajectory generates a unique CSV file: submissions/{competition_name}/hierarchical_react/hierarchical_react_agent_{trajectory_id}_{uuid}.csv
- Success is determined by whether the expected output file was actually created
- Summary statistics including success rate, execution times, and speedup factor
- Detailed error reporting for failed trajectories and trajectories that didn't create output files

Performance:
- Scales roughly linearly with available CPU cores
- Each trajectory is completely independent with its own stage progression
- Memory usage scales with number of parallel processes
"""

from collections import defaultdict

from langchain.callbacks import get_openai_callback
import feature_engineering as fe
import hyperparameter_tuning as hpt
import evaluation as ev
import mle_utils as ut
import modeling as md
import loading as ld
from func_to_tool_decorator_hierarchical import make_get_and_set_tool, make_get_tool, make_set_tool, override_tool
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

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import sys, ipdb, os, json
import inspect
import warnings
import uuid
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from functools import partial
import argparse
import pickle

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    scratchpad: NotRequired[Dict[str, Any]]
    current_stage: NotRequired[str]
    plan: Dict[str, Any]
    

def setup_hierarchical_tools_and_models(competition_name: str):
    """Set up all decorated functions, stage mappings, and models for a trajectory."""
    load_dotenv()
    
    allowed_functions = set(['read_data', 'create_conditional_feature', 'concatenate_train_test', 'fillna_with_median', 'fillna_with_mode',
        'extract_string_pattern', 'split_string_column', 'create_group_aggregation', 'drop_feature', 'cast_columns',
        'normalize_all_numerical_columns', 'encode_all_categorical_columns', 'split_combined_into_train_test',
        'drop_feature', 'get_features', 'tune_catboost_classifier', 'tune_random_forest_classifier',
        'tune_lightgbm_classifier', 'tune_logistic_classifier', 'tune_xgboost_classifier', 'predict_target', 
        'rename_feature', 'concatenate_dataframes', 'save_dataframe_to_csv'
    ])

    # Load the decorator mapping from JSON file
    with open("function_to_decorator_map.json", "r") as f:
        decorator_map = json.load(f)
    
    # Load the stage mapping from JSON file
    with open("stage_function_mapping_compact.json", "r") as f:
        stage_mapping = json.load(f)
    
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

    # Create stage-specific tool sets
    stage_tools = {}
    for stage_name, stage_info in stage_mapping.items():
        stage_tools[stage_name] = []
        for func_name in stage_info["functions"]:
            if func_name in tool_functions:
                stage_tools[stage_name].append(tool_functions[func_name])

    # Create stage-specific models
    bind_kwargs = {'parallel_tool_calls': False} if('o3' not in os.environ['OPENAI_MODEL']) else {}

    models = {
            stage_name: ChatOpenAI(model=os.environ['OPENAI_MODEL'],
                            base_url=os.environ['OPENAI_BASE_URL'],
                            api_key=os.environ['OPENAI_API_KEY'],
                            temperature=float(os.environ['TEMPERATURE'])
                            ).bind_tools(
                                stage_tools[stage_name], 
                                **bind_kwargs
                            ) 
                        for stage_name in stage_mapping.keys()
    }
    
    # Create submissions directory
    save_directory = os.path.join(os.environ['SUBMISSIONS_DIR_PREFIX'], f"submissions/{competition_name}/hierarchical_react")
    os.makedirs(save_directory, exist_ok=True)
    os.makedirs(f"model_saves/{competition_name}/", exist_ok=True)
    
    return stage_mapping, stage_tools, models


def run_single_hierarchical_trajectory(trajectory_id: int, competition_name: str, verbose: bool = False):
    """
    Run a single hierarchical trajectory for the data science agent.
    
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
        - final_stage: Final stage reached
        - success: Whether the trajectory created the expected output file
        - error: Error message if trajectory failed or didn't create output file
        - output_file: Path to the created output file (if successful)
    """
    
    # Generate unique ID for this trajectory
    mid = f"{trajectory_id}_{uuid.uuid4()}"
    interaction_info = defaultdict(int)
    try:
        if verbose:
            print(f"\n" + "="*80)
            print(f"STARTING HIERARCHICAL TRAJECTORY {trajectory_id}")
            print("="*80)
            print("", flush=True)
        
        # Setup tools and models for this trajectory
        stage_mapping, stage_tools, models = setup_hierarchical_tools_and_models(competition_name)

        save_directory = os.path.join(os.environ['SUBMISSIONS_DIR_PREFIX'], f"submissions/{competition_name}/hierarchical_react")
        
        competition_description = HumanMessage(content=CHALLENGES[competition_name].format(save_directory=save_directory, save_file_name=f"hierarchical_react_agent_{mid}.csv"))
        
        # Stage descriptions (keeping them inside the function to maintain encapsulation)
        stage_descriptions = {
            "train_data_loading": f"You are a Data Scientist in the Data Loading stage of solving a Kaggle challenge, using only the tools available to you. Load the train and test data and explore the data thoroughly. This stage ends when you have loaded the train data successfully.",
            "test_data_loading": f"You are a Data Scientist in the Data Loading stage of solving a Kaggle challenge, using only the tools available to you. This stage ends when you have loaded the test data successfully.",
            "combine_train_test": f"You are a Data Scientist in the Data Loading stage of solving a Kaggle challenge, using only the tools available to you. This stage ends when you have combined the train and test data into a single dataframe successfully, to be used for downstream Data Cleaning and Feature Engineering. This is to ensure same data preprocessing techniques are applied to both partitions.",
            "data_cleaning": f"You are a Data Scientist in the Data Cleaning stage of solving a Kaggle challenge, using only the tools available to you. This stage ends when there are no missing values present in the data. This also includes the column corresponding to the target variable, that may have Nans in the test partition since the target variable is not present in the test partition. You are allowed to be innovative in filling the missing values based on your understanding of the data",
            "feature_engineering": f"You are a Data Scientist in the Feature Engineering stage of solving a Kaggle challenge, using only the tools available to you. Create new features, or delete unimportant features or transform existing features as needed. You are not allowed to delete or modify features that indicate if the row in the data belongs to the train or test partition. You are also not allowed to augment the feature corresponding to the target variable. Use your understanding of the data to aid your decisions. This stage ends when you feel that the features are good enough for modeling, and categorical and numerical features have been properly encoded. After the end of this stage, all the features should be int or float. Ensure that the number of features in the transformed data is not exorbitantly large.",
            "split_train_test": f"You are a Data Scientist in the Split Train Test stage of solving a Kaggle challenge, using only the tools available to you. Split the combined train and test data into train and test dataframes. This stage ends when the train and test dataframes are successfully split from the combined dataframe.",
            "train_data_to_features_target": f"You are a Data Scientist in the Converting the Train Data to Features and Target stage of solving a Kaggle challenge, using only the tools available to you. Convert the train data into features and target. This stage ends when the train data is successfully converted into features and target, for making downstream modeling upon.",
            "test_data_to_features": f"You are a Data Scientist in the Converting the Test Data to Features stage of solving a Kaggle challenge, using only the tools available to you. Convert the test data into features. This stage ends when the test data is successfully converted into features, for making downsream predictions upon.",
            "modeling": f"You are a Data Scientist in the Modeling stage of solving a Kaggle challenge, using only the tools available to you. Train and tune models. You might need to experiment with different model choices and properly tune your hyperparameters to get good performance. Use the provided evaluation tools to evaluate your trained models if needed. This stage ends when the agent has successfully created a model, that it considers to be the best",
            "create_submission_dataframe": f"You are a Data Scientist in the Create Submission stage of solving a Kaggle challenge, using only the tools available to you. Make predictions on the test data, and create a submission dataframe that contains the predictions in the requested format.  This stage ends when the submission dataframe in the correct format is created."
        }
    
        def plan_call(state: AgentState) -> AgentState:
            """Create a high-level plan for the current stage of the ML pipeline."""
            current_stage = state.get("current_stage")
            if(state.get("plan", {}).get(current_stage) is not None):
                return state
            system_prompt = SystemMessage(content=stage_descriptions.get(current_stage, "You are a Data Scientist."))
            response = models.get(current_stage).invoke([f"{system_prompt}. Propose a step by step plan for the current stage of the problem i.e {current_stage}, that will help in clearing the current stage of the problem. Only propose the plan for the current stage you are at. Do not include any other stage's tasks in the plan.\n\n"] + [competition_description])
            state_dict = {**state}
            state_dict['plan'].update({current_stage: response.content})
            return state_dict

        def model_call(state: AgentState) -> AgentState:
            current_stage = state.get("current_stage")
            system_prompt = SystemMessage(content=stage_descriptions.get(current_stage, "You are a Data Scientist."))
            response = models.get(current_stage).invoke([system_prompt] + [competition_description] + [f"\n\nThe plan to solve the current stage i.e {current_stage} of the problem is: {state.get('plan',{}).get(current_stage)}. Use the plan to guide your choice of tools to call\n\n"] + state["messages"])
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

        def should_advance_stage(state: AgentState):
            """Determine if we should advance to the next stage."""
            current_stage = state.get("current_stage")
            stages = ["train_data_loading", "test_data_loading", "combine_train_test", "data_cleaning", "feature_engineering", "split_train_test", "train_data_to_features_target", "test_data_to_features", "modeling", "create_submission_dataframe"]
            
            try:
                current_index = stages.index(current_stage)
                if current_index < len(stages) - 1:
                    if verbose:
                        print(f"Advancing from stage: {current_stage} to stage: {stages[current_index + 1]}")
                    return stages[current_index + 1]
                else:
                    if verbose:
                        print(f"Advancing from stage: {current_stage} to end")
                    return "end"
            except ValueError:
                raise ValueError(f"Invalid stage: {current_stage}")

        def advance_stage(state: AgentState) -> AgentState:
            if(state.get("current_stage") == "create_submission_dataframe"):
                return {"messages": state["messages"], "current_stage": "create_submission_dataframe"}
            else:
                next_stage = should_advance_stage(state)
                return {"messages": state["messages"], "current_stage": next_stage}

        # Create the main graph
        graph = StateGraph(AgentState)
        
        # Add nodes for each stage
        for stage_name in stage_mapping.keys():
            graph.add_node(stage_name, model_call)
            graph.add_node(f"{stage_name}_plan", plan_call)
            # Create tool node for this stage
            stage_tool_node = ToolNode(tools=stage_tools[stage_name])
            graph.add_node(f"{stage_name}_tools", stage_tool_node)
            
            # Add conditional edges for this stage
            if stage_name == "create_submission_dataframe":
                # Last stage goes directly to END
                graph.add_conditional_edges(
                    stage_name,
                    should_continue,
                    {
                        "continue": f"{stage_name}_tools",
                        "end": END,
                    },
                )
            else:
                # Other stages go to stage_advance
                graph.add_conditional_edges(
                    stage_name,
                    should_continue,
                    {
                        "continue": f"{stage_name}_tools",
                        "end": "stage_advance",
                    },
                )

            # Connect tools back to the stage
            graph.add_edge(f"{stage_name}_tools", stage_name)
            # Connect planner to stage
            graph.add_edge(f"{stage_name}_plan", stage_name)

        # Add stage advancement node
        graph.add_node("stage_advance", advance_stage)
        
        # Add conditional edges from stage_advance
        graph.add_conditional_edges(
            "stage_advance",
            lambda state: state.get("current_stage"),
            {
                "train_data_loading": "train_data_loading_plan",
                "test_data_loading": "test_data_loading_plan",
                "combine_train_test": "combine_train_test_plan",
                "data_cleaning": "data_cleaning_plan",
                "feature_engineering": "feature_engineering_plan",
                "split_train_test": "split_train_test_plan",
                "train_data_to_features_target": "train_data_to_features_target_plan",
                "test_data_to_features": "test_data_to_features_plan",
                "modeling": "modeling_plan",
                "create_submission_dataframe": "create_submission_dataframe_plan",
            },
        )
        
        # Set entry point
        graph.set_entry_point("train_data_loading_plan")

        app = graph.compile()

        # Run the trajectory automatically (no interactive session)
        current_state = {"messages": [], "current_stage": "train_data_loading", "plan": {}}
        start_time = time.time()
        with get_openai_callback() as cb:
            response = app.invoke(current_state, config={"recursion_limit": 500})  # Higher limit for hierarchical
            interaction_info['total_tokens'] += cb.total_tokens
            interaction_info['prompt_tokens'] += cb.prompt_tokens
            interaction_info['completion_tokens'] += cb.completion_tokens
            interaction_info['total_cost'] += cb.total_cost
        end_time = time.time()
        
        # Check if the expected output file was created
        expected_file_path = f"submissions/{competition_name}/hierarchical_react/hierarchical_react_agent_{mid}.csv"
        file_exists = os.path.exists(expected_file_path)
        
        result = {
            'trajectory_id': trajectory_id,
            'mid': mid,
            'execution_time': end_time - start_time,
            'final_response': response['messages'][-1].content if response.get('messages') else None,
            'num_messages': len(response.get('messages', [])),
            'final_stage': response.get('current_stage', 'unknown'),
            'success': file_exists,
            'error': None if file_exists else f"Expected output file not found: {expected_file_path}",
            'output_file': expected_file_path if file_exists else None,
            **interaction_info
        }
        
        if verbose:
            status_symbol = "✓" if file_exists else "⚠"
            status_msg = "completed successfully" if file_exists else "completed but no output file found"
            print(f"{status_symbol} Hierarchical Trajectory {trajectory_id} {status_msg} in {result['execution_time']:.2f}s")
            print(f"Final stage: {result['final_stage']}")
            if file_exists:
                print(f"Output file: {expected_file_path}")
            else:
                print(f"Expected file missing: {expected_file_path}")
            print(f"Final response: {result['final_response'][:200]}..." if result['final_response'] else "No final response")
        
        return result
        
    except Exception as e:
        result = {
            'trajectory_id': trajectory_id,
            'mid': mid,
            'success': False,
            'error': str(e)
        }
        
        if verbose:
            print(f"✗ Hierarchical Trajectory {trajectory_id} failed: {e}")
        
        return result


def run_parallel_hierarchical_trajectories(num_trajectories: int, competition_name: str, max_workers: int = None, verbose: bool = True):
    """
    Run multiple hierarchical trajectories in parallel using multiprocessing.
    
    Args:
        num_trajectories: Number of parallel trajectories to run
        competition_name: Name of the competition to solve (must be in CHALLENGES)
        max_workers: Maximum number of worker processes (defaults to min(num_trajectories, cpu_count))
        verbose: Whether to print detailed progress information
    
    Returns:
        List of result dictionaries, one for each trajectory
    
    Example:
        results = run_parallel_hierarchical_trajectories(
            num_trajectories=8,
            competition_name='spaceship_titanic',
            max_workers=4,
            verbose=True
        )
    """
    if max_workers is None:
        max_workers = min(num_trajectories, mp.cpu_count())
    
    print(f"\n" + "="*80)
    print(f"RUNNING {num_trajectories} PARALLEL HIERARCHICAL TRAJECTORIES")
    print(f"Max workers: {max_workers}")
    print(f"Competition: {competition_name}")
    print("="*80, flush=True)
    
    start_time = time.time()
    results = []
    
    # Use ProcessPoolExecutor for true parallelism
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all trajectories
        future_to_trajectory = {
            executor.submit(run_single_hierarchical_trajectory, i, competition_name, verbose): i 
            for i in range(num_trajectories)
        }
        
        print(f"All {num_trajectories} hierarchical trajectories submitted to worker pool. Waiting for completion...")
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
                        print(f"✓ Hierarchical Trajectory {trajectory_id} FINISHED SUCCESSFULLY - output file created ({completed_count}/{num_trajectories})")
                    elif result.get('output_file'):
                        print(f"⚠ Hierarchical Trajectory {trajectory_id} finished with errors but output file exists ({completed_count}/{num_trajectories})") 
                    else:
                        print(f"✗ Hierarchical Trajectory {trajectory_id} NOT FINISHED SUCCESSFULLY ({completed_count}/{num_trajectories})")
                    print("", flush=True)  # Force output to appear immediately
                    
            except Exception as e:
                print(f"✗ Hierarchical Trajectory {trajectory_id} generated an exception: {e} ({completed_count}/{num_trajectories})")
                print("", flush=True)
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
    print("PARALLEL HIERARCHICAL EXECUTION SUMMARY")
    print("="*80)
    print(f"Total trajectories: {num_trajectories}")
    print(f"Successful: {len(successful_trajectories)}")
    print(f"Failed: {len(failed_trajectories)}")
    print(f"Successful Trajectory Names: {[r['mid'] for r in successful_trajectories]}")
    print(f"Total execution time: {total_time:.2f}s")
    
    if successful_trajectories:
        avg_time = sum(r['execution_time'] for r in successful_trajectories) / len(successful_trajectories)
        print(f"Average trajectory time: {avg_time:.2f}s")
        print(f"Speedup factor: {avg_time * num_trajectories / total_time:.2f}x")
    
    # Print failed trajectories
    if failed_trajectories:
        print(f"\nFailed trajectories:")
        for result in failed_trajectories:
            print(f"  - Trajectory {result['trajectory_id']}: {result['error']}")
    
    # Print successful trajectory files
    files_created = [r for r in results if r.get('output_file')]
    if files_created:
        print(f"\nGenerated output files:")
        for result in files_created:
            print(f"  - {result['output_file']}")
    
    # Report trajectories that completed but didn't create files
    completed_no_file = [r for r in results if not r.get('output_file') and r.get('error') and 'Expected output file not found' in r['error']]
    if completed_no_file:
        print(f"\nTrajectories that completed but didn't create output files:")
        for result in completed_no_file:
            print(f"  - Trajectory {result['trajectory_id']}: Execution completed but no CSV file generated")
    
    print(f"\n" + "="*80)
    print("PARALLEL HIERARCHICAL EXECUTION COMPLETED")
    print("="*80)
    print("", flush=True)
    
    return results


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run parallel hierarchical data science agent trajectories')
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
    print(f"Hierarchical Agent Configuration:")
    print(f"  NUM_TRAJECTORIES: {NUM_TRAJECTORIES} ({'CLI' if args.trajectories is not None else 'ENV/DEFAULT'})")
    print(f"  MAX_WORKERS: {MAX_WORKERS} ({'CLI' if args.max_workers is not None else 'ENV/DEFAULT'})")
    print(f"  COMPETITION_NAME: {COMPETITION_NAME} ({'CLI' if args.competition is not None else 'ENV/DEFAULT'})")
    print(f"  VERBOSE: {VERBOSE} ({'CLI' if args.verbose or args.quiet else 'ENV/DEFAULT'})")
    print("", flush=True)
    
    # Run parallel hierarchical trajectories
    results = run_parallel_hierarchical_trajectories(
        num_trajectories=NUM_TRAJECTORIES,
        competition_name=COMPETITION_NAME, 
        max_workers=MAX_WORKERS,
        verbose=VERBOSE
    )
    
    # Save results to pickle file
    log_directory = os.path.join(os.environ['SUBMISSIONS_DIR_PREFIX'], f"logs/{COMPETITION_NAME}/hierarchical_react")
    os.makedirs(log_directory, exist_ok=True)
    with open(os.path.join(log_directory, f"hierarchical_react_{NUM_TRAJECTORIES}_trajectories_results.pkl"), 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to pickle file")
    
    print(f"\nParallel hierarchical execution completed with {len(results)} trajectories.")
