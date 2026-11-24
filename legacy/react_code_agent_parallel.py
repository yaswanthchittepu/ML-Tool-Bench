"""
Parallelized React Code Agent for Kaggle Competitions

This module provides a parallelized version of the React code generation agent
that can run multiple trajectories simultaneously for improved exploration.

Key Features:
- Multiprocessing-based parallelization for true concurrency
- Configurable number of parallel trajectories
- Unique file outputs for each trajectory
- Comprehensive result tracking and statistics
- Generate-Execute-Reflect loop per trajectory

Usage:

1. Command line (overrides environment variables):
   python react_code_agent_parallel.py --competition spaceship_titanic --trajectories 8 --workers 4 --verbose
   python react_code_agent_parallel.py -c customer_churn_prediction -t 12 -w 6 -q
   
2. Environment Variables (used as defaults):
   export NUM_TRAJECTORIES=8        # Number of parallel trajectories (default: 4)
   export MAX_WORKERS=4             # Max worker processes (default: cpu_count)
   export COMPETITION_NAME=spaceship_titanic  # Competition to solve
   export MAX_ITERATIONS=10         # Max iterations per trajectory (default: 10)
   export VERBOSE=true              # Enable detailed logging
   
3. Direct execution:
   python react_code_agent_parallel.py

Output:
- Each successful trajectory generates:
  - solution_final_{trajectory_id}_{uuid}.py - Final successful code
  - react_code_agent_{trajectory_id}_{uuid}.csv - Submission file
  - trajectory_{trajectory_id}_{uuid}.json - Execution log
- Summary statistics including success rate, execution times, and speedup factor
"""

import sys
import os
import io
import traceback
import pandas as pd
import numpy as np
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from collections import defaultdict
import time
import json
import uuid
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import pickle

from typing import Annotated, Sequence, NotRequired, Any, Dict
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv

from challenges import CHALLENGES


# Set matplotlib to non-interactive backend to avoid GUI threading issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    trajectory: NotRequired[list]
    current_iteration: NotRequired[int]
    max_iterations: NotRequired[int]
    solution_approved: NotRequired[bool]


def run_single_trajectory(trajectory_id: int, competition_name: str, max_iterations: int = 10, verbose: bool = False):
    """
    Run a single trajectory for the React code agent.
    
    Args:
        trajectory_id: Unique identifier for this trajectory
        competition_name: Name of the competition to solve
        max_iterations: Maximum number of generate-execute-reflect iterations
        verbose: Whether to print detailed progress information
    
    Returns:
        Dictionary containing trajectory results
    """
    
    # Generate unique ID for this trajectory
    mid = f"{trajectory_id}_{uuid.uuid4()}"
    interaction_info = defaultdict(int)
    worst_case_info = defaultdict(int)
    
    # Global namespace to maintain state across code executions
    EXEC_NAMESPACE = {
        'pd': pd,
        'np': np,
        'plt': plt,
        'matplotlib': matplotlib,
        '__builtins__': __builtins__,
    }
    
    def log_trajectory_step(state: AgentState, step_type: str, step_data: dict) -> AgentState:
        """Add a step to the trajectory log"""
        if 'trajectory' not in state:
            state['trajectory'] = []
        
        trajectory_step = {
            'step_number': len(state['trajectory']) + 1,
            'timestamp': datetime.now().isoformat(),
            'step_type': step_type,
            'iteration': state.get('current_iteration', 0),
            **step_data
        }
        
        state['trajectory'].append(trajectory_step)
        if verbose:
            print(f"[TRAJ-{trajectory_id}] Step {trajectory_step['step_number']}: {step_type} (Iter {state.get('current_iteration', 0)})")
        return state
    
    def save_trajectory(trajectory: list, save_directory: str, agent_id: str):
        """Save the complete trajectory to a JSON file"""
        trajectory_file = os.path.join(save_directory, f"trajectory_{agent_id}.json")
        
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
        
        if verbose:
            print(f"[TRAJ-{trajectory_id}] Saved trajectory to {trajectory_file}")
        return trajectory_file
    
    @tool
    def execute_full_solution(code: str) -> str:
        """Execute a COMPLETE Python solution and return all output."""
        nonlocal EXEC_NAMESPACE
        
        # Reset namespace for fresh execution (keep only imports)
        EXEC_NAMESPACE = {
            'pd': pd,
            'np': np,
            'plt': plt,
            'matplotlib': matplotlib,
            '__builtins__': __builtins__,
        }
        
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        result = {
            'success': False,
            'stdout': '',
            'stderr': '',
            'error': None,
        }
        
        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, EXEC_NAMESPACE)
            
            result['success'] = True
            result['stdout'] = stdout_capture.getvalue()
            result['stderr'] = stderr_capture.getvalue()
            
        except Exception as e:
            result['error'] = f"{type(e).__name__}: {str(e)}"
            result['traceback'] = traceback.format_exc()
            result['stderr'] = stderr_capture.getvalue()
        
        # Format output
        output_parts = []
        
        if result['success']:
            output_parts.append("✓ CODE EXECUTED SUCCESSFULLY")
        else:
            output_parts.append("✗ CODE EXECUTION FAILED")
        
        if result['stdout']:
            output_parts.append(f"\n--- STDOUT ---\n{result['stdout']}")
        
        if result['stderr']:
            output_parts.append(f"\n--- STDERR ---\n{result['stderr']}")
        
        if result['error']:
            output_parts.append(f"\n--- ERROR ---\n{result['error']}")
            output_parts.append(f"\n--- TRACEBACK ---\n{result['traceback']}")
        
        return "\n".join(output_parts)
    
    @tool
    def mark_solution_complete() -> str:
        """Call when satisfied with your solution."""
        return "Solution marked as complete. Execution will terminate."
    
    try:
        if verbose:
            print(f"\n{'='*80}")
            print(f"STARTING TRAJECTORY {trajectory_id}")
            print(f"{'='*80}")
        
        load_dotenv()
        
        # Setup
        save_directory = f"submissions/{competition_name}/react_code"
        os.makedirs(save_directory, exist_ok=True)
        
        # Initialize model
        tools = [execute_full_solution, mark_solution_complete]
        
        model_name = os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')
        bind_kwargs = {}
        if 'o3' not in model_name and 'claude' not in model_name:
            bind_kwargs['parallel_tool_calls'] = False
        
        model = ChatOpenAI(
            model=model_name,
            base_url=os.environ.get('OPENAI_BASE_URL'),
            api_key=os.environ.get('OPENAI_API_KEY'),
            temperature=float(os.environ.get('TEMPERATURE', '0.7'))
        ).bind_tools(tools, **bind_kwargs)
        
        # Competition description
        competition_description = HumanMessage(
            content=CHALLENGES[competition_name].format(
                save_directory=save_directory,
                save_file_name=f"react_code_agent_{mid}.csv"
            )
        )
        
        def generate_solution(state: AgentState) -> dict:
            """Generate a complete solution or reflection."""
            
            iteration = state.get('current_iteration', 0)
            
            if iteration == 0:
                # First iteration - generate initial solution
                system_prompt = SystemMessage(content="""
    You are a Data Scientist solving a Kaggle competition by writing Python code.
    
    Your task: Write a COMPLETE solution as a single Python script that:
    1. Loads the training and test data
    2. Performs data exploration and preprocessing
    3. Engineers features as needed
    4. Trains a machine learning model
    5. Makes predictions on test data
    6. Saves predictions to the specified CSV file
    
    Available packages (already installed):
    - pandas (imported as pd), numpy (imported as np), matplotlib.pyplot (imported as plt)
    - scikit-learn - all modules (ensemble, linear_model, preprocessing, model_selection, metrics, etc.)
    - xgboost, lightgbm, catboost - gradient boosting
    - matplotlib (NON-INTERACTIVE: use plt.savefig(), NEVER plt.show()), plotly
    - scipy, joblib
    
    Guidelines:
    - Write the ENTIRE solution in one code block
    - Add print statements to show progress and key metrics
    - Handle missing values appropriately
    - Use cross-validation to evaluate your model
    - For plots: use plt.savefig('plot.png') then plt.close(), NEVER plt.show()
    
    CRITICAL: You MUST call the execute_full_solution tool with your code. Do NOT just return code in markdown.
    Call: execute_full_solution(code="your entire solution code here")
    """)
                
                messages = [system_prompt, competition_description, HumanMessage(content="Write and execute the complete solution using the execute_full_solution tool.")]
            else:
                # Subsequent iterations - reflect and improve
                system_prompt = SystemMessage(content="""
    You are reflecting on the execution of your previous solution.
    
    Analyze the output/errors and:
    - If successful and predictions saved: call mark_solution_complete()
    - If errors occurred: generate IMPROVED solution using execute_full_solution(code="...")
    
    Available packages (already installed):
    - pandas (pd), numpy (np), matplotlib.pyplot (plt) - pre-imported
    - scikit-learn, xgboost, lightgbm, catboost
    - matplotlib (use plt.savefig(), NEVER plt.show()), plotly
    
    CRITICAL: Either call mark_solution_complete() OR execute_full_solution(code="..."). 
    Do NOT just return code in markdown - you must invoke a tool!
    """)
                
                messages = [system_prompt, competition_description] + list(state["messages"])
            
            nonlocal worst_case_info
            with get_openai_callback() as cb:
                response = model.invoke(messages)
                if cb.total_tokens > worst_case_info['total_tokens']:
                    worst_case_info.update({
                        'total_tokens': cb.total_tokens,
                        'prompt_tokens': cb.prompt_tokens,
                        'prompt_tokens_cached': cb.prompt_tokens_cached,
                        'completion_tokens': cb.completion_tokens,
                        'total_cost': cb.total_cost
                    })
            
            # Log generation
            log_trajectory_step(state, 'solution_generation', {
                'action': 'generated_solution' if iteration == 0 else 'reflected_and_regenerated',
                'has_tool_calls': hasattr(response, 'tool_calls') and len(response.tool_calls) > 0,
                'tokens_used': cb.total_tokens,
                'cost': cb.total_cost
            })
            
            return {"messages": [response]}
        
        def should_continue(state: AgentState) -> str:
            """Determine whether to continue or end."""
            messages = state["messages"]
            last_message = messages[-1]
            
            # Check if solution marked complete
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                for tool_call in last_message.tool_calls:
                    if tool_call['name'] == 'mark_solution_complete':
                        return "end"
                return "execute"
            
            # Check max iterations
            current_iter = state.get('current_iteration', 0)
            max_iter = state.get('max_iterations', 10)
            
            if current_iter >= max_iter:
                if verbose:
                    print(f"[TRAJ-{trajectory_id}] Maximum iterations ({max_iter}) reached. Stopping.")
                return "end"
            
            return "end"
        
        def execute_node(state: AgentState) -> AgentState:
            """Execute the generated solution code and save final successful version."""
            messages = state["messages"]
            last_message = messages[-1]
            
            # Extract code from tool calls
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                for tool_call in last_message.tool_calls:
                    if tool_call['name'] == 'execute_full_solution':
                        code = tool_call['args'].get('code', '')
                        iteration = state.get('current_iteration', 0)
                        
                        log_trajectory_step(state, 'execution_start', {
                            'action': 'executing_solution',
                            'iteration': iteration,
                            'code_length': len(code),
                            'code_preview': code[:300] + "..." if len(code) > 300 else code
                        })
                        
                        # Execute using ToolNode
                        standard_tool_node = ToolNode(tools=tools)
                        result = standard_tool_node.invoke(state)
                        
                        if isinstance(result, dict) and 'messages' in result:
                            state['messages'].extend(result['messages'])
                        
                        # Log execution result
                        execution_output = str(state['messages'][-1].content)
                        is_success = '✓ CODE EXECUTED SUCCESSFULLY' in execution_output
                        
                        log_trajectory_step(state, 'execution_complete', {
                            'action': 'execution_finished',
                            'success': is_success,
                            'iteration': iteration,
                            'output_preview': execution_output[:500] + "..." if len(execution_output) > 500 else execution_output
                        })
                        
                        # Save only final successful code
                        if is_success:
                            final_code_filename = os.path.join(save_directory, f"solution_final_{mid}.py")
                            with open(final_code_filename, 'w') as f:
                                f.write(code)
                            if verbose:
                                print(f"[TRAJ-{trajectory_id}] Final solution saved to: {final_code_filename}")
                        
                        # Increment iteration
                        state['current_iteration'] = state.get('current_iteration', 0) + 1
                        
                        return state
            
            # If mark_solution_complete was called
            standard_tool_node = ToolNode(tools=tools)
            result = standard_tool_node.invoke(state)
            if isinstance(result, dict) and 'messages' in result:
                state['messages'].extend(result['messages'])
            
            state['solution_approved'] = True
            return state
        
        # Build graph
        graph = StateGraph(AgentState)
        
        graph.add_node("generate", generate_solution)
        graph.add_node("execute", execute_node)
        
        graph.set_entry_point("generate")
        graph.add_conditional_edges(
            "generate",
            should_continue,
            {
                "execute": "execute",
                "end": END,
            },
        )
        graph.add_edge("execute", "generate")
        
        app = graph.compile()
        
        # Run the agent
        start_time = time.time()
        
        current_state = {
            "messages": [],
            "current_iteration": 0,
            "max_iterations": max_iterations,
            "solution_approved": False
        }
        
        with get_openai_callback() as cb:
            response = app.invoke(current_state, config={"recursion_limit": 250})
            interaction_info['total_tokens'] = cb.total_tokens
            interaction_info['prompt_tokens'] = cb.prompt_tokens
            interaction_info['prompt_tokens_cached'] = cb.prompt_tokens_cached
            interaction_info['completion_tokens'] = cb.completion_tokens
            interaction_info['total_cost'] = cb.total_cost
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Check if the expected output file was created
        expected_file_path = f"submissions/{competition_name}/react_code/react_code_agent_{mid}.csv"
        file_exists = os.path.exists(expected_file_path)
        
        # Save trajectory
        if 'trajectory' in response and response['trajectory']:
            log_trajectory_step(response, 'execution_summary', {
                'action': 'agent_execution_completed',
                'total_execution_time': total_time,
                'total_iterations': response.get('current_iteration', 0),
                'total_tokens': interaction_info['total_tokens'],
                'total_cost': interaction_info['total_cost'],
                'solution_approved': response.get('solution_approved', False),
                'competition_name': competition_name
            })
            
            trajectory_file = save_trajectory(
                response['trajectory'],
                save_directory,
                str(mid)
            )
        
        result = {
            'trajectory_id': trajectory_id,
            'mid': mid,
            'execution_time': total_time,
            'total_iterations': response.get('current_iteration', 0),
            'solution_approved': response.get('solution_approved', False),
            'final_response': response['messages'][-1].content if response.get('messages') else None,
            'num_messages': len(response.get('messages', [])),
            'success': file_exists,
            'error': None if file_exists else f"Expected output file not found: {expected_file_path}",
            'output_file': expected_file_path if file_exists else None,
            'interaction_info': dict(interaction_info),
            'worst_case_info': dict(worst_case_info),
        }
        
        if verbose:
            status_symbol = "✓" if file_exists else "⚠"
            status_msg = "SUCCESS" if file_exists else "NO OUTPUT FILE"
            print(f"\n[TRAJ-{trajectory_id}] {status_symbol} {status_msg} - {total_time:.2f}s - {result['total_iterations']} iterations")
            print(f"[TRAJ-{trajectory_id}] Total cost: ${result['interaction_info']['total_cost']:.4f}")
        
        return result
        
    except Exception as e:
        if verbose:
            print(f"[TRAJ-{trajectory_id}] ✗ EXCEPTION: {str(e)}")
            traceback.print_exc()
        return {
            'trajectory_id': trajectory_id,
            'mid': mid,
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def run_parallel_trajectories(num_trajectories: int, competition_name: str, max_iterations: int = 10, 
                              max_workers: int = None, verbose: bool = True):
    """
    Run multiple trajectories in parallel using multiprocessing.
    
    Args:
        num_trajectories: Number of parallel trajectories to run
        competition_name: Name of the competition to solve
        max_iterations: Maximum iterations per trajectory
        max_workers: Maximum number of worker processes
        verbose: Whether to print detailed progress information
    
    Returns:
        List of result dictionaries, one for each trajectory
    """
    if max_workers is None:
        max_workers = min(num_trajectories, mp.cpu_count())
    
    print(f"\n{'='*80}")
    print(f"RUNNING {num_trajectories} PARALLEL REACT CODE AGENT TRAJECTORIES")
    print(f"Max workers: {max_workers}")
    print(f"Competition: {competition_name}")
    print(f"Max iterations per trajectory: {max_iterations}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    results = []
    
    # Use ProcessPoolExecutor for true parallelism
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all trajectories
        future_to_trajectory = {
            executor.submit(run_single_trajectory, i, competition_name, max_iterations, verbose): i 
            for i in range(num_trajectories)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_trajectory):
            trajectory_id = future_to_trajectory[future]
            try:
                result = future.result()
                results.append(result)
                
                if not verbose:
                    if result['success']:
                        print(f"✓ Trajectory {trajectory_id} COMPLETED SUCCESSFULLY")
                    else:
                        print(f"✗ Trajectory {trajectory_id} FAILED")
                    
            except Exception as e:
                print(f"✗ Trajectory {trajectory_id} exception: {e}")
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
    
    print(f"\n{'='*80}")
    print("PARALLEL EXECUTION SUMMARY")
    print(f"{'='*80}")
    print(f"Total trajectories: {num_trajectories}")
    print(f"Successful: {len(successful_trajectories)}")
    print(f"Failed: {len(failed_trajectories)}")
    
    if successful_trajectories:
        print(f"\nSuccessful trajectory IDs: {[r['mid'] for r in successful_trajectories]}")
        avg_time = sum(r['execution_time'] for r in successful_trajectories) / len(successful_trajectories)
        avg_iterations = sum(r.get('total_iterations', 0) for r in successful_trajectories) / len(successful_trajectories)
        total_cost = sum(r['interaction_info']['total_cost'] for r in successful_trajectories)
        print(f"Average trajectory time: {avg_time:.2f}s")
        print(f"Average iterations: {avg_iterations:.1f}")
        print(f"Total cost: ${total_cost:.4f}")
        print(f"Speedup factor: {avg_time * num_trajectories / total_time:.2f}x")
    
    print(f"\nTotal parallel execution time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    
    return results


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run parallel React code agent trajectories')
    parser.add_argument('--competition', '-c', type=str, 
                       help='Competition name (overrides COMPETITION_NAME env var)')
    parser.add_argument('--trajectories', '-t', type=int,
                       help='Number of parallel trajectories (overrides NUM_TRAJECTORIES env var)')
    parser.add_argument('--max_iterations', '-i', type=int,
                       help='Max iterations per trajectory (overrides MAX_ITERATIONS env var)')
    parser.add_argument('--max_workers', '-w', type=int,
                       help='Maximum number of worker processes (overrides MAX_WORKERS env var)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Disable verbose output')
    
    args = parser.parse_args()
    
    load_dotenv()
    
    # Configuration - command line args override environment variables
    NUM_TRAJECTORIES = args.trajectories if args.trajectories is not None else int(os.environ.get('NUM_TRAJECTORIES', 4))
    MAX_WORKERS = args.max_workers if args.max_workers is not None else int(os.environ.get('MAX_WORKERS', mp.cpu_count()))
    MAX_ITERATIONS = args.max_iterations if args.max_iterations is not None else int(os.environ.get('MAX_ITERATIONS', 10))
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
    print(f"  MAX_ITERATIONS: {MAX_ITERATIONS} ({'CLI' if args.max_iterations is not None else 'ENV/DEFAULT'})")
    print(f"  COMPETITION_NAME: {COMPETITION_NAME} ({'CLI' if args.competition is not None else 'ENV/DEFAULT'})")
    print(f"  VERBOSE: {VERBOSE} ({'CLI' if args.verbose or args.quiet else 'ENV/DEFAULT'})")
    
    # Run parallel trajectories
    results = run_parallel_trajectories(
        num_trajectories=NUM_TRAJECTORIES,
        competition_name=COMPETITION_NAME,
        max_iterations=MAX_ITERATIONS,
        max_workers=MAX_WORKERS,
        verbose=VERBOSE
    )
    
    # Save results to pickle file
    log_directory = f"logs/{COMPETITION_NAME}/react_code"
    os.makedirs(log_directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(log_directory, f"react_code_agent_{NUM_TRAJECTORIES}_trajectories_{timestamp}.pkl")
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to: {results_file}")
    
    print(f"\nParallel execution completed with {len(results)} trajectories.")

