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

# Global namespace to maintain state across code executions
EXEC_NAMESPACE = {
    'pd': pd,
    'np': np,
    'plt': plt,
    'matplotlib': matplotlib,
    '__builtins__': __builtins__,
}


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    trajectory: NotRequired[list]
    current_iteration: NotRequired[int]
    max_iterations: NotRequired[int]
    solution_approved: NotRequired[bool]


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
    print(f"[TRAJECTORY] Step {trajectory_step['step_number']}: {step_type} (Iteration {state.get('current_iteration', 0)})")
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
    
    print(f"[TRAJECTORY] Saved complete trajectory to {trajectory_file}")
    return trajectory_file


@tool
def execute_full_solution(code: str) -> str:
    """
    Execute a COMPLETE Python solution for the competition and return all output.
    
    Args:
        code: Complete Python code to solve the entire competition task.
              Pre-imported: pandas as 'pd', numpy as 'np', matplotlib.pyplot as 'plt'
              The code should load data, process it, train model, and save predictions.
    
    Returns:
        String containing stdout, stderr, execution status, and any errors.
    """
    global EXEC_NAMESPACE
    
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
        # Execute the complete code
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
    """
    Call this tool when you are satisfied with your solution and believe it successfully
    completes the competition requirements (data loaded, model trained, predictions saved).
    
    Returns:
        Confirmation message.
    """
    return "Solution marked as complete. Execution will terminate."


def generate_solution(state: AgentState, model: ChatOpenAI, competition_description: HumanMessage,
                     worst_case_info: dict) -> dict:
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
        print(f"\n⚠ Maximum iterations ({max_iter}) reached. Stopping.")
        return "end"
    
    return "end"


def execute_node(state: AgentState, tools: list, save_directory: str, agent_id: str) -> AgentState:
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
                    final_code_filename = os.path.join(save_directory, f"solution_final_{agent_id}.py")
                    with open(final_code_filename, 'w') as f:
                        f.write(code)
                    print(f"\n[FINAL CODE] Successful solution saved to: {final_code_filename}")
                
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


def main():
    load_dotenv()
    
    # Setup
    competition_name = os.environ.get('COMPETITION_NAME', 'spaceship_titanic')
    model_name = os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')
    max_iterations = int(os.environ.get('MAX_ITERATIONS', '10'))
    
    mid = uuid.uuid4()
    save_directory = f"submissions/{competition_name}/react_code"
    os.makedirs(save_directory, exist_ok=True)
    
    # Initialize model
    tools = [execute_full_solution, mark_solution_complete]
    
    bind_kwargs = {}
    # Disable parallel tool calls for most models
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
    
    # Tracking
    worst_case_info = defaultdict(int)
    interaction_info = defaultdict(int)
    
    # Build graph
    graph = StateGraph(AgentState)
    
    def generate_node(state):
        return generate_solution(state, model, competition_description, worst_case_info)
    
    def execute_node_wrapper(state):
        return execute_node(state, tools, save_directory, str(mid))
    
    graph.add_node("generate", generate_node)
    graph.add_node("execute", execute_node_wrapper)
    
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
    print(f"\n{'='*80}")
    print(f"React Code Agent - Generate-Execute-Reflect Loop")
    print(f"Competition: {competition_name}")
    print(f"Model: {model_name}")
    print(f"Max Iterations: {max_iterations}")
    print(f"{'='*80}\n")
    
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
    
    # Print results
    print(f"\n{'='*80}")
    print("EXECUTION COMPLETE")
    print(f"{'='*80}")
    print(f"Total iterations: {response.get('current_iteration', 0)}")
    print(f"Solution approved: {response.get('solution_approved', False)}")
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"\nFinal response: {response['messages'][-1].content}")
    print(f"\nOverall Interaction Info: {dict(interaction_info)}")
    print(f"Worst Case Per Query Info: {dict(worst_case_info)}")
    
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
        print(f"\n✓ Trajectory saved to: {trajectory_file}")
    else:
        print("\n⚠ No trajectory data found to save")


if __name__ == "__main__":
    main()
