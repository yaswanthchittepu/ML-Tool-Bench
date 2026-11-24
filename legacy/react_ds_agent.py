import feature_engineering as fe
import hyperparameter_tuning as hpt
import evaluation as ev
import mle_utils as ut
import modeling as md
import loading as ld
from func_to_tool_decorator import make_get_and_set_tool, make_get_tool, make_set_tool, override_tool
from collections import defaultdict
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
import time
from dotenv import load_dotenv
import sys, ipdb, os, json
import inspect
import uuid
from datetime import datetime

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    scratchpad: NotRequired[Dict[str, Any]]
    plan: NotRequired[str]
    summary: NotRequired[RunningSummary]
    num_tool_calls: NotRequired[int]
    trajectory: NotRequired[list]

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

if __name__ == "__main__":

    load_dotenv()

    # allowed_functions = set(['read_data', 'create_conditional_feature', 'concatenate_train_test', 'fillna_with_median', 'fillna_with_mode',
    #     'extract_string_pattern', 'split_string_column', 'create_group_aggregation', 'drop_feature', 'cast_columns',
    #     'normalize_all_numerical_columns', 'encode_all_categorical_columns', 'split_combined_into_train_test',
    #     'drop_feature', 'get_features', 'tune_catboost_classifier', 'tune_random_forest_classifier',
    #     'tune_lightgbm_classifier', 'tune_logistic_classifier', 'tune_xgboost_classifier', 'predict_target', 
    #     'rename_feature', 'concatenate_dataframes', 'save_dataframe_to_csv'
    # ])

    # print(len(allowed_functions))
    # sys.exit()

    
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
            try:
                # Get the original function
                original_func = getattr(module, func_name)
                
                # Get the decorator function
                decorator_func = decorator_func_map[decorator_name]
                
                # Apply the decorator
                # verbose is a dummy arg for now
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


    # funcs = list(all_decorated_functions.values())
    # func = tool(funcs[0])
    # print(f" \n\n\n\n {func.name}, {func.description}, {func.args} \n\n\n\n")

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

    # print(tool_functions['read_data'].name, tool_functions['read_data'].description, tool_functions['read_data'].args)
    # sys.exit()


    # ------------------------------------------------------------
    # Model Definition and Binding Tools to Model
    # ------------------------------------------------------------

    tools = list(tool_functions.values())

    bind_kwargs = {'parallel_tool_calls': False} if('o3' not in os.environ['OPENAI_MODEL'] and 'claude' not in os.environ['OPENAI_MODEL']) else {}
    # bind_kwargs = {'parallel_tool_calls': False} 
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


    # ------------------------------
    # Graph Definition
    # ------------------------------
    mid = uuid.uuid4()
    save_directory = f"submissions/{os.environ['COMPETITION_NAME']}/react"
    os.makedirs(save_directory, exist_ok=True)
    os.makedirs(f"model_saves/{os.environ['COMPETITION_NAME']}/", exist_ok=True)
    competition_description = HumanMessage(content=CHALLENGES[os.environ['COMPETITION_NAME']].format(save_directory=save_directory, save_file_name=f"react_agent_{mid}.csv"))
    worst_case_per_query_info = defaultdict(int)
    # You are only allowed a maximum of 25 tool calls to solve the problem.

    # The csv should have the following columns: PassengerId - Id for each passenger in the test set.
    # Transported - The predicted value. For each passenger, predict either True or False. Name the file as react_agent_{model_name}.csv. Look at the sample_submission.csv file in the data folder for an example of the format the results should be in.


    def plan_call(state: AgentState) -> AgentState:
        """Create a high-level plan for the entire ML pipeline."""
        if(state.get("plan") is not None):
            return state
            
        plan_prompt = SystemMessage(content="""
        You are a Data Scientist tasked with solving the Kaggle competition provided below. Create a high level textual plan of what steps you will take to solve the problem.
        """)
        
        # # Log the planning step initiation
        # state = log_trajectory_step(state, 'plan_initiation', {
        #     'action': 'starting_planning_phase',
        #     'competition_name': os.environ.get('COMPETITION_NAME', 'unknown'),
        #     'prompt_sent': plan_prompt.content
        # })
        
        global worst_case_per_query_info
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
    #     summarization_system_prompt = SystemMessage(content="""
    #         Create a summary of the messages in the state. The summary should contain information about the tools called at each time step, along with the any relevant information about the tool call.
    #     """)
    #     # import ipdb; ipdb.set_trace()

    #     if(state.get('messages') is None or (len(state['messages']) == 1 and state['messages'][0].content == '')):
    #         response = model.invoke([system_prompt] + [competition_description] + [f"\n\nThe plan to solve the problem is: {state['plan']}\n\n"] + state["messages"])
    #         state_update = {"messages": [response]}
    #     else:
    #         # summarization_result = summarize_messages(
    #         #     state["messages"],
    #         #     running_summary=state.get("summary"),
    #         #     model=model,
    #         #     max_tokens=25000,
    #         #     # max_tokens_before_summary=256,
    #         #     max_summary_tokens=20000,
    #         # )
    #         summarization_result = summarization_model.invoke([summarization_system_prompt] + state["messages"])
    #         response = model.invoke([system_prompt] + [competition_description] + [f"\n\nThe plan to solve the problem is: {state['plan']}\n\n"] + [summarization_result])
    #         state_update = {"messages": [response], 'summary': summarization_result.content}
    #         # state_update["summary"] = summarization_result.running_summary
    #     return state_update

    def model_call(state:AgentState) -> AgentState:
        system_prompt = SystemMessage(content=
            "You are a Data Scientist tasked with solving the Kaggle competition provided below. "
        )
        
        # Log the reasoning step initiation
        # log_trajectory_step(state, 'reasoning_initiation', {
        #     'action': 'starting_reasoning_phase',
        #     'current_message_count': len(state.get("messages", [])),
        #     'model_type': 'llama' if 'llama' in os.environ['OPENAI_MODEL'] else 'gpt',
        #     'has_plan': 'plan' in state and state['plan'] is not None
        # })
        
        global worst_case_per_query_info
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
        
        # # Extract tool information for logging
        # tool_calls_info = []
        # if hasattr(response, 'tool_calls') and response.tool_calls:
        #     for tool_call in response.tool_calls:
        #         tool_calls_info.append({
        #             'tool_name': tool_call['name'],
        #             'tool_args': tool_call['args'],
        #             'tool_id': tool_call['id']
        #         })
        
        # Log the reasoning step completion 
        # Might need to change response_content for GPT models
        # log_trajectory_step(state, 'reasoning_completion', {
        #     'action': 'reasoning_completed',
        #     'response_content': response.content,
        #     'has_tool_calls': len(tool_calls_info) > 0,
        #     'tool_calls': tool_calls_info,
        #     'tokens_used': cb.total_tokens,
        #     'cost': cb.total_cost
        # })
        
        return {"messages": [response]}

    # For llama
    # {'messages': [AIMessage(content='{"name": "read_data", "parameters": {"output": "train_data", "func_kwargs": "{\'filepath\': \'data/spaceship_titanic/train.csv\'}"}}', 
    # additional_kwargs={'refusal': None}, response_metadata={'token_usage': {'completion_tokens': 39, 'prompt_tokens': 28837, 'total_tokens': 28876, 
    # 'completion_tokens_details': None, 'prompt_tokens_details': None}, 'model_name': 'hosted_vllm/meta-llama/Meta-Llama-3.1-70B-Instruct', 
    # 'system_fingerprint': None, 'id': 'chatcmpl-1afa62c4721d4aa2974ad0ed331025e0', 'service_tier': None, 'finish_reason': 'stop', 'logprobs': None},
    #  id='run--337715b2-31f2-42ca-83ae-1d3b25048a6a-0', usage_metadata={'input_tokens': 28837, 'output_tokens': 39, 'total_tokens': 28876, 'input_token_details': {}, 
    # 'output_token_details': {}})]}

    # For GPT
    # {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_zqliYvQJRVIl7lMVjEVPFfQK', 
    # 'function': {'arguments': '{"func_kwargs":{"filepath":"data/spaceship_titanic/train.csv"},"output":"train_data"}', 'name': 'read_data'}, 'type': 'function'}], 'refusal': None}, 
    # response_metadata={'token_usage': {'completion_tokens': 32, 'prompt_tokens': 19732, 'total_tokens': 19764, 
    # 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 
    # 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4.1-mini-2025-04-14', 
    # 'system_fingerprint': 'fp_3dcd5944f5', 'id': 'chatcmpl-CRPl7pusUdMUcAXf0ckO4mtz7hUoZ', 'service_tier': None, 'finish_reason': 'tool_calls', 'logprobs': None}, 
    # id='run--6a88225b-5044-4313-8f80-4d19aed66427-0', 
    # tool_calls=[{'name': 'read_data', 'args': {'func_kwargs': {'filepath': 'data/spaceship_titanic/train.csv'}, 'output': 'train_data'}, 
    # 'id': 'call_zqliYvQJRVIl7lMVjEVPFfQK', 'type': 'tool_call'}], 
    # usage_metadata={'input_tokens': 19732, 'output_tokens': 32, 'total_tokens': 19764, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}})], 
    # plan: <PLAN>}
    
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
        

    graph = StateGraph(AgentState)
    
    # Custom ToolNode wrapper to handle both Llama and GPT formats
    def custom_tool_node(state: AgentState) -> AgentState:
        """Custom tool node that handles both Llama JSON format and GPT tool_calls format"""

        # ipdb.set_trace()
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
                    tool_execution_info['parsing_error'] = str(e)

        # import ipdb; ipdb.set_trace()
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
        
        # Use standard ToolNode and capture results
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


    app = graph.compile()


    # current_state = {"messages": []}

    # Ensure that you do not create too many features, such that the context window of the LLM fills up

    # Lets go step by step, perform an operation. Describe what you are doing to me, and wait for my response before performing the next operation.

    # Give me the evaluation resuls, that show the goodness of the model fit to the data. Also save the model to the model_saves folder.

    current_state = {"messages": []}
    
    # Record start time
    start_time = time.time()
    print(f"Starting React DS agent execution at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    
    # Run the app
    interaction_info = defaultdict(int)
    with get_openai_callback() as cb:
        response = app.invoke(current_state, config={"recursion_limit": 250})
        interaction_info['total_tokens'] += cb.total_tokens
        interaction_info['prompt_tokens'] += cb.prompt_tokens
        interaction_info['prompt_tokens_cached'] += cb.prompt_tokens_cached
        interaction_info['completion_tokens'] += cb.completion_tokens
        interaction_info['total_cost'] += cb.total_cost
    # Record end time and calculate duration
    end_time = time.time()
    total_time = end_time - start_time
    
    # Show assistant message
    print(f"\n\n\n {response} \n\n\n")
    print(f"Assistant: {response['messages'][-1].content}")

    print(f"\nExecution completed at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Overall Interaction Information: {interaction_info}")
    print(f"Worst Case Per Query Information: {worst_case_per_query_info}")
    
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

    # while True:
        
    #     user_input = input("\nYou: ")
    #     if user_input.lower() in ['quit', 'exit', 'q']:
    #         print("Goodbye!")
    #         break
    #     if user_input.lower() in ['f']:
    #         print("Useless to interact with such a braindead agent!")
    #         break

    #     # Add the new user message to the ongoing state
    #     current_state["messages"].append(HumanMessage(content=user_input))
        
    #     # Run the app
    #     response = app.invoke(current_state, config={"recursion_limit": 250})

    #     # Update the full state
    #     current_state = response

    #     # Show assistant message
    #     print(f"\n\n\n {response} \n\n\n")
    #     print(f"Assistant: {response['messages'][-1].content}")