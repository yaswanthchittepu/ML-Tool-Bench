import math
import random
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
import time
from datetime import datetime

import ipdb
from dotenv import load_dotenv
load_dotenv()

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


# class AgentState(TypedDict):
#     messages: Annotated[Sequence[BaseMessage], add_messages]
#     scratchpad: NotRequired[Dict[str, Any]]
#     plan: NotRequired[str]
#     summary: NotRequired[RunningSummary]
#     num_tool_calls: NotRequired[int]


if __name__ == "__main__":

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
    
    # Change this to work with a simpler problem for testing
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

    # Change this to work with a simpler problem for testing
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a Data Scientist tasked with solving the Kaggle competition provided below, with the tools available to you. Propose tool candidates that would help solving the problem at the current stage, given the existing context available to you.",
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

    # initial_response = initial_answer_chain.invoke(
    #     {"input": "Write a research report on lithium pollution."}
    # )

    ### IN our case, this has to be a pass through function
    def generate_initial_response(state: TreeState) -> dict:
        return state

    # Define the node we will add to the graph
    # def generate_initial_response(state: TreeState) -> dict:
    #     """Generate the initial candidate response."""
    #     ipdb.set_trace()
    #     res = initial_answer_chain.invoke({"input": state["input"]})
    #     parsed = parser.invoke(res)
    #     # parsed = [parsed[0]]
    #     # parsed = [{'args': {'output': 'train_data', 'func_kwargs': {'filepath': 'data/train.csv'}}, 'id': parsed[0]['id'], 'type': 'read_data'}]
    #     # {   
    #     #     **state,
    #     #     'root': Node(messages=[]),
    #     #     "messages": [
    #     #         AIMessage(
    #     #             content="",
    #     #             tool_calls=[
    #     #                 {"name": rrr["type"], "args": rrr["args"], "id": rrr["id"]}
    #     #             ],
    #     #         )
    #     #     ],
    #     # }
    #     tool_responses = [
    #         tool_node.invoke(
    #             {   
    #                 **state,
    #                 'root': Node(messages=[]),
    #                 "messages": [
    #                     AIMessage(
    #                         content="",
    #                         tool_calls=[
    #                             {"name": r["type"], "args": r["args"], "id": r["id"]}
    #                         ],
    #                     )
    #                 ],
    #             }
    #         )
    #         for r in parsed
    #     ]
    #     output_messages = [res] + [tr["messages"][0] for tr in tool_responses]
    #     reflection = reflection_chain.invoke(
    #         {"input": state["input"], "candidate": output_messages}
    #     )
    #     root = Node(output_messages, reflection=reflection)
    #     return {
    #         **state,
    #         "root": root,
    #     }
    
    # This generates N candidate values
    # for a single input to sample actions from the environment


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

    def select(root: Node) -> dict:
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
        
        # ipdb.set_trace()
        global interaction_info

        # ipdb.set_trace()

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
        # parsed = parser.batch(new_candidates)
        # Need to make input to parser a list and take the first element
        if('llama' in os.environ['OPENAI_MODEL']):
            new_candidates = parse_llama_json({'messages': [new_candidates]})['messages'][0]
        parsed = parser.batch([new_candidates])
        print(f"\n {parsed} \n")
        # YC: We need to add a remove duplicates function here
        # flattened = [
        #     (i, tool_call)
        #     for i, tool_calls in enumerate(parsed)
        #     for tool_call in tool_calls
        # ]
        # YC: need to change how we id tool calls
        flattened = [
            (i, tool_call)
            for i, tool_call in enumerate(parsed[0])
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

        print(f"\n {tool_responses} \n")
        # If successful, use the Commmand to update the state
        # We only deal with scratchpad updates here
        # All the following updates can be merged in a single loop for efficiency ## Future work
        new_node_scratchpads = {}
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

        # collected_responses = defaultdict(list)
        # for i, resp in tool_responses:
        #     collected_responses[i].append(resp["messages"])
        output_messages = []
        # for i, candidate in enumerate(new_candidates):
        #     output_messages.append([candidate] + collected_responses[i])

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

        # Reflect on each candidate with retry logic
        # For tasks with external validation, you'd add that here.
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
                break  # Success, exit retry loop
            except Exception as e:
                print(f"Error in reflection_chain.batch (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:  # Last attempt
                    raise e  # Re-raise the exception after all retries exhausted
                else:
                    print(f"Retrying reflection... (attempt {attempt + 2}/{max_retries})")
                    continue
        
        # Grow tree
        child_nodes = [
                Node(cand, parent=best_candidate, reflection=reflection, scratchpad=new_node_scratchpads.get(i, {}))
                for i, (cand, reflection) in enumerate(zip(output_messages, reflections))
            ]
        best_candidate.children.extend(child_nodes)
        # We have already extended the tree directly, so we just return the state
        return state

    def should_loop(state: TreeState):
        """Determine whether to continue the tree search."""
        root = state["root"]
        # # If there is a solved node, we stop the tree search
        # solved_nodes = root.get_solved_nodes()
        # if(len(solved_nodes) > 0):
        #     return END
        if root.is_solved:
            return END
        if root.height > int(os.environ.get('MAX_TREE_DEPTH', '50')):
            return END
        return "expand"


    builder = StateGraph(TreeState)
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
    save_directory = f"submissions/{os.environ['COMPETITION_NAME']}/lats_reflection"
    question = CHALLENGES[os.environ['COMPETITION_NAME']].format(save_directory=save_directory, save_file_name=f"lats_reflection_agent_{mid}.csv")
    os.makedirs(save_directory, exist_ok=True)
    os.makedirs(f"model_saves/{os.environ['COMPETITION_NAME']}/", exist_ok=True)
    # Most Importantly, save your predictions for the test data, as a csv, to the submissions folder. The csv should have the following columns: PassengerId - Id for each passenger in the test set.
    # Transported - The predicted value. For each passenger, predict either True or False. Name the file as react_agent_{model_name}.csv. Look at the sample_submission.csv file in the data folder for an example of the format the results should be in.

    # You are only allowed a maximum of 20 tool calls in a trajectory to solve the problem.
    last_step = None
    tree_node = {'input': question}
    root_node = Node(messages=[], tree_node=tree_node)
    tree_node['root'] = root_node
    
    # Record start time
    start_time = time.time()
    print(f"Starting LATS reflection search at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    
    # for step in graph.stream({"input": question}):
    for step in graph.stream(tree_node, config={"recursion_limit": 500}):
        last_step = step
        step_name, step_state = next(iter(step.items()))
        print(step_name)
        print("rolled out: ", step_state["root"].height)
        print("---")
    
    # Record end time and calculate duration
    end_time = time.time()
    total_time = end_time - start_time

    # import ipdb; ipdb.set_trace()
    
    print(f"\nSearch completed at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Total search time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Interaction info: {interaction_info}")
    print(f"Worst case per query info: {worst_case_per_query_info}")


    # Get best node - either solution or highest valued leaf if no solution found
    root_node = last_step["expand"]["root"]
    solution_node = root_node.get_best_solution()
    
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
    
    print("="*60)

    # question = "Generate a table with the average size and weight, as well as the oldest recorded instance for each of the top 5 most common birds."
    # last_step = None
    # for step in graph.stream({"input": question}):
    #     last_step = step
    #     step_name, step_state = next(iter(step.items()))
    #     print(step_name)
    #     print("rolled out: ", step_state["root"].height)
    #     print("---")

    # solution_node = last_step["expand"]["root"].get_best_solution()
    # best_trajectory = solution_node.get_trajectory(include_reflections=False)
    # print(best_trajectory[-1].content)

    # question = "Write out magnus carlson series of moves in his game against Alireza Firouzja and propose an alternate strategy"
    # last_step = None
    # for step in graph.stream({"input": question}):
    #     last_step = step
    #     step_name, step_state = next(iter(step.items()))
    #     print(step_name)
    #     print("rolled out: ", step_state["root"].height)
    #     print("---")

    # solution_node = last_step["expand"]["root"].get_best_solution()
    # best_trajectory = solution_node.get_trajectory(include_reflections=False)
    # print(best_trajectory[-1].content)