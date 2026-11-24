from pydantic import BaseModel, Field
import math

from collections import deque, defaultdict
from typing import Optional
from typing_extensions import TypedDict
from typing import Literal
from typing import Annotated, Sequence, NotRequired, Any, Union, Dict

from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage, SystemMessage # The foundational class for all message types in LangGraph
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command
from langgraph.graph.message import add_messages
from langgraph.prebuilt import InjectedState

import pandas as pd
import numpy as np
import ipdb


class Reflection(BaseModel):
    reflections: str = Field(
        description="The critique and reflections on the sufficiency, superfluency,"
        " and general quality of the response"
    )
    score: int = Field(
        description="Score from 0-10 on the quality of the candidate response.",
        gte=0,
        lte=10,
    )
    found_solution: bool = Field(
        description="Whether the response has fully solved the question or task."
    )

    def as_message(self):
        return HumanMessage(
            content=f"Reasoning: {self.reflections}\nScore: {self.score}"
        )

    @property
    def normalized_score(self) -> float:
        return self.score / 10.0


class Node:
    # We add tree_node as an attribute for quick access to scratchpad and potentially other useful information in the TreeNode
    def __init__(
        self,
        messages: list[BaseMessage],
        scratchpad: Dict[str, Any] = {},
        reflection: Optional[Reflection] = None,
        parent: Optional["Node"] = None,
        tree_node: Optional["TreeState"] = None
    ):
        self.messages = messages
        self.parent = parent
        self.tree_node = tree_node
        self.scratchpad = scratchpad
        self.children = []
        self.value = 0
        self.visits = 0
        self.reflection = reflection
        self.depth = parent.depth + 1 if parent is not None else 1
        if(self.reflection is not None):
            self._is_solved = reflection.found_solution
            if self._is_solved:
                self._mark_tree_as_solved()
            self.backpropagate(reflection.normalized_score)
        else:
            self._is_solved = False


    def __repr__(self) -> str:
        if(self.reflection is not None):
            return (
                f"<Node value={self.value}, visits={self.visits},"
                f" solution={self.messages} reflection={self.reflection}/>"
            )
        else:
            return (
                f"<Node value={self.value}, visits={self.visits},"
                f" solution={self.messages}, reflection is None/>"
            )

    @property
    def is_solved(self):
        """If any solutions exist, we can end the search."""
        return self._is_solved

    @property
    def is_terminal(self):
        return not self.children

    @property
    def best_child_score(self):
        """Return the child with the highest value."""
        if not self.children:
            return None
        return max(self.children, key=lambda child: int(child.is_solved) * child.value)

    @property
    def height(self) -> int:
        """Check for how far we've rolled out the tree."""
        if self.children:
            return 1 + max([child.height for child in self.children])
        return 1

    def upper_confidence_bound(self, exploration_weight=1.0):
        """Return the UCT score. This helps balance exploration vs. exploitation of a branch."""
        if self.parent is None:
            raise ValueError("Cannot obtain UCT from root node")
        if self.visits == 0:
            return self.value
        # Encourages exploitation of high-value trajectories
        average_reward = self.value / self.visits
        # Encourages exploration of less-visited trajectories
        exploration_term = math.sqrt(math.log(self.parent.visits) / self.visits)
        return average_reward + exploration_weight * exploration_term

    def backpropagate(self, reward: float):
        """Update the score of this node and its parents."""
        node = self
        while node:
            node.visits += 1
            node.value = (node.value * (node.visits - 1) + reward) / node.visits
            node = node.parent

    def get_messages(self, include_reflections: bool = True):
        if include_reflections:
            return self.messages + ([self.reflection.as_message()] if(self.reflection) else [])
        return self.messages

    def get_trajectory(self, include_reflections: bool = True) -> list[BaseMessage]:
        """Get messages representing this search branch."""
        messages = []
        node = self
        while node:
            messages.extend(
                node.get_messages(include_reflections=include_reflections)[::-1]
            )
            node = node.parent
        # Reverse the final back-tracked trajectory to return in the correct order
        return messages[::-1]  # root solution, reflection, child 1, ...

    def get_trajectory_scratchpad(self):
        """Combine the node's scratchpad along the trajectory to root. The combined scratchpad is available to the next node"""
        node = self
        lst = []
        scratchpad = {}
        while node:
            # scratchpad.update(node.scratchpad)
            lst.append(node.scratchpad)
            node = node.parent
        rev_lst = lst[::-1]
        for d in rev_lst:
            scratchpad.update(d)
        return scratchpad

    def _get_all_children(self):
        all_nodes = []
        nodes = deque()
        nodes.append(self)
        while nodes:
            node = nodes.popleft()
            all_nodes.extend(node.children)
            for n in node.children:
                nodes.append(n)
        return all_nodes

    def get_best_solution(self):
        """Return the best solution from within the current sub-tree."""
        all_nodes = [self] + self._get_all_children()
        best_node = max(
            all_nodes,
            # We filter out all non-terminal, non-solution trajectories
            key=lambda node: int(node.is_terminal and node.is_solved) * node.value,
        )
        return best_node

    def _mark_tree_as_solved(self):
        parent = self.parent
        while parent:
            parent._is_solved = True
            parent = parent.parent

    def get_solved_nodes(self):
        solved_nodes = []
        nodes = deque()
        nodes.append(self)
        while nodes:
            node = nodes.popleft()
            if(node.is_solved):
                solved_nodes.append(node)
            for n in node.children:
                nodes.append(n)
        return solved_nodes


class TreeState(TypedDict):
    # The full tree
    messages: Annotated[Sequence[BaseMessage], add_messages]
    root: NotRequired[Node]
    input: NotRequired[str]
    scratchpad: NotRequired[Dict[str, Any]]
    # plan: NotRequired[str]
    # summary: NotRequired[RunningSummary]
    num_tool_calls: NotRequired[int]


class Node_Verifiable_Rewards:
    # We add tree_node as an attribute for quick access to scratchpad and potentially other useful information in the TreeNode
    def __init__(
        self,
        messages: list[BaseMessage],
        scratchpad: Dict[str, Any] = {},
        parent: Optional["Node"] = None,
        tree_node: Optional["TreeState_Verifiable_Rewards"] = None,
        reward: Optional[Dict[str, Any]] = defaultdict(float)
    ):
        self.messages = messages
        self.parent = parent
        self.tree_node = tree_node
        self.scratchpad = scratchpad
        self.children = []
        self.value = 0
        self.visits = 0
        self.depth = parent.depth + 1 if parent is not None else 1
        # reward is the reward recieved by the current node
        # trajectory reward is the sum of rewards from all nodes along the trajectory
        # Penalizing the depth of the tree is done separately
        # Net reward is trajectory reward - K * depth
        # Lets choose K = 0.1
        self.reward = reward
        # self.reward is either empty {} or a single element dictionary
        rew = -0.1 # penalty per depth
        stage = None
        if(self.reward):
            stage, r = list(self.reward.items())[0]
            rew += r

        all_stages = [
            'train_data_loading',
            'test_data_loading',
            'combine_train_test',
            'data_cleaning',
            'feature_engineering',
            'split_train_test',
            'train_data_to_features_target',
            'test_data_to_features',
            'modeling',
            'create_submission_dataframe',
            # 'create_submission_file',
        ]

        if(stage is None):
            # Use the last stage in the trajectory
            node = self
            while node:
                if(node.reward):
                    # stage = max(node.reward, key=lambda k: all_stages.index(k))
                    stage, _ = list(node.reward.items())[0]
                    break
                node = node.parent
        # assert stage is not None, "No stage found in the trajectory"
        if(stage is None):
            stage = 'init'
        # How do we know if the user query is solved, without access to reflection?
        # We check if create_submission is done and there is a valid submission file saved to disk / scratchpad contains valid submission style dataframe
        if(self.reward['create_submission_dataframe'] > 0):
            self._is_solved = True
            # This is so we dont stop exploring once we have a valid trajectory
            # self._mark_tree_as_solved()
        else:
            self._is_solved = False

        self.backpropagate(rew)


    def __repr__(self) -> str:
        return (
            f"<Node value={self.value}, visits={self.visits},"
            f" solution={self.messages}/>"
        )

    @property
    def is_solved(self):
        """If any solutions exist, we can end the search."""
        return self._is_solved

    @property
    def is_terminal(self):
        return not self.children

    @property
    def best_child_score(self):
        """Return the child with the highest value."""
        if not self.children:
            return None
        return max(self.children, key=lambda child: int(child.is_solved) * child.value)

    @property
    def height(self) -> int:
        """Check for how far we've rolled out the tree."""
        if self.children:
            return 1 + max([child.height for child in self.children])
        return 1

    def upper_confidence_bound(self, exploration_weight=1.0):
        """Return the UCT score. This helps balance exploration vs. exploitation of a branch."""
        if self.parent is None:
            raise ValueError("Cannot obtain UCT from root node")
        if self.visits == 0:
            return self.value
        # Encourages exploitation of high-value trajectories
        average_reward = self.value / self.visits
        # Encourages exploration of less-visited trajectories
        exploration_term = math.sqrt(math.log(self.parent.visits) / self.visits)
        return average_reward + exploration_weight * exploration_term

    def backpropagate(self, reward: float):
        """Update the score of this node and its parents."""
        node = self
        while node:
            node.visits += 1
            node.value = (node.value * (node.visits - 1) + reward) / node.visits
            node = node.parent

    def get_messages(self):
        return self.messages

    def get_trajectory(self, include_reflections: bool = True) -> list[BaseMessage]:
        """Get messages representing this search branch."""
        messages = []
        node = self
        while node:
            messages.extend(
                node.get_messages()[::-1]
            )
            node = node.parent
        # Reverse the final back-tracked trajectory to return in the correct order
        return messages[::-1]  # root solution, reflection, child 1, ...

    def get_trajectory_reward(self):
        rewards = defaultdict(float)
        node = self
        while node:
            rewards.update(node.reward)
            node = node.parent
        return rewards

    def get_trajectory_scratchpad(self):
        """Combine the node's scratchpad along the trajectory to root. The combined scratchpad is available to the next node"""
        node = self
        lst = []
        scratchpad = {}
        while node:
            # scratchpad.update(node.scratchpad)
            lst.append(node.scratchpad)
            node = node.parent
        rev_lst = lst[::-1]
        for d in rev_lst:
            scratchpad.update(d)
        return scratchpad

    def get_solved_nodes(self):
        solved_nodes = []
        nodes = deque()
        nodes.append(self)
        while nodes:
            node = nodes.popleft()
            if(node.is_solved):
                solved_nodes.append(node)
            for n in node.children:
                nodes.append(n)
        return solved_nodes

    def _get_all_children(self):
        all_nodes = []
        nodes = deque()
        nodes.append(self)
        while nodes:
            node = nodes.popleft()
            all_nodes.extend(node.children)
            for n in node.children:
                nodes.append(n)
        return all_nodes

    def get_best_solution(self):
        """Return the best solution from within the current sub-tree."""
        all_nodes = [self] + self._get_all_children()
        best_node = max(
            all_nodes,
            # We filter out all non-terminal, non-solution trajectories
            key=lambda node: int(node.is_terminal and node.is_solved) * node.value,
        )
        return best_node

    def _mark_tree_as_solved(self):
        parent = self.parent
        while parent:
            parent._is_solved = True
            parent = parent.parent

class TreeState_Verifiable_Rewards(TypedDict):
    # The full tree
    messages: Annotated[Sequence[BaseMessage], add_messages]
    root: NotRequired[Node_Verifiable_Rewards]
    input: NotRequired[str]
    scratchpad: NotRequired[Dict[str, Any]]
    # plan: NotRequired[str]
    # summary: NotRequired[RunningSummary]
    num_tool_calls: NotRequired[int]

class Node_Verifiable_Rewards_Rollout:
    # We add tree_node as an attribute for quick access to scratchpad and potentially other useful information in the TreeNode
    def __init__(
        self,
        messages: list[BaseMessage],
        scratchpad: Dict[str, Any] = {},
        parent: Optional["Node"] = None,
        tree_node: Optional["TreeState_Verifiable_Rewards"] = None,
        reward: Optional[float] = 0.,
        stage_solved: Optional[str] = None,
        rollout_depth: Optional[int] = 3,
        rollout_policy: Optional[str] = 'random',
    ):
        self.messages = messages
        self.parent = parent
        self.tree_node = tree_node
        self.scratchpad = scratchpad
        self.children = []
        self.value = 0
        self.visits = 0
        self.depth = parent.depth + 1 if parent is not None else 1
        self.rollout_depth = rollout_depth
        self.rollout_policy = rollout_policy
        # reward is the rollout cumulative reward recieved by the current node
        # reward is a scalar value
        # since while doing rollout, we need to account for depth penalty
        if(stage_solved == 'init'):
            self.reward = reward
        else:
            self.reward = reward - (0.1 * rollout_depth)

        all_stages = [
            'train_data_loading',
            'test_data_loading',
            'combine_train_test',
            'data_cleaning',
            'feature_engineering',
            'split_train_test',
            'train_data_to_features_target',
            'test_data_to_features',
            'modeling',
            'create_submission_dataframe',
            # 'create_submission_file',
        ]

        self.stage_solved = stage_solved

        # How do we know if the user query is solved, without access to reflection?
        # We check if create_submission is done and there is a valid submission file saved to disk / scratchpad contains valid submission style dataframe
        if(self.stage_solved == 'create_submission_dataframe'):
            self._is_solved = True
            # This is so we dont stop exploring once we have a valid trajectory
            # self._mark_tree_as_solved()
        else:
            self._is_solved = False

        self.backpropagate(self.reward)


    def __repr__(self) -> str:
        return (
            f"<Node value={self.value}, visits={self.visits},"
            f" solution={self.messages}/>"
        )
    @property
    def is_solved(self):
        """If any solutions exist, we can end the search."""
        return self._is_solved

    @property
    def is_terminal(self):
        return not self.children

    @property
    def best_child_score(self):
        """Return the child with the highest value."""
        if not self.children:
            return None
        return max(self.children, key=lambda child: int(child.is_solved) * child.value)

    @property
    def height(self) -> int:
        """Check for how far we've rolled out the tree."""
        if self.children:
            return 1 + max([child.height for child in self.children])
        return 1

    def upper_confidence_bound(self, exploration_weight=1.0):
        """Return the UCT score. This helps balance exploration vs. exploitation of a branch."""
        if self.parent is None:
            raise ValueError("Cannot obtain UCT from root node")
        if self.visits == 0:
            return self.value
        # Encourages exploitation of high-value trajectories
        average_reward = self.value / self.visits
        # Encourages exploration of less-visited trajectories
        exploration_term = math.sqrt(math.log(self.parent.visits) / self.visits)
        return average_reward + exploration_weight * exploration_term

    def backpropagate(self, reward: float):
        """Update the score of this node and its parents."""
        node = self
        while node:
            node.visits += 1
            node.value = (node.value * (node.visits - 1) + reward) / node.visits
            node = node.parent

    def get_messages(self):
        return self.messages

    def get_trajectory(self, include_reflections: bool = True) -> list[BaseMessage]:
        """Get messages representing this search branch."""
        messages = []
        node = self
        while node:
            messages.extend(
                node.get_messages()[::-1]
            )
            node = node.parent
        # Reverse the final back-tracked trajectory to return in the correct order
        return messages[::-1]  # root solution, reflection, child 1, ...

    # reward is now a scalar value
    def get_trajectory_reward(self):
        rewards = 0.
        node = self
        while node:
            rewards += node.reward
            node = node.parent
        return rewards

    # returns a set of stages solved along the trajectory
    def get_trajectory_stages_solved(self):
        stages_solved = []
        node = self
        while node:
            if(node.stage_solved):
                stages_solved.append(node.stage_solved)
            node = node.parent
        return stages_solved

    def get_trajectory_scratchpad(self):
        """Combine the node's scratchpad along the trajectory to root. The combined scratchpad is available to the next node"""
        node = self
        lst = []
        scratchpad = {}
        while node:
            # scratchpad.update(node.scratchpad)
            lst.append(node.scratchpad)
            node = node.parent
        rev_lst = lst[::-1]
        for d in rev_lst:
            scratchpad.update(d)
        return scratchpad

    def get_solved_nodes(self):
        solved_nodes = []
        nodes = deque()
        nodes.append(self)
        while nodes:
            node = nodes.popleft()
            if(node.is_solved):
                solved_nodes.append(node)
            for n in node.children:
                nodes.append(n)
        return solved_nodes

    def _get_all_children(self):
        all_nodes = []
        nodes = deque()
        nodes.append(self)
        while nodes:
            node = nodes.popleft()
            all_nodes.extend(node.children)
            for n in node.children:
                nodes.append(n)
        return all_nodes

    def get_best_solution(self):
        """Return the best solution from within the current sub-tree."""
        all_nodes = [self] + self._get_all_children()
        best_node = max(
            all_nodes,
            # We filter out all non-terminal, non-solution trajectories
            key=lambda node: int(node.is_terminal and node.is_solved) * node.value,
        )
        return best_node

    def _mark_tree_as_solved(self):
        parent = self.parent
        while parent:
            parent._is_solved = True
            parent = parent.parent

class TreeState_Verifiable_Rewards_Rollout(TypedDict):
    # The full tree
    messages: Annotated[Sequence[BaseMessage], add_messages]
    root: NotRequired[Node_Verifiable_Rewards_Rollout]
    input: NotRequired[str]
    scratchpad: NotRequired[Dict[str, Any]]
    # plan: NotRequired[str]
    # summary: NotRequired[RunningSummary]
    num_tool_calls: NotRequired[int]


# Provides a verifiable reward alongside a human reflection based on the reward
def verifiable_rewards(output_messages, traj_rewards, new_node_scratchpads, tool_docs, problem_type):
    penalty = 0
    reward_map = {
        'train_data_loading': 1,
        'test_data_loading': 1,
        'combine_train_test': 1,
        'data_cleaning': 2,
        'feature_engineering': 2,
        'split_train_test': 1,
        'train_data_to_features_target': 1,
        'test_data_to_features': 1,
        'modeling': 2,
        'create_submission_dataframe': 2,
        # 'create_submission_file': 1,
    }
    all_candidate_rewards = []
    all_candidate_feedback_msgs = []
    # print(f"traj_rewards: {traj_rewards}")
    # if(traj_rewards['create_submission_dataframe'] > 0):
    #     ipdb.set_trace()
    # import ipdb; ipdb.set_trace()
    for i, output_msg in enumerate(output_messages):
        curr_reward = defaultdict(float)
        if(traj_rewards['train_data_loading'] == 0):
            try:
                if(
                    output_msg[0].additional_kwargs['tool_calls'][0]['name'] == 'read_data' and \
                    'train.csv' in output_msg[0].additional_kwargs['tool_calls'][0]['args']['func_kwargs']['filepath'] and \
                    'Applied read_data ' in output_msg[1].content
                ):
                    curr_reward['train_data_loading'] = reward_map['train_data_loading']
                    feedback_msg = HumanMessage(content=f"Human feedback: Verified that the train data was loaded successfully")
                elif(
                    output_msg[0].additional_kwargs['tool_calls'][0]['name'] == 'read_data' and \
                    'test.csv' in output_msg[0].additional_kwargs['tool_calls'][0]['args']['func_kwargs']['filepath'] and \
                    'Applied read_data ' in output_msg[1].content
                ):
                    curr_reward['train_data_loading'] = penalty
                    feedback_msg = HumanMessage(content=f"Human feedback: Loaded test data before train. Penalized for this")
                else:
                    feedback_msg = HumanMessage(content=f"Human feedback: Verified that the train data was NOT loaded successfully. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
            except:
                feedback_msg = HumanMessage(content=f"Human feedback: Verified that the train data was NOT loaded successfully. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
                print(f"Unable to correctly parse if read_data was successfully called for train data")
        elif(traj_rewards['test_data_loading'] == 0):
            try:
                if(
                    output_msg[0].additional_kwargs['tool_calls'][0]['name'] == 'read_data' and \
                    'test.csv' in output_msg[0].additional_kwargs['tool_calls'][0]['args']['func_kwargs']['filepath'] and \
                    'Applied read_data ' in output_msg[1].content
                ):
                    curr_reward['test_data_loading'] = reward_map['test_data_loading']
                    feedback_msg = HumanMessage(content=f"Human feedback: Verified that the test data was loaded successfully")
                else:
                    feedback_msg = HumanMessage(content=f"Human feedback: Verified that the test data was NOT loaded successfully. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
            except:
                feedback_msg = HumanMessage(content=f"Human feedback: Verified that the test data was NOT loaded successfully. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
                print(f"Unable to correctly parse if read_data was successfully called for test data")
        elif(traj_rewards['combine_train_test'] == 0):
            try:
                if('Applied concatenate_train_test' in output_msg[1].content):
                    curr_reward['combine_train_test'] = reward_map['combine_train_test']
                    feedback_msg = HumanMessage(content=f"Human feedback: Verified that the train and test data were concatenated successfully")
                else:
                    feedback_msg = HumanMessage(content=f"Human feedback: Verified that the train and test data were NOT concatenated successfully. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
            except:
                feedback_msg = HumanMessage(content=f"Human feedback: Verified that the train and test data were NOT concatenated successfully. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
                print(f"combine_train_test not applied successfully")
        # elif(traj_rewards['data_cleaning'] == 0):
        #     try:
        #         if('Applied split_combined_into_train_test' in output_msg[1].content):
        #             curr_reward['data_cleaning'] = penalty
        #             print(f"Data cleaning reward penalized - split_combined_into_train_test was applied")
        #             feedback_msg = HumanMessage(content=f"Human Feedback: The combined train and test data was split back into its components, before the data cleaning and feature engineering stages were completed. This is bad practice.")
        #         else:
        #             temp_scratch = new_node_scratchpads[i]
        #             # We expect this to only have 1 item 
        #             assert len(temp_scratch) == 1, f"The Scratchpad in the data_cleaning phase is expected to only have 1 item"
        #             df_ = list(temp_scratch.values())[0]
        #             # check if df_ has no Nans
        #             if not df_.isnull().any().any():
        #                 curr_reward['data_cleaning'] = reward_map['data_cleaning']     
        #                 feedback_msg = HumanMessage(content=f"Human Feedback: Verified that the data cleaning was successful and no Nans were found in the data at the current step")
        #             else:
        #                 feedback_msg = HumanMessage(content=f"Human Feedback: Verified that the data cleaning is still in progress and Nans were found in the data at the current step")
        #     except:
        #         print(f"Unable to correctly parse if data_cleaning was successfully called")
        #         feedback_msg = HumanMessage(content=f"Data Cleaning is still in progress")
        # elif(traj_rewards['feature_engineering'] == 0):
        #     try:
        #         if('Applied split_combined_into_train_test' in output_msg[1].content):
        #             curr_reward['feature_engineering'] = penalty
        #             print(f"Feature engineering reward penalized - split_combined_into_train_test was applied")
        #             feedback_msg = HumanMessage(content=f"Human Feedback: The combined train and test data was split back into its components, before the data cleaning and feature engineering stages were completed. This is bad practice.")
        #         else:
        #             temp_scratch = new_node_scratchpads[i]
        #             # We expect this to only have 1 item 
        #             assert len(temp_scratch) == 1, f"The Scratchpad in the feature_engineering phase is expected to only have 1 item"
        #             df_ = list(temp_scratch.values())[0]

        #             if not df_.isnull().any().any():
        #                 curr_reward['feature_engineering'] = penalty
        #                 feedback_msg = HumanMessage(content=f"Human Feedback: Verified that the feature engineering is incomplete and Nans were found in the data at the current step. No Nan's should be existing after the previous stage i.e data cleaning")
        #                 print(f"Feature engineering reward penalized - Nan's present in data at the current step. No Nan's should be existing after the previous stage i.e data cleaning")
        #             else:
        #                 # Check feature engineering conditions
        #                 reward_granted = False
                        
        #                 # Condition 1: All columns are int or float
        #                 numeric_types = ['int64', 'int32', 'float64', 'float32', 'int', 'float']
        #                 all_numeric = all(str(df_[col].dtype) in numeric_types or 
        #                                     pd.api.types.is_numeric_dtype(df_[col]) 
        #                                     for col in df_.columns)
                        
        #                 if all_numeric:
        #                     curr_reward['feature_engineering'] = reward_map['feature_engineering']
        #                     reward_granted = True
        #                     print(f"Feature engineering reward granted - All columns are numeric (int/float)")
        #                     feedback_msg = HumanMessage(content=f"Human Feedback: Verified that the feature engineering is complete and all columns are numeric (int/float)")
        #                 else:
        #                     # Condition 2: Check categorical columns have <= 30 unique values
        #                     categorical_valid = True
        #                     for col in df_.columns:
        #                         if (pd.api.types.is_categorical_dtype(df_[col]) or 
        #                             pd.api.types.is_object_dtype(df_[col])):
        #                             unique_count = df_[col].nunique()
        #                             if unique_count > 30:
        #                                 categorical_valid = False
        #                                 print(f"Column '{col}' has {unique_count} unique values (> 30)")
        #                                 # feedback_msg = HumanMessage(content=f"Human Feedback: Verified that the feature engineering is incomplete with categorical columns with > 30 unique values still present. Either drop these columns, encode them, or transform these features so that the important information present in these columns is extracted.")
        #                                 break
        #                     if categorical_valid:
        #                         curr_reward['feature_engineering'] = reward_map['feature_engineering']
        #                         reward_granted = True
        #                         feedback_msg = HumanMessage(content=f"Human Feedback: Verified that the feature engineering is complete and all columns are either numeric (int/float) or categorical with <= 30 unique values")
        #                         print(f"Feature engineering reward granted - All categorical columns have <= 30 unique values")
                        
        #                 if not reward_granted:
        #                     feedback_msg = HumanMessage(content=f"Human Feedback: Verified that the feature engineering is incomplete due to invalid column dtypes present. Only int/float or category column data types are allowed at the end of feature engineering stage. Categorical data type columns if present should have <= 30 unique values. In case of such categorical columns either drop these columns, encode them, or transform these features so that the important information present in these columns is extracted.")
        #                     print(f"Feature engineering not complete - columns need to be numeric or categorical with <= 30 unique values")
        #     except:
        #         feedback_msg = HumanMessage(content=f"Feature Engineering is still in progress")
        #         print(f"Unable to correctly parse if feature_engineering was successfully called")
        elif(traj_rewards['data_cleaning'] == 0):
            try:
                temp_scratch = new_node_scratchpads[i]
                # We expect this to only have 1 item 
                assert len(temp_scratch) == 1, f"The Scratchpad in the data_cleaning phase is expected to only have 1 item"
                df_ = list(temp_scratch.values())[0]
                # check if df_ has no Nans except in target variable 'Transported'
                # Drop 'Transported' column for checking if only allowed nulls exist
                # if 'Transported' in df_.columns:
                #     df_check = df_.drop('Transported', axis=1)
                # else:
                #     df_check = df_
                
                df_check = df_
                if not df_check.isnull().any().any():
                    curr_reward['data_cleaning'] = reward_map['data_cleaning']     
                    feedback_msg = HumanMessage(content=f"Human Feedback: Verified that the data cleaning was successful and no Nans were found in non-target columns at the current step")
                else:
                    null_cols = df_check.columns[df_check.isnull().any()].tolist()
                    feedback_msg = HumanMessage(content=f"Human Feedback: Verified that the data cleaning is still in progress. Nulls found in columns: {null_cols}")
            except:
                print(f"Unable to correctly parse if data_cleaning was successfully called. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
                feedback_msg = HumanMessage(content=f"Data Cleaning is still in progress. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
        elif(traj_rewards['feature_engineering'] == 0):
            # Only reward if all the features are numeric
            try:
                temp_scratch = new_node_scratchpads[i]
                # We expect this to only have 1 item 
                assert len(temp_scratch) == 1, f"The Scratchpad in the feature_engineering phase is expected to only have 1 item"
                df_ = list(temp_scratch.values())[0]

                # If number of columns is > 1000, we doont reward, irrespective of the dtypes
                if len(df_.columns) > 100000:
                    feedback_msg = HumanMessage(content=f"Human Feedback: Verified that the feature engineering is incomplete and number of columns is greater than 100000. Issue is probably from encoding category columns with high cardinality. Please consider dropping these columns instead of encoding them.")
                    print(f"Feature engineering reward not granted - Number of columns in dataframe is greater than 100000")
                else:
                    # Check feature engineering conditions
                    reward_granted = False
                    
                    # Store information about problematic columns
                    non_categorical_dtypes = []
                    categorical_dtypes = []
                    
                    # Condition 1: All columns are int or float
                    numeric_types = ['int64', 'int32', 'float64', 'float32', 'int', 'float']
                    all_numeric = all(str(df_[col].dtype) in numeric_types or 
                                        pd.api.types.is_numeric_dtype(df_[col]) 
                                        for col in df_.columns)
                    
                    if all_numeric:
                        curr_reward['feature_engineering'] = reward_map['feature_engineering']
                        print(f"Feature engineering reward granted - All columns are numeric (int/float)")
                        feedback_msg = HumanMessage(content=f"Human Feedback: Verified that the feature engineering is complete and all columns are numeric (int/float)")
                    else:
                        # Condition 2: Check all columns and store issues
                        
                        for col in df_.columns:
                            col_dtype = str(df_[col].dtype)
                            unique_count = df_[col].nunique()
                            
                            if (pd.api.types.is_categorical_dtype(df_[col]) or 
                                pd.api.types.is_object_dtype(df_[col])):
                                categorical_dtypes.append({
                                    'column': col,
                                    'unique_count': unique_count,
                                    'dtype': col_dtype
                                })
                                print(f"Column '{col}' has {unique_count} unique values")
                            elif not (col_dtype in numeric_types or pd.api.types.is_numeric_dtype(df_[col]) or
                                    pd.api.types.is_categorical_dtype(df_[col]) or 
                                    pd.api.types.is_object_dtype(df_[col])):
                                # Store problematic dtypes (not numeric, not categorical, not object)
                                non_categorical_dtypes.append({
                                    'column': col,
                                    'dtype': col_dtype,
                                    'unique_count': unique_count
                                })
                                print(f"Column '{col}' has non categorical dtype: {col_dtype}")
                        
                            # Store detailed information about issues
                        issues = []
                        if non_categorical_dtypes:
                            issues.append(f"Non-categorical dtypes: {non_categorical_dtypes}")
                        if categorical_dtypes:
                            issues.append(f"Categorical dtypes: {categorical_dtypes}")
                        feedback_msg = HumanMessage(content=f"Categorical columns found: {categorical_dtypes}. Columns with dtypes that are not numeric/categorical found: {non_categorical_dtypes}. Please consider casting them to the correct dtypes or encoding them. Consider dropping categorical columns with a large number of categories, i.e exceeding 100. Only numeric columns are allowed at the end of feature engineering stage.")
                        print(f"Feature engineering not complete. Issues found")
                    
                    # if not reward_granted:
                    #     feedback_msg = HumanMessage(content=f"Human Feedback: Verified that the feature engineering is incomplete due to invalid column dtypes present. Only int/float or category column data types are allowed at the end of feature engineering stage. Categorical data type columns if present should have <= 100 unique values. In case of such categorical columns either drop these columns, encode them, or transform these features so that the important information present in these columns is extracted.")
                    #     print(f"Feature engineering not complete - columns need to be numeric or categorical with <= 30 unique values")
            except:
                feedback_msg = HumanMessage(content=f"Feature Engineering is still in progress. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
                print(f"Unable to correctly parse if feature_engineering was successfully called. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")   
        elif(traj_rewards['split_train_test'] == 0):
            # import ipdb; ipdb.set_trace()
            try:
                if('Applied split_combined_into_train_test' in output_msg[1].content):
                    curr_reward['split_train_test'] = reward_map['split_train_test']
                    feedback_msg = HumanMessage(content=f"Human Feedback: Verified that the train and test data were split successfully")
                else:
                    feedback_msg = HumanMessage(content=f"Human Feedback: Verified that the train and test data were NOT split successfully. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
            except:
                feedback_msg = HumanMessage(content=f"Human Feedback: Verified that the train and test data were NOT split successfully. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
                print(f"split_combined_into_train_test not applied successfully")
        elif(traj_rewards['train_data_to_features_target'] == 0):
            try:
                temp_scratch = new_node_scratchpads[i]
                # Check if tool was applied and scratchpad has X_train and y_train
                if 'Applied convert_dataframe_to_features_target' in output_msg[1].content:
                    # Check if scratchpad has exactly X_train and y_train
                    if 'X_train' in temp_scratch and 'Y_train' in temp_scratch and len(temp_scratch) == 2:
                        curr_reward['train_data_to_features_target'] = reward_map['train_data_to_features_target']
                        feedback_msg = HumanMessage(content=f"Human Feedback: Verified that the train data was successfully converted to features (X_train) and target (Y_train)")
                        print(f"Train data to features/target reward granted - Found X_train and Y_train in scratchpad")
                    else:
                        feedback_msg = HumanMessage(content=f"Human Feedback: Tool applied but scratchpad missing X_train and/or Y_train. Found keys: {list(temp_scratch.keys())}")
                        print(f"Train data conversion incomplete - scratchpad keys: {list(temp_scratch.keys())}")
                else:
                    feedback_msg = HumanMessage(content=f"Human Feedback: convert_dataframe_to_features_target tool not successfully applied. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
                    print(f"convert_dataframe_to_features_target not successfully applied")
            except Exception as e:
                feedback_msg = HumanMessage(content=f"Human Feedback: Error checking train data to features/target conversion: {str(e)}. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
                print(f"Error in train_data_to_features_target verification: {e}")
        elif(traj_rewards['test_data_to_features'] == 0):
            try:
                temp_scratch = new_node_scratchpads[i]
                # Check if tool was applied and scratchpad has X_test
                if 'Applied convert_dataframe_to_features_target' in output_msg[1].content:
                    # Check if scratchpad has X_test
                    if 'X_test' in temp_scratch and len(temp_scratch) == 1:
                        curr_reward['test_data_to_features'] = reward_map['test_data_to_features']
                        feedback_msg = HumanMessage(content=f"Human Feedback: Verified that the test data was successfully converted to features (X_test)")
                        print(f"Test data to features reward granted - Found X_test in scratchpad")
                    else:
                        feedback_msg = HumanMessage(content=f"Human Feedback: Tool applied but scratchpad missing X_test or has extra items. Found keys: {list(temp_scratch.keys())}")
                        print(f"Test data conversion incomplete - scratchpad keys: {list(temp_scratch.keys())}")
                else:
                    feedback_msg = HumanMessage(content=f"Human Feedback: convert_dataframe_to_features_target tool not successfully applied. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
                    print(f"convert_dataframe_to_features_target not successfully applied")
            except Exception as e:
                feedback_msg = HumanMessage(content=f"Human Feedback: Error checking test data to features conversion: {str(e)}. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
                print(f"Error in test_data_to_features verification: {e}")
        elif(traj_rewards['modeling'] == 0):
            # import ipdb; ipdb.set_trace()
            feedback_msg = HumanMessage(content=f"Modeling is still in progress")
            try:
                # if('tune' not in output_msg[0].additional_kwargs['tool_calls'][0]['name']):
                #     curr_reward['modeling'] = penalty
                tool_message_content = output_msg[1].content
                # Check for tune_{model}_{type} pattern
                models = ['catboost', 'lightgbm', 'logistic', 'xgboost', 'random_forest']
                types = ['regressor', 'classifier']
                tuning_success = False
                for model in models:
                    for model_type in types:
                        if f'Applied tune_{model}_{model_type}' in tool_message_content:
                            tuning_success = True
                            print(f"Modeling reward granted - Found successful tuning: tune_{model}_{model_type}")
                            break
                    if tuning_success:
                        break
                if tuning_success:
                    # import ipdb; ipdb.set_trace()
                    temp_scratch = new_node_scratchpads[i]
                    scale_factor = 2.5
                    base_reward = reward_map['modeling']
                    cv_bonus = 0

                    if 'best_cv_score' in temp_scratch:
                        cv_score = temp_scratch['best_cv_score'].item()
                        if pd.isna(cv_score):
                            cv_bonus = penalty  # Penalize for NaN CV score
                        else:
                            cv_bonus = cv_score * scale_factor  # Bonus for valid CV score

                    curr_reward['modeling'] = base_reward + cv_bonus
                    # curr_reward['modeling'] = reward_map['modeling'] + (temp_scratch['best_cv_score'].item()*scale_factor if 'best_cv_score' in temp_scratch else 0)
                    feedback_msg = HumanMessage(content=f"Human Feedback: Verified that the modeling was successful")
            except:
                feedback_msg = HumanMessage(content=f"Modeling is still in progress. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
                print(f"Unable to parse ToolMessage for modeling tuning")
        elif(traj_rewards['create_submission_dataframe'] == 0):
            feedback_msg = HumanMessage(content=f"Human Feedback: submission DataFrame was NOT created successfully. Please check the signature of the wrapped function if any, and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
            try:
                temp_scratch = new_node_scratchpads[i]
                if temp_scratch:
                    for key, value in temp_scratch.items():
                        # Check if value is a DataFrame
                        if hasattr(value, 'columns') and hasattr(value, 'shape'):
                            # Check if DataFrame has no NaNs
                            if not value.isnull().any().any():
                                num_cols = len(value.columns)
                                
                                def is_valid_prediction_column(col_data, expected_type=None):
                                    """Check if column contains valid prediction values based on problem type"""
                                    unique_vals = col_data.unique()
                                    
                                    if expected_type == 'binary-acc':
                                        # Binary classification: True/False or 0/1
                                        if all(val in [True, False] for val in unique_vals):
                                            return True, "boolean"
                                        if all(val in [0, 1] for val in unique_vals):
                                            return True, "binary"
                                        return False, "invalid"
                                    
                                    elif expected_type == 'binary-probs':
                                        # Probability values between 0 and 1
                                        if all(isinstance(val, (int, float)) and 0 <= val <= 1 for val in unique_vals):
                                            return True, "probability"
                                        return False, "invalid"
                                    
                                    elif expected_type == 'regression':
                                        # Regression: numeric values (floats/ints)
                                        try:
                                            import numpy as np
                                            if all(isinstance(val, (int, float, np.number)) for val in unique_vals):
                                                return True, "regression"
                                        except ImportError:
                                            if all(isinstance(val, (int, float)) for val in unique_vals):
                                                return True, "regression"
                                        return False, "invalid"
                                    
                                    elif expected_type == 'multiclass':
                                        # Multi-class classification: set of integers
                                        try:
                                            import numpy as np
                                            if all(isinstance(val, (int, np.integer)) for val in unique_vals):
                                                if len(unique_vals) <= 50:  # Reasonable number of classes
                                                    return True, "multiclass"
                                        except ImportError:
                                            if all(isinstance(val, int) for val in unique_vals):
                                                if len(unique_vals) <= 50:  # Reasonable number of classes
                                                    return True, "multiclass"
                                        return False, "invalid"
                                    
                                    else:
                                        # General validation (backward compatibility)
                                        if all(val in [True, False] for val in unique_vals):
                                            return True, "boolean"
                                        if all(val in [0, 1] for val in unique_vals):
                                            return True, "binary"
                                        if all(isinstance(val, (int, float)) and 0 <= val <= 1 for val in unique_vals):
                                            return True, "probability"
                                        try:
                                            import numpy as np
                                            if all(isinstance(val, (int, np.integer)) for val in unique_vals):
                                                if len(unique_vals) <= 20:
                                                    return True, "multiclass"
                                            if all(isinstance(val, (int, float, np.number)) for val in unique_vals):
                                                return True, "regression"
                                        except ImportError:
                                            if all(isinstance(val, int) for val in unique_vals):
                                                if len(unique_vals) <= 20:
                                                    return True, "multiclass"
                                            if all(isinstance(val, (int, float)) for val in unique_vals):
                                                return True, "regression"
                                        
                                    return False, "invalid"
                                
                                # Validate based on problem type and number of columns
                                cols = list(value.columns)
                                prediction_valid = False
                                pred_type = "unknown"
                                
                                if problem_type == 'binary-probs':
                                    if num_cols == 3:
                                        # 3 columns: ID + prob_0 + prob_1
                                        prob_cols = 0
                                        for col in cols:
                                            is_valid, p_type = is_valid_prediction_column(value[col], 'binary-probs')
                                            if is_valid:
                                                prob_cols += 1
                                        if prob_cols >= 2:  # At least 2 probability columns
                                            prediction_valid = True
                                            pred_type = "binary-probs-3col"
                                    elif num_cols == 2:
                                        # 2 columns: prob_0 + prob_1
                                        prob_cols = 0
                                        for col in cols:
                                            is_valid, p_type = is_valid_prediction_column(value[col], 'binary-probs')
                                            if is_valid:
                                                prob_cols += 1
                                        if prob_cols == 2:  # Both are probability columns
                                            prediction_valid = True
                                            pred_type = "binary-probs-2col"
                                    elif num_cols == 1:
                                        # 1 column: single probability
                                        is_valid, pred_type = is_valid_prediction_column(value[cols[0]], 'binary-probs')
                                        if is_valid:
                                            prediction_valid = True
                                            pred_type = "binary-probs-1col"
                                else:
                                    # For binary-acc, regression, multiclass
                                    if num_cols == 2:
                                        # Two columns: check if one looks like predictions
                                        for col in cols:
                                            is_valid, pred_type = is_valid_prediction_column(value[col], problem_type)
                                            if is_valid:
                                                prediction_valid = True
                                                break
                                    elif num_cols == 1:
                                        # Single column: check if it contains valid predictions
                                        is_valid, pred_type = is_valid_prediction_column(value[cols[0]], problem_type)
                                        if is_valid:
                                            prediction_valid = True
                                
                                # Check if validation passed
                                if prediction_valid:
                                    curr_reward['create_submission_dataframe'] = reward_map['create_submission_dataframe']
                                    feedback_msg = HumanMessage(content=f"Human Feedback: Verified that the submission DataFrame was created successfully with {num_cols} columns ({pred_type}) and no missing values.")
                                    print(f"Create submission reward granted - Found submission DataFrame with {num_cols} columns: {cols}, prediction type: {pred_type}")
                                    break
                                else:
                                    if problem_type == 'binary-probs':
                                        expected_cols = "1 (single prob), 2 (prob_0 + prob_1), or 3 (ID + prob_0 + prob_1)"
                                    else:
                                        expected_cols = "1 (prediction) or 2 (ID + prediction)"
                                    feedback_msg = HumanMessage(content=f"Human Feedback: Submission DataFrame validation failed for {problem_type}. Has {num_cols} columns, expected {expected_cols}.")
                                    print(f"DataFrame validation failed for {problem_type}: {num_cols} columns, expected {expected_cols}")
                                    break
                            else:
                                feedback_msg = HumanMessage(content=f"Human Feedback: Submission DataFrame creation failed. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
                                print(f"DataFrame contains missing values")
                                break
            except Exception as e:
                feedback_msg = HumanMessage(content=f"Human Feedback: Unable to verify submission DataFrame creation: {e}. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
                print(f"Unable to parse scratchpad for submission DataFrame: {e}")
        else:
            feedback_msg = HumanMessage(content=f"The trajectory has already been solved. No further rewards will be granted.")
            curr_reward = 0
        # elif(traj_rewards['create_submission_file'] == 0):
        #     try:
        #         if('Applied save_dataframe_to_csv' in output_msg[1].content):
        #             curr_reward['create_submission_file'] = reward_map['create_submission_file']
        #             feedback_msg = HumanMessage(content=f"Human feedback: Verified that the submission csv was written to disk successfully")
        #         else:
        #             feedback_msg = HumanMessage(content=f"Human feedback: Verified that the submission csv was NOT written to disk successfully")
        #     except:
        #         feedback_msg = HumanMessage(content=f"Human feedback: Verified that the submission csv was NOT written to disk successfully")
        #         print(f"combine_train_test not applied successfully")
        all_candidate_rewards.append(curr_reward)
        all_candidate_feedback_msgs.append(feedback_msg)
    return all_candidate_rewards, all_candidate_feedback_msgs


# Provides a verifiable reward alongside a human reflection based on the reward
def verifiable_rewards_outcome_based(output_messages, traj_rewards, new_node_scratchpads, tool_docs, problem_type):
    penalty = 0
    reward_map = {
        'modeling': 0,
        'create_submission_dataframe': 2,
        # 'create_submission_file': 1,
    }
    all_candidate_rewards = []
    all_candidate_feedback_msgs = []
    # import ipdb; ipdb.set_trace()
    for i, output_msg in enumerate(output_messages):
        curr_reward = defaultdict(float)
        
        if(traj_rewards['modeling'] == 0):
            # import ipdb; ipdb.set_trace()
            feedback_msg = HumanMessage(content=f"Modeling is still in progress")
            try:
                # if('tune' not in output_msg[0].additional_kwargs['tool_calls'][0]['name']):
                #     curr_reward['modeling'] = penalty
                tool_message_content = output_msg[1].content
                # Check for tune_{model}_{type} pattern
                models = ['catboost', 'lightgbm', 'logistic', 'xgboost', 'random_forest']
                types = ['regressor', 'classifier']
                tuning_success = False
                for model in models:
                    for model_type in types:
                        if f'Applied tune_{model}_{model_type}' in tool_message_content:
                            tuning_success = True
                            print(f"Modeling reward granted - Found successful tuning: tune_{model}_{model_type}")
                            break
                    if tuning_success:
                        break
                if tuning_success:
                    # import ipdb; ipdb.set_trace()
                    temp_scratch = new_node_scratchpads[i]
                    scale_factor = 2.5
                    base_reward = reward_map['modeling']
                    cv_bonus = 0

                    if 'best_cv_score' in temp_scratch:
                        cv_score = temp_scratch['best_cv_score'].item()
                        if pd.isna(cv_score):
                            pass
                        else:
                            cv_bonus = cv_score * scale_factor  # Bonus for valid CV score

                    curr_reward['modeling'] = base_reward + cv_bonus
                    # curr_reward['modeling'] = reward_map['modeling'] + (temp_scratch['best_cv_score'].item()*scale_factor if 'best_cv_score' in temp_scratch else 0)
                    feedback_msg = HumanMessage(content=f"Human Feedback: Verified that the modeling was successful")
            except:
                feedback_msg = HumanMessage(content=f"Modeling is still in progress. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
                print(f"Unable to parse ToolMessage for modeling tuning")
        elif(traj_rewards['create_submission_dataframe'] == 0):
            feedback_msg = HumanMessage(content=f"Human Feedback: submission DataFrame was NOT created successfully. Please check the signature of the wrapped function if any, and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
            try:
                temp_scratch = new_node_scratchpads[i]
                if temp_scratch:
                    for key, value in temp_scratch.items():
                        # Check if value is a DataFrame
                        if hasattr(value, 'columns') and hasattr(value, 'shape'):
                            # Check if DataFrame has no NaNs
                            if not value.isnull().any().any():
                                num_cols = len(value.columns)
                                
                                def is_valid_prediction_column(col_data, expected_type=None):
                                    """Check if column contains valid prediction values based on problem type"""
                                    unique_vals = col_data.unique()
                                    
                                    if expected_type == 'binary-acc':
                                        # Binary classification: True/False or 0/1
                                        if all(val in [True, False] for val in unique_vals):
                                            return True, "boolean"
                                        if all(val in [0, 1] for val in unique_vals):
                                            return True, "binary"
                                        return False, "invalid"
                                    
                                    elif expected_type == 'binary-probs':
                                        # Probability values between 0 and 1
                                        if all(isinstance(val, (int, float)) and 0 <= val <= 1 for val in unique_vals):
                                            return True, "probability"
                                        return False, "invalid"
                                    
                                    elif expected_type == 'regression':
                                        # Regression: numeric values (floats/ints)
                                        try:
                                            import numpy as np
                                            if all(isinstance(val, (int, float, np.number)) for val in unique_vals):
                                                return True, "regression"
                                        except ImportError:
                                            if all(isinstance(val, (int, float)) for val in unique_vals):
                                                return True, "regression"
                                        return False, "invalid"
                                    
                                    elif expected_type == 'multiclass':
                                        # Multi-class classification: set of integers
                                        try:
                                            import numpy as np
                                            if all(isinstance(val, (int, np.integer)) for val in unique_vals):
                                                if len(unique_vals) <= 50:  # Reasonable number of classes
                                                    return True, "multiclass"
                                        except ImportError:
                                            if all(isinstance(val, int) for val in unique_vals):
                                                if len(unique_vals) <= 50:  # Reasonable number of classes
                                                    return True, "multiclass"
                                        return False, "invalid"
                                    
                                    else:
                                        # General validation (backward compatibility)
                                        if all(val in [True, False] for val in unique_vals):
                                            return True, "boolean"
                                        if all(val in [0, 1] for val in unique_vals):
                                            return True, "binary"
                                        if all(isinstance(val, (int, float)) and 0 <= val <= 1 for val in unique_vals):
                                            return True, "probability"
                                        try:
                                            import numpy as np
                                            if all(isinstance(val, (int, np.integer)) for val in unique_vals):
                                                if len(unique_vals) <= 20:
                                                    return True, "multiclass"
                                            if all(isinstance(val, (int, float, np.number)) for val in unique_vals):
                                                return True, "regression"
                                        except ImportError:
                                            if all(isinstance(val, int) for val in unique_vals):
                                                if len(unique_vals) <= 20:
                                                    return True, "multiclass"
                                            if all(isinstance(val, (int, float)) for val in unique_vals):
                                                return True, "regression"
                                        
                                    return False, "invalid"
                                
                                # Validate based on problem type and number of columns
                                cols = list(value.columns)
                                prediction_valid = False
                                pred_type = "unknown"
                                
                                if problem_type == 'binary-probs':
                                    if num_cols == 3:
                                        # 3 columns: ID + prob_0 + prob_1
                                        prob_cols = 0
                                        for col in cols:
                                            is_valid, p_type = is_valid_prediction_column(value[col], 'binary-probs')
                                            if is_valid:
                                                prob_cols += 1
                                        if prob_cols >= 2:  # At least 2 probability columns
                                            prediction_valid = True
                                            pred_type = "binary-probs-3col"
                                    elif num_cols == 2:
                                        # 2 columns: prob_0 + prob_1
                                        prob_cols = 0
                                        for col in cols:
                                            is_valid, p_type = is_valid_prediction_column(value[col], 'binary-probs')
                                            if is_valid:
                                                prob_cols += 1
                                        if prob_cols == 2:  # Both are probability columns
                                            prediction_valid = True
                                            pred_type = "binary-probs-2col"
                                    elif num_cols == 1:
                                        # 1 column: single probability
                                        is_valid, pred_type = is_valid_prediction_column(value[cols[0]], 'binary-probs')
                                        if is_valid:
                                            prediction_valid = True
                                            pred_type = "binary-probs-1col"
                                else:
                                    # For binary-acc, regression, multiclass
                                    if num_cols == 2:
                                        # Two columns: check if one looks like predictions
                                        for col in cols:
                                            is_valid, pred_type = is_valid_prediction_column(value[col], problem_type)
                                            if is_valid:
                                                prediction_valid = True
                                                break
                                    elif num_cols == 1:
                                        # Single column: check if it contains valid predictions
                                        is_valid, pred_type = is_valid_prediction_column(value[cols[0]], problem_type)
                                        if is_valid:
                                            prediction_valid = True
                                
                                # Check if validation passed
                                if prediction_valid:
                                    curr_reward['create_submission_dataframe'] = reward_map['create_submission_dataframe']
                                    feedback_msg = HumanMessage(content=f"Human Feedback: Verified that the submission DataFrame was created successfully with {num_cols} columns ({pred_type}) and no missing values.")
                                    print(f"Create submission reward granted - Found submission DataFrame with {num_cols} columns: {cols}, prediction type: {pred_type}")
                                    break
                                else:
                                    if problem_type == 'binary-probs':
                                        expected_cols = "1 (single prob), 2 (prob_0 + prob_1), or 3 (ID + prob_0 + prob_1)"
                                    else:
                                        expected_cols = "1 (prediction) or 2 (ID + prediction)"
                                    feedback_msg = HumanMessage(content=f"Human Feedback: Submission DataFrame validation failed for {problem_type}. Has {num_cols} columns, expected {expected_cols}.")
                                    print(f"DataFrame validation failed for {problem_type}: {num_cols} columns, expected {expected_cols}")
                                    break
                            else:
                                feedback_msg = HumanMessage(content=f"Human Feedback: Submission DataFrame creation failed. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
                                print(f"DataFrame contains missing values")
                                break
            except Exception as e:
                feedback_msg = HumanMessage(content=f"Human Feedback: Unable to verify submission DataFrame creation: {e}. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
                print(f"Unable to parse scratchpad for submission DataFrame: {e}")
        else:
            feedback_msg = HumanMessage(content=f"The trajectory has already been solved. No further rewards will be granted.")
            curr_reward = 0
        all_candidate_rewards.append(curr_reward)
        all_candidate_feedback_msgs.append(feedback_msg)
    return all_candidate_rewards, all_candidate_feedback_msgs


###########################################################
###########################################################
###########################################################
###########################################################     
###########################################################
###########################################################
###########################################################
###########################################################


def verifiable_rewards_rollout(output_messages, traj_rewards, new_node_scratchpads, traj_stages_solved, tool_docs, problem_type):
    penalty = 0
    reward_map = {
        'train_data_loading': 1,
        'test_data_loading': 1,
        'combine_train_test': 1,
        'data_cleaning': 2,
        'feature_engineering': 2,
        'split_train_test': 1,
        'train_data_to_features_target': 1,
        'test_data_to_features': 1,
        'modeling': 2,
        'create_submission_dataframe': 2,
        # 'create_submission_file': 1,
    }
    STAGES = ['init', 'train_data_loading', 'test_data_loading', 'combine_train_test', 'data_cleaning', 'feature_engineering', 'split_train_test', 'train_data_to_features_target', 'test_data_to_features', 'modeling', 'create_submission_dataframe', 'end_stage']
    max_stage_index = max([STAGES.index(stage) for stage in traj_stages_solved])
    next_stage = STAGES[max_stage_index + 1]
    # We only check for next stage to be solved
    all_candidate_rewards = []
    all_candidate_feedback_msgs = []
    # import ipdb; ipdb.set_trace()
    for i, output_msg in enumerate(output_messages):
        curr_reward = 0
        if(next_stage == 'train_data_loading'):
            try:
                if(
                    output_msg[0].additional_kwargs['tool_calls'][0]['name'] == 'read_data' and \
                    'train.csv' in output_msg[0].additional_kwargs['tool_calls'][0]['args']['func_kwargs']['filepath'] and \
                    'Applied read_data ' in output_msg[1].content
                ):
                    curr_reward = reward_map['train_data_loading']    
                    feedback_msg = HumanMessage(content=f"Human feedback: Verified that the train data was loaded successfully")
                elif(
                    output_msg[0].additional_kwargs['tool_calls'][0]['name'] == 'read_data' and \
                    'test.csv' in output_msg[0].additional_kwargs['tool_calls'][0]['args']['func_kwargs']['filepath'] and \
                    'Applied read_data ' in output_msg[1].content
                ):
                    curr_reward = penalty
                    feedback_msg = HumanMessage(content=f"Human feedback: Loaded test data before train. Penalized for this")
                else:
                    feedback_msg = HumanMessage(content=f"Human feedback: Verified that the train data was NOT loaded successfully. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
            except:
                feedback_msg = HumanMessage(content=f"Human feedback: Verified that the train data was NOT loaded successfully. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
                print(f"Unable to correctly parse if read_data was successfully called for train data")
        elif(next_stage == 'test_data_loading'):
            try:
                if(
                    output_msg[0].additional_kwargs['tool_calls'][0]['name'] == 'read_data' and \
                    'test.csv' in output_msg[0].additional_kwargs['tool_calls'][0]['args']['func_kwargs']['filepath'] and \
                    'Applied read_data ' in output_msg[1].content
                ):
                    curr_reward = reward_map['test_data_loading']
                    feedback_msg = HumanMessage(content=f"Human feedback: Verified that the test data was loaded successfully")
                else:
                    feedback_msg = HumanMessage(content=f"Human feedback: Verified that the test data was NOT loaded successfully. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
            except:
                feedback_msg = HumanMessage(content=f"Human feedback: Verified that the test data was NOT loaded successfully. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
                print(f"Unable to correctly parse if read_data was successfully called for test data")
        elif(next_stage == 'combine_train_test'):
            try:
                if('Applied concatenate_train_test' in output_msg[1].content):
                    curr_reward = reward_map['combine_train_test']
                    feedback_msg = HumanMessage(content=f"Human feedback: Verified that the train and test data were concatenated successfully")
                else:
                    feedback_msg = HumanMessage(content=f"Human feedback: Verified that the train and test data were NOT concatenated successfully. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
            except:
                feedback_msg = HumanMessage(content=f"Human feedback: Verified that the train and test data were NOT concatenated successfully. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
                print(f"combine_train_test not applied successfully")
        elif(next_stage == 'data_cleaning'):
            try:
                temp_scratch = new_node_scratchpads[i]
                # We expect this to only have 1 item 
                assert len(temp_scratch) == 1, f"The Scratchpad in the data_cleaning phase is expected to only have 1 item"
                df_ = list(temp_scratch.values())[0]
                # check if df_ has no Nans except in target variable 'Transported'
                # Drop 'Transported' column for checking if only allowed nulls exist
                # if 'Transported' in df_.columns:
                #     df_check = df_.drop('Transported', axis=1)
                # else:
                #     df_check = df_
                
                df_check = df_
                if not df_check.isnull().any().any():
                    curr_reward = reward_map['data_cleaning']     
                    feedback_msg = HumanMessage(content=f"Human Feedback: Verified that the data cleaning was successful and no Nans were found in non-target columns at the current step")
                else:
                    null_cols = df_check.columns[df_check.isnull().any()].tolist()
                    feedback_msg = HumanMessage(content=f"Human Feedback: Verified that the data cleaning is still in progress. Nulls found in columns: {null_cols}")
            except:
                print(f"Unable to correctly parse if data_cleaning was successfully called. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
                feedback_msg = HumanMessage(content=f"Data Cleaning is still in progress. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
        elif(next_stage == 'feature_engineering'):
            # Only reward if all the features are numeric
            try:
                temp_scratch = new_node_scratchpads[i]
                # We expect this to only have 1 item 
                assert len(temp_scratch) == 1, f"The Scratchpad in the feature_engineering phase is expected to only have 1 item"
                df_ = list(temp_scratch.values())[0]

                # If number of columns is > 1000, we doont reward, irrespective of the dtypes
                if len(df_.columns) > 100000:
                    feedback_msg = HumanMessage(content=f"Human Feedback: Verified that the feature engineering is incomplete and number of columns is greater than 1000. Issue is probably from encoding category columns with high cardinality. Please consider dropping these columns instead of encoding them.")
                    print(f"Feature engineering reward not granted - Number of columns in dataframe is greater than 1000")
                else:
                    # Check feature engineering conditions
                    reward_granted = False
                    
                    # Store information about problematic columns
                    non_categorical_dtypes = []
                    categorical_dtypes = []
                    
                    # Condition 1: All columns are int or float
                    numeric_types = ['int64', 'int32', 'float64', 'float32', 'int', 'float']
                    all_numeric = all(str(df_[col].dtype) in numeric_types or 
                                        pd.api.types.is_numeric_dtype(df_[col]) 
                                        for col in df_.columns)
                    
                    if all_numeric:
                        curr_reward = reward_map['feature_engineering']
                        print(f"Feature engineering reward granted - All columns are numeric (int/float)")
                        feedback_msg = HumanMessage(content=f"Human Feedback: Verified that the feature engineering is complete and all columns are numeric (int/float)")
                    else:
                        # Condition 2: Check all columns and store issues
                        
                        for col in df_.columns:
                            col_dtype = str(df_[col].dtype)
                            unique_count = df_[col].nunique()
                            
                            if (pd.api.types.is_categorical_dtype(df_[col]) or 
                                pd.api.types.is_object_dtype(df_[col])):
                                categorical_dtypes.append({
                                    'column': col,
                                    'unique_count': unique_count,
                                    'dtype': col_dtype
                                })
                                print(f"Column '{col}' has {unique_count} unique values")
                            elif not (col_dtype in numeric_types or pd.api.types.is_numeric_dtype(df_[col]) or
                                    pd.api.types.is_categorical_dtype(df_[col]) or 
                                    pd.api.types.is_object_dtype(df_[col])):
                                # Store problematic dtypes (not numeric, not categorical, not object)
                                non_categorical_dtypes.append({
                                    'column': col,
                                    'dtype': col_dtype,
                                    'unique_count': unique_count
                                })
                                print(f"Column '{col}' has non categorical dtype: {col_dtype}")
                        
                            # Store detailed information about issues
                        issues = []
                        if non_categorical_dtypes:
                            issues.append(f"Non-categorical dtypes: {non_categorical_dtypes}")
                        if categorical_dtypes:
                            issues.append(f"Categorical dtypes: {categorical_dtypes}")
                        feedback_msg = HumanMessage(content=f"Categorical columns found: {categorical_dtypes}. Columns with dtypes that are not numeric/categorical found: {non_categorical_dtypes}. Please consider casting them to the correct dtypes or encoding them. Consider dropping categorical columns with a large number of categories, i.e exceeding 100. Only numeric columns are allowed at the end of feature engineering stage.")
                        print(f"Feature engineering not complete. Issues found")
                    
                    # if not reward_granted:
                    #     feedback_msg = HumanMessage(content=f"Human Feedback: Verified that the feature engineering is incomplete due to invalid column dtypes present. Only int/float or category column data types are allowed at the end of feature engineering stage. Categorical data type columns if present should have <= 100 unique values. In case of such categorical columns either drop these columns, encode them, or transform these features so that the important information present in these columns is extracted.")
                    #     print(f"Feature engineering not complete - columns need to be numeric or categorical with <= 30 unique values")
            except:
                feedback_msg = HumanMessage(content=f"Feature Engineering is still in progress. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
                print(f"Unable to correctly parse if feature_engineering was successfully called. Please check the signature of the wrapped function and call it with the correct arguments.")   
        elif(next_stage == 'split_train_test'):
            # import ipdb; ipdb.set_trace()
            try:
                if('Applied split_combined_into_train_test' in output_msg[1].content):
                    curr_reward = reward_map['split_train_test']
                    feedback_msg = HumanMessage(content=f"Human Feedback: Verified that the train and test data were split successfully")
                else:
                    feedback_msg = HumanMessage(content=f"Human Feedback: Verified that the train and test data were NOT split successfully. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
            except:
                feedback_msg = HumanMessage(content=f"Human Feedback: Verified that the train and test data were NOT split successfully. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
                print(f"split_combined_into_train_test not applied successfully")
        elif(next_stage == 'train_data_to_features_target'):
            try:
                temp_scratch = new_node_scratchpads[i]
                # Check if tool was applied and scratchpad has X_train and y_train
                if 'Applied convert_dataframe_to_features_target' in output_msg[1].content:
                    # Check if scratchpad has exactly X_train and y_train
                    if 'X_train' in temp_scratch and 'Y_train' in temp_scratch and len(temp_scratch) == 2:
                        curr_reward = reward_map['train_data_to_features_target']
                        feedback_msg = HumanMessage(content=f"Human Feedback: Verified that the train data was successfully converted to features (X_train) and target (Y_train)")
                        print(f"Train data to features/target reward granted - Found X_train and Y_train in scratchpad")
                    else:
                        feedback_msg = HumanMessage(content=f"Human Feedback: Tool applied but scratchpad missing X_train and/or Y_train. Found keys: {list(temp_scratch.keys())}")
                        print(f"Train data conversion incomplete - scratchpad keys: {list(temp_scratch.keys())}")
                else:
                    feedback_msg = HumanMessage(content=f"Human Feedback: convert_dataframe_to_features_target tool not successfully applied. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
                    print(f"convert_dataframe_to_features_target not successfully applied")
            except Exception as e:
                feedback_msg = HumanMessage(content=f"Human Feedback: Error checking train data to features/target conversion: {str(e)}. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
                print(f"Error in train_data_to_features_target verification: {e}")
        elif(next_stage == 'test_data_to_features'):
            try:
                temp_scratch = new_node_scratchpads[i]
                # Check if tool was applied and scratchpad has X_test
                if 'Applied convert_dataframe_to_features_target' in output_msg[1].content:
                    # Check if scratchpad has X_test
                    if 'X_test' in temp_scratch and len(temp_scratch) == 1:
                        curr_reward = reward_map['test_data_to_features']
                        feedback_msg = HumanMessage(content=f"Human Feedback: Verified that the test data was successfully converted to features (X_test)")
                        print(f"Test data to features reward granted - Found X_test in scratchpad")
                    else:
                        feedback_msg = HumanMessage(content=f"Human Feedback: Tool applied but scratchpad missing X_test or has extra items. Found keys: {list(temp_scratch.keys())}")
                        print(f"Test data conversion incomplete - scratchpad keys: {list(temp_scratch.keys())}")
                else:
                    feedback_msg = HumanMessage(content=f"Human Feedback: convert_dataframe_to_features_target tool not successfully applied. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
                    print(f"convert_dataframe_to_features_target not successfully applied")
            except Exception as e:
                feedback_msg = HumanMessage(content=f"Human Feedback: Error checking test data to features conversion: {str(e)}. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
                print(f"Error in test_data_to_features verification: {e}")
        elif(next_stage == 'modeling'):
            # import ipdb; ipdb.set_trace()
            feedback_msg = HumanMessage(content=f"Modeling is still in progress")
            try:
                # if('tune' not in output_msg[0].additional_kwargs['tool_calls'][0]['name']):
                #     curr_reward['modeling'] = penalty
                tool_message_content = output_msg[1].content
                # Check for tune_{model}_{type} pattern
                models = ['catboost', 'lightgbm', 'logistic', 'xgboost', 'random_forest']
                types = ['regressor', 'classifier']
                tuning_success = False
                for model in models:
                    for model_type in types:
                        if f'Applied tune_{model}_{model_type}' in tool_message_content:
                            tuning_success = True
                            print(f"Modeling reward granted - Found successful tuning: tune_{model}_{model_type}")
                            break
                    if tuning_success:
                        break
                if tuning_success:
                    # import ipdb; ipdb.set_trace()
                    temp_scratch = new_node_scratchpads[i]
                    scale_factor = 2.5
                    base_reward = reward_map['modeling']
                    cv_bonus = 0

                    if 'best_cv_score' in temp_scratch:
                        cv_score = temp_scratch['best_cv_score'].item()
                        if pd.isna(cv_score):
                            cv_bonus = penalty  # Penalize for NaN CV score
                        else:
                            cv_bonus = cv_score * scale_factor  # Bonus for valid CV score

                    curr_reward = base_reward + cv_bonus
                    # curr_reward['modeling'] = reward_map['modeling'] + (temp_scratch['best_cv_score'].item()*scale_factor if 'best_cv_score' in temp_scratch else 0)
                    feedback_msg = HumanMessage(content=f"Human Feedback: Verified that the modeling was successful")
            except:
                feedback_msg = HumanMessage(content=f"Modeling is still in progress. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
                print(f"Unable to parse ToolMessage for modeling tuning")
        elif(next_stage == 'create_submission_dataframe'):
            feedback_msg = HumanMessage(content=f"Human Feedback: submission DataFrame was NOT created successfully. Please check the signature of the wrapped function if any, and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
            try:
                temp_scratch = new_node_scratchpads[i]
                if temp_scratch:
                    for key, value in temp_scratch.items():
                        # Check if value is a DataFrame
                        if hasattr(value, 'columns') and hasattr(value, 'shape'):
                            # Check if DataFrame has no NaNs
                            if not value.isnull().any().any():
                                num_cols = len(value.columns)
                                
                                def is_valid_prediction_column(col_data, expected_type=None):
                                    """Check if column contains valid prediction values based on problem type"""
                                    unique_vals = col_data.unique()
                                    
                                    if expected_type == 'binary-acc':
                                        # Binary classification: True/False or 0/1
                                        if all(val in [True, False] for val in unique_vals):
                                            return True, "boolean"
                                        if all(val in [0, 1] for val in unique_vals):
                                            return True, "binary"
                                        return False, "invalid"
                                    
                                    elif expected_type == 'binary-probs':
                                        # Probability values between 0 and 1
                                        if all(isinstance(val, (int, float)) and 0 <= val <= 1 for val in unique_vals):
                                            return True, "probability"
                                        return False, "invalid"
                                    
                                    elif expected_type == 'regression':
                                        # Regression: numeric values (floats/ints)
                                        try:
                                            import numpy as np
                                            if all(isinstance(val, (int, float, np.number)) for val in unique_vals):
                                                return True, "regression"
                                        except ImportError:
                                            if all(isinstance(val, (int, float)) for val in unique_vals):
                                                return True, "regression"
                                        return False, "invalid"
                                    
                                    elif expected_type == 'multiclass':
                                        # Multi-class classification: set of integers
                                        try:
                                            import numpy as np
                                            if all(isinstance(val, (int, np.integer)) for val in unique_vals):
                                                if len(unique_vals) <= 50:  # Reasonable number of classes
                                                    return True, "multiclass"
                                        except ImportError:
                                            if all(isinstance(val, int) for val in unique_vals):
                                                if len(unique_vals) <= 50:  # Reasonable number of classes
                                                    return True, "multiclass"
                                        return False, "invalid"
                                    
                                    else:
                                        # General validation (backward compatibility)
                                        if all(val in [True, False] for val in unique_vals):
                                            return True, "boolean"
                                        if all(val in [0, 1] for val in unique_vals):
                                            return True, "binary"
                                        if all(isinstance(val, (int, float)) and 0 <= val <= 1 for val in unique_vals):
                                            return True, "probability"
                                        try:
                                            import numpy as np
                                            if all(isinstance(val, (int, np.integer)) for val in unique_vals):
                                                if len(unique_vals) <= 20:
                                                    return True, "multiclass"
                                            if all(isinstance(val, (int, float, np.number)) for val in unique_vals):
                                                return True, "regression"
                                        except ImportError:
                                            if all(isinstance(val, int) for val in unique_vals):
                                                if len(unique_vals) <= 20:
                                                    return True, "multiclass"
                                            if all(isinstance(val, (int, float)) for val in unique_vals):
                                                return True, "regression"
                                        
                                    return False, "invalid"
                                
                                # Validate based on problem type and number of columns
                                cols = list(value.columns)
                                prediction_valid = False
                                pred_type = "unknown"
                                
                                if problem_type == 'binary-probs':
                                    if num_cols == 3:
                                        # 3 columns: ID + prob_0 + prob_1
                                        prob_cols = 0
                                        for col in cols:
                                            is_valid, p_type = is_valid_prediction_column(value[col], 'binary-probs')
                                            if is_valid:
                                                prob_cols += 1
                                        if prob_cols >= 2:  # At least 2 probability columns
                                            prediction_valid = True
                                            pred_type = "binary-probs-3col"
                                    elif num_cols == 2:
                                        # 2 columns: prob_0 + prob_1
                                        prob_cols = 0
                                        for col in cols:
                                            is_valid, p_type = is_valid_prediction_column(value[col], 'binary-probs')
                                            if is_valid:
                                                prob_cols += 1
                                        if prob_cols == 2:  # Both are probability columns
                                            prediction_valid = True
                                            pred_type = "binary-probs-2col"
                                    elif num_cols == 1:
                                        # 1 column: single probability
                                        is_valid, pred_type = is_valid_prediction_column(value[cols[0]], 'binary-probs')
                                        if is_valid:
                                            prediction_valid = True
                                            pred_type = "binary-probs-1col"
                                else:
                                    # For binary-acc, regression, multiclass
                                    if num_cols == 2:
                                        # Two columns: check if one looks like predictions
                                        for col in cols:
                                            is_valid, pred_type = is_valid_prediction_column(value[col], problem_type)
                                            if is_valid:
                                                prediction_valid = True
                                                break
                                    elif num_cols == 1:
                                        # Single column: check if it contains valid predictions
                                        is_valid, pred_type = is_valid_prediction_column(value[cols[0]], problem_type)
                                        if is_valid:
                                            prediction_valid = True
                                
                                # Check if validation passed
                                if prediction_valid:
                                    curr_reward = reward_map['create_submission_dataframe']
                                    feedback_msg = HumanMessage(content=f"Human Feedback: Verified that the submission DataFrame was created successfully with {num_cols} columns ({pred_type}) and no missing values.")
                                    print(f"Create submission reward granted - Found submission DataFrame with {num_cols} columns: {cols}, prediction type: {pred_type}")
                                    break
                                else:
                                    if problem_type == 'binary-probs':
                                        expected_cols = "1 (single prob), 2 (prob_0 + prob_1), or 3 (ID + prob_0 + prob_1)"
                                    else:
                                        expected_cols = "1 (prediction) or 2 (ID + prediction)"
                                    feedback_msg = HumanMessage(content=f"Human Feedback: Submission DataFrame validation failed for {problem_type}. Has {num_cols} columns, expected {expected_cols}.")
                                    print(f"DataFrame validation failed for {problem_type}: {num_cols} columns, expected {expected_cols}")
                                    break
                            else:
                                feedback_msg = HumanMessage(content=f"Human Feedback: Submission DataFrame creation failed. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
                                print(f"DataFrame contains missing values")
                                break
            except Exception as e:
                feedback_msg = HumanMessage(content=f"Human Feedback: Unable to verify submission DataFrame creation: {e}. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
                print(f"Unable to parse scratchpad for submission DataFrame: {e}")
        else:
            feedback_msg = HumanMessage(content=f"The trajectory has already been solved. No further rewards will be granted.")
            curr_reward = 0
        # elif(traj_rewards['create_submission_file'] == 0):
        #     try:
        #         if('Applied save_dataframe_to_csv' in output_msg[1].content):
        #             curr_reward['create_submission_file'] = reward_map['create_submission_file']
        #             feedback_msg = HumanMessage(content=f"Human feedback: Verified that the submission csv was written to disk successfully")
        #         else:
        #             feedback_msg = HumanMessage(content=f"Human feedback: Verified that the submission csv was NOT written to disk successfully")
        #     except:
        #         feedback_msg = HumanMessage(content=f"Human feedback: Verified that the submission csv was NOT written to disk successfully")
        #         print(f"combine_train_test not applied successfully")
        all_candidate_rewards.append(curr_reward)
        all_candidate_feedback_msgs.append(feedback_msg)

    all_candidate_stages_solved = [next_stage  if(r > 0) else None for r in all_candidate_rewards]
    return all_candidate_rewards, all_candidate_feedback_msgs, all_candidate_stages_solved

# Provides a verifiable reward alongside a human reflection based on the reward
def verifiable_rewards_outcome_based_rollout(output_messages, traj_rewards, new_node_scratchpads, traj_stages_solved, tool_docs, problem_type):
    penalty = 0
    reward_map = {
        'modeling': 0,
        'create_submission_dataframe': 2,
    }
    STAGES = ['init', 'modeling', 'create_submission_dataframe']
    max_stage_index = max([STAGES.index(stage) for stage in traj_stages_solved])
    next_stage = STAGES[max_stage_index + 1]

    all_candidate_rewards = []
    all_candidate_feedback_msgs = []
    # import ipdb; ipdb.set_trace()
    for i, output_msg in enumerate(output_messages):
        curr_reward = 0
        
        if(next_stage == 'modeling'):
            # import ipdb; ipdb.set_trace()
            feedback_msg = HumanMessage(content=f"Modeling is still in progress")
            try:
                tool_message_content = output_msg[1].content
                # Check for tune_{model}_{type} pattern
                models = ['catboost', 'lightgbm', 'logistic', 'xgboost', 'random_forest']
                types = ['regressor', 'classifier']
                tuning_success = False
                for model in models:
                    for model_type in types:
                        if f'Applied tune_{model}_{model_type}' in tool_message_content:
                            tuning_success = True
                            print(f"Modeling reward granted - Found successful tuning: tune_{model}_{model_type}")
                            break
                    if tuning_success:
                        break
                if tuning_success:
                    # import ipdb; ipdb.set_trace()
                    temp_scratch = new_node_scratchpads[i]
                    scale_factor = 2.5
                    base_reward = reward_map['modeling']
                    cv_bonus = 0

                    if 'best_cv_score' in temp_scratch:
                        cv_score = temp_scratch['best_cv_score'].item()
                        if pd.isna(cv_score):
                            pass
                        else:
                            cv_bonus = cv_score * scale_factor  # Bonus for valid CV score

                    curr_reward = base_reward + cv_bonus
                    # curr_reward['modeling'] = reward_map['modeling'] + (temp_scratch['best_cv_score'].item()*scale_factor if 'best_cv_score' in temp_scratch else 0)
                    feedback_msg = HumanMessage(content=f"Human Feedback: Verified that the modeling was successful")
            except:
                feedback_msg = HumanMessage(content=f"Modeling is still in progress. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
                print(f"Unable to parse ToolMessage for modeling tuning")
        elif(next_stage == 'create_submission_dataframe'):
            feedback_msg = HumanMessage(content=f"Human Feedback: submission DataFrame was NOT created successfully. Please check the signature of the wrapped function if any, and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
            try:
                temp_scratch = new_node_scratchpads[i]
                if temp_scratch:
                    for key, value in temp_scratch.items():
                        # Check if value is a DataFrame
                        if hasattr(value, 'columns') and hasattr(value, 'shape'):
                            # Check if DataFrame has no NaNs
                            if not value.isnull().any().any():
                                num_cols = len(value.columns)
                                
                                def is_valid_prediction_column(col_data, expected_type=None):
                                    """Check if column contains valid prediction values based on problem type"""
                                    unique_vals = col_data.unique()
                                    
                                    if expected_type == 'binary-acc':
                                        # Binary classification: True/False or 0/1
                                        if all(val in [True, False] for val in unique_vals):
                                            return True, "boolean"
                                        if all(val in [0, 1] for val in unique_vals):
                                            return True, "binary"
                                        return False, "invalid"
                                    
                                    elif expected_type == 'binary-probs':
                                        # Probability values between 0 and 1
                                        if all(isinstance(val, (int, float)) and 0 <= val <= 1 for val in unique_vals):
                                            return True, "probability"
                                        return False, "invalid"
                                    
                                    elif expected_type == 'regression':
                                        # Regression: numeric values (floats/ints)
                                        try:
                                            import numpy as np
                                            if all(isinstance(val, (int, float, np.number)) for val in unique_vals):
                                                return True, "regression"
                                        except ImportError:
                                            if all(isinstance(val, (int, float)) for val in unique_vals):
                                                return True, "regression"
                                        return False, "invalid"
                                    
                                    elif expected_type == 'multiclass':
                                        # Multi-class classification: set of integers
                                        try:
                                            import numpy as np
                                            if all(isinstance(val, (int, np.integer)) for val in unique_vals):
                                                if len(unique_vals) <= 50:  # Reasonable number of classes
                                                    return True, "multiclass"
                                        except ImportError:
                                            if all(isinstance(val, int) for val in unique_vals):
                                                if len(unique_vals) <= 50:  # Reasonable number of classes
                                                    return True, "multiclass"
                                        return False, "invalid"
                                    
                                    else:
                                        # General validation (backward compatibility)
                                        if all(val in [True, False] for val in unique_vals):
                                            return True, "boolean"
                                        if all(val in [0, 1] for val in unique_vals):
                                            return True, "binary"
                                        if all(isinstance(val, (int, float)) and 0 <= val <= 1 for val in unique_vals):
                                            return True, "probability"
                                        try:
                                            import numpy as np
                                            if all(isinstance(val, (int, np.integer)) for val in unique_vals):
                                                if len(unique_vals) <= 20:
                                                    return True, "multiclass"
                                            if all(isinstance(val, (int, float, np.number)) for val in unique_vals):
                                                return True, "regression"
                                        except ImportError:
                                            if all(isinstance(val, int) for val in unique_vals):
                                                if len(unique_vals) <= 20:
                                                    return True, "multiclass"
                                            if all(isinstance(val, (int, float)) for val in unique_vals):
                                                return True, "regression"
                                        
                                    return False, "invalid"
                                
                                # Validate based on problem type and number of columns
                                cols = list(value.columns)
                                prediction_valid = False
                                pred_type = "unknown"
                                
                                if problem_type == 'binary-probs':
                                    if num_cols == 3:
                                        # 3 columns: ID + prob_0 + prob_1
                                        prob_cols = 0
                                        for col in cols:
                                            is_valid, p_type = is_valid_prediction_column(value[col], 'binary-probs')
                                            if is_valid:
                                                prob_cols += 1
                                        if prob_cols >= 2:  # At least 2 probability columns
                                            prediction_valid = True
                                            pred_type = "binary-probs-3col"
                                    elif num_cols == 2:
                                        # 2 columns: prob_0 + prob_1
                                        prob_cols = 0
                                        for col in cols:
                                            is_valid, p_type = is_valid_prediction_column(value[col], 'binary-probs')
                                            if is_valid:
                                                prob_cols += 1
                                        if prob_cols == 2:  # Both are probability columns
                                            prediction_valid = True
                                            pred_type = "binary-probs-2col"
                                    elif num_cols == 1:
                                        # 1 column: single probability
                                        is_valid, pred_type = is_valid_prediction_column(value[cols[0]], 'binary-probs')
                                        if is_valid:
                                            prediction_valid = True
                                            pred_type = "binary-probs-1col"
                                else:
                                    # For binary-acc, regression, multiclass
                                    if num_cols == 2:
                                        # Two columns: check if one looks like predictions
                                        for col in cols:
                                            is_valid, pred_type = is_valid_prediction_column(value[col], problem_type)
                                            if is_valid:
                                                prediction_valid = True
                                                break
                                    elif num_cols == 1:
                                        # Single column: check if it contains valid predictions
                                        is_valid, pred_type = is_valid_prediction_column(value[cols[0]], problem_type)
                                        if is_valid:
                                            prediction_valid = True
                                
                                # Check if validation passed
                                if prediction_valid:
                                    curr_reward = reward_map['create_submission_dataframe']
                                    feedback_msg = HumanMessage(content=f"Human Feedback: Verified that the submission DataFrame was created successfully with {num_cols} columns ({pred_type}) and no missing values.")
                                    print(f"Create submission reward granted - Found submission DataFrame with {num_cols} columns: {cols}, prediction type: {pred_type}")
                                    break
                                else:
                                    if problem_type == 'binary-probs':
                                        expected_cols = "1 (single prob), 2 (prob_0 + prob_1), or 3 (ID + prob_0 + prob_1)"
                                    else:
                                        expected_cols = "1 (prediction) or 2 (ID + prediction)"
                                    feedback_msg = HumanMessage(content=f"Human Feedback: Submission DataFrame validation failed for {problem_type}. Has {num_cols} columns, expected {expected_cols}.")
                                    print(f"DataFrame validation failed for {problem_type}: {num_cols} columns, expected {expected_cols}")
                                    break
                            else:
                                feedback_msg = HumanMessage(content=f"Human Feedback: Submission DataFrame creation failed. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
                                print(f"DataFrame contains missing values")
                                break
            except Exception as e:
                feedback_msg = HumanMessage(content=f"Human Feedback: Unable to verify submission DataFrame creation: {e}. Please check the signature of the wrapped function and call it with the correct arguments. The tool signature is {tool_docs[i] if(i < len(tool_docs)) else ''}")
                print(f"Unable to parse scratchpad for submission DataFrame: {e}")
        else:
            feedback_msg = HumanMessage(content=f"The trajectory has already been solved. No further rewards will be granted.")
            curr_reward = 0
        all_candidate_rewards.append(curr_reward)
        all_candidate_feedback_msgs.append(feedback_msg)
    
    all_candidate_stages_solved = [next_stage  if(r > 0) else None for r in all_candidate_rewards]
    return all_candidate_rewards, all_candidate_feedback_msgs, all_candidate_stages_solved
