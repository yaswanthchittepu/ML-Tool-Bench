"""Search-tree node shared by LATS, MCTS and Hierarchical MCTS.

Each node holds the messages of one step (the AI tool call, its ToolMessage and
optional feedback/reflection) and only the scratchpad *delta* that step produced.
The state at a node is the union of deltas along its path to the root, so
branches never share mutable state.
"""
from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterator
from typing import Any

from langchain_core.messages import BaseMessage


class Node:
    def __init__(
        self,
        messages: list[BaseMessage],
        parent: Node | None = None,
        scratchpad: dict[str, Any] | None = None,
        reward: float = 0.0,
        *,
        solved: bool = False,
        stage_solved: str | None = None,
        reflection: Any | None = None,
        backpropagate: bool = True,
    ):
        self.messages = messages
        self.parent = parent
        self.scratchpad = scratchpad or {}
        self.reward = reward
        self.stage_solved = stage_solved
        self.reflection = reflection  # LATS: a Reflection; its message joins the trajectory
        self.children: list[Node] = []
        self.value = 0.0  # running mean of rewards backpropagated through this node
        self.visits = 0
        self.depth = parent.depth + 1 if parent else 1
        self._solved = solved
        if parent is not None:
            parent.children.append(self)
        if backpropagate:
            self.backpropagate(reward)

    # ------------------------------------------------------------- MCTS bookkeeping
    def backpropagate(self, reward: float) -> None:
        node: Node | None = self
        while node:
            node.visits += 1
            node.value = (node.value * (node.visits - 1) + reward) / node.visits
            node = node.parent

    def uct(self, exploration_weight: float, *, average_value_twice: bool = False) -> float:
        """Upper confidence bound. ``value`` is already a mean; ``average_value_twice``
        reproduces the paper's code, which divided it by ``visits`` again."""
        if self.parent is None:
            raise ValueError("The root has no UCT score")
        if self.visits == 0:
            return self.value
        exploit = self.value / self.visits if average_value_twice else self.value
        return exploit + exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)

    # ------------------------------------------------------------- solved status
    @property
    def is_solved(self) -> bool:
        return self._solved

    def mark_solved(self, *, ancestors: bool = False) -> None:
        self._solved = True
        if ancestors:
            node = self.parent
            while node:
                node._solved = True
                node = node.parent

    # ------------------------------------------------------------- tree queries
    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def height(self) -> int:
        return 1 + max((c.height for c in self.children), default=0)

    def path(self) -> list[Node]:
        """Nodes from the root down to this node."""
        nodes, node = [], self
        while node:
            nodes.append(node)
            node = node.parent
        return nodes[::-1]

    def own_messages(self, include_reflection: bool = True) -> list[BaseMessage]:
        if include_reflection and self.reflection is not None:
            return self.messages + [self.reflection.as_message()]
        return self.messages

    def trajectory(self, include_reflections: bool = True) -> list[BaseMessage]:
        return [m for n in self.path() for m in n.own_messages(include_reflections)]

    def path_scratchpad(self) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for n in self.path():
            merged.update(n.scratchpad)
        return merged

    def trajectory_reward(self) -> float:
        return sum(n.reward for n in self.path())

    def stages_solved(self) -> list[str]:
        return [n.stage_solved for n in self.path() if n.stage_solved]

    def subtree(self) -> Iterator[Node]:
        """This node and all descendants, breadth first."""
        queue = deque([self])
        while queue:
            node = queue.popleft()
            yield node
            queue.extend(node.children)

    def __repr__(self) -> str:
        return f"<Node depth={self.depth} reward={self.reward:.3f} value={self.value:.3f} visits={self.visits}>"
