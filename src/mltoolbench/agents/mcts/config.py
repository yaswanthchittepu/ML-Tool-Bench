"""Search configuration shared by the tree-search agents.

Design choices common to all of them: shaped stage rewards (1/1/1/2/2/1/1/1/
2+2.5·cv/2), outcome rewards (2.5·cv for modeling, 2 for a submission), a stage counts
as solved when its reward is > 0, and a penalty of ``rollout_penalty`` per rollout
level (none at the default depth 0). Flat agents allow several tool calls per
expansion (several children); Hierarchical MCTS asks for one. MCTS agents never see
``save_dataframe_to_csv``: the framework writes the chosen node's submission. LATS
stops at the first solution its judge accepts, flat MCTS at the first solved node.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Literal

STAGE_DEPTH_LIMITS = {
    "train_data_loading": 10, "test_data_loading": 10, "combine_train_test": 10, "data_cleaning": 15,
    "feature_engineering": 20, "split_train_test": 10, "train_data_to_features_target": 10,
    "test_data_to_features": 10, "modeling": 20, "create_submission_dataframe": 10,
}


@dataclass
class MCTSConfig:
    # --- search ------------------------------------------------------------------
    exploration_weight: float = 1.0
    #: Stop once the tree is taller than this (flat agents). Defaults: LATS 30, MCTS 50.
    max_depth: int = 50
    #: Maximum search steps (one per expansion / stage transition); the search then stops with
    #: the best result found so far.
    step_budget: int = 750
    #: Append the verifier's feedback message to each child (enters the LLM context).
    use_feedback: bool = True
    #: Retry an expansion with the last 10k..5, 1 messages when the LLM call fails (e.g. context length).
    reduce_context_on_error: bool = True
    #: Let one LLM response propose several tool calls, each becoming a child.
    allow_parallel_candidates: bool = True

    # --- rollouts ----------------------------------------------------------------
    rollout_depth: int = 0
    rollout_policy: Literal["random", "greedy"] = "random"
    #: Penalty per rollout level subtracted from each child's reward.
    rollout_penalty: float = 0.1
    #: Hierarchical only: stages in which rollouts are performed.
    rollout_stages: tuple[str, ...] = ("data_cleaning", "feature_engineering", "modeling",
                                      "create_submission_dataframe")

    # --- hierarchical --------------------------------------------------------------
    stage_depth_limits: dict[str, int] = field(default_factory=lambda: dict(STAGE_DEPTH_LIMITS))
    tool_masking: bool = True
    #: Max solutions carried into the next stage for the multi-solution stages.
    max_carried_solutions: int = 10

    @classmethod
    def build(cls, defaults: dict | None = None, **overrides) -> MCTSConfig:
        """Agent defaults first, then user overrides; unknown keys raise."""
        known = {f.name for f in fields(cls)}
        merged = {**(defaults or {}), **{k: v for k, v in overrides.items() if v is not None}}
        unknown = set(merged) - known
        if unknown:
            raise TypeError(f"Unknown MCTS option(s): {sorted(unknown)}")
        if "rollout_stages" in merged:
            merged["rollout_stages"] = tuple(merged["rollout_stages"])
        return cls(**merged)

    def with_(self, **changes) -> MCTSConfig:
        return replace(self, **changes)
