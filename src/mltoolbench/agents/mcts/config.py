"""Search configuration and the ``paper`` / ``fixed`` profiles.

The paper's numbers were produced by three legacy scripts: LATS by
``lats_ds_reflection_parallel.py``; MCTS-Outcome / MCTS-Shaped by
``lats_ds_verifiable_rewards_rollout_parallel.py`` and Hierarchical MCTS by
``lats_hierarchical_rollout_parallel.py``, both with rollout depth 0 (the
``*_rd_0_*`` result directories read by the paper's evaluation script).
``profile="paper"`` reproduces those scripts, quirks included; ``profile="fixed"``
(the default) corrects the quirks that are bugs. Every divergence is one flag:

=============================  ============  ===========  ==============================================
flag                           paper         fixed        why
=============================  ============  ===========  ==============================================
average_value_twice            True          False        UCT divided the running-mean ``value`` by
                                                          ``visits`` again (from the LangChain LATS
                                                          tutorial), shrinking exploitation as 1/visits.
legacy_tuner_matching          True          False        Modeling reward matched ``tune_{catboost,
                                                          lightgbm,logistic,xgboost,random_forest}_
                                                          {regressor,classifier}``: tune_linear_regressor
                                                          could never earn the modeling reward.
predict_with_best_estimator    True          False        predict_target always used
                                                          ``scratchpad['best_estimator']``, ignoring the
                                                          agent's ``model`` binding.
legacy_feedback_text           True          False        Flat rollout feedback said "more than 1000
                                                          columns" while checking > 100000.
penalize_roots                 True          False        Hierarchical: the rollout penalty skipped the
                                                          root via ``depth == 0`` (never true), so stage
                                                          roots and transition copies were penalised.
solved_from_rollout_reward     True          False        Hierarchical with rollouts: a node counted as a
                                                          stage solution if the *simulated future* earned
                                                          reward, even when its own step did not.
fail_on_step_budget            True          False        Exceeding the LangGraph recursion limit raised
                                                          and the run produced no submission; fixed stops
                                                          the search and keeps the best result so far.
=============================  ============  ===========  ==============================================

Behaviour that is a design choice, identical in both profiles: shaped stage
rewards (1/1/1/2/2/1/1/1/2+2.5·cv/2), outcome rewards (2.5·cv, 2), a stage
counts as solved when its reward is > 0, a 0.1 penalty per rollout level (none at
depth 0), flat agents allow several tool calls per expansion (several children)
while Hierarchical MCTS asks for one, MCTS agents never see ``save_dataframe_to_csv``
(the framework writes the chosen node's submission), LATS stops at the first
solution the judge accepts, flat MCTS at the first solved node.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Literal

Profile = Literal["paper", "fixed"]

STAGE_DEPTH_LIMITS = {
    "train_data_loading": 10, "test_data_loading": 10, "combine_train_test": 10, "data_cleaning": 15,
    "feature_engineering": 20, "split_train_test": 10, "train_data_to_features_target": 10,
    "test_data_to_features": 10, "modeling": 20, "create_submission_dataframe": 10,
}

_PROFILE_FLAGS = {
    "paper": dict(average_value_twice=True, legacy_tuner_matching=True, predict_with_best_estimator=True,
                  legacy_feedback_text=True, penalize_roots=True, solved_from_rollout_reward=True,
                  fail_on_step_budget=True),
    "fixed": dict(average_value_twice=False, legacy_tuner_matching=False, predict_with_best_estimator=False,
                  legacy_feedback_text=False, penalize_roots=False, solved_from_rollout_reward=False,
                  fail_on_step_budget=False),
}


@dataclass
class MCTSConfig:
    profile: Profile = "fixed"

    # --- search ------------------------------------------------------------------
    exploration_weight: float = 1.0
    #: Stop once the tree is taller than this (flat agents). Paper: LATS 30, MCTS 50.
    max_depth: int = 50
    #: LangGraph recursion limit of the paper's graphs (start + one step per expansion / transition).
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

    # --- profile-controlled flags (see module docstring) ----------------------------
    average_value_twice: bool | None = None
    legacy_tuner_matching: bool | None = None
    predict_with_best_estimator: bool | None = None
    legacy_feedback_text: bool | None = None
    penalize_roots: bool | None = None
    solved_from_rollout_reward: bool | None = None
    fail_on_step_budget: bool | None = None

    def __post_init__(self) -> None:
        if self.profile not in _PROFILE_FLAGS:
            raise ValueError(f"profile must be 'paper' or 'fixed', got {self.profile!r}")
        for name, value in _PROFILE_FLAGS[self.profile].items():
            if getattr(self, name) is None:
                setattr(self, name, value)

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
