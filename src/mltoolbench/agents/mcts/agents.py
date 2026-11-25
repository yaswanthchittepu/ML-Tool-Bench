"""The tree-search baselines: LATS, MCTS-Outcome, MCTS-Shaped and Hierarchical MCTS."""
from __future__ import annotations

import random
from typing import Any, Literal

from mltoolbench.agents.base import Agent, RunContext
from mltoolbench.agents.mcts.config import MCTSConfig
from mltoolbench.agents.mcts.expansion import FRAMEWORK_WRITES_SUBMISSION, Expander, mcts_tools
from mltoolbench.agents.mcts.prompts import LATS_SYSTEM, MCTS_SYSTEM, stage_prompt
from mltoolbench.agents.mcts.rewards import STAGES, StageVerifier, problem_type
from mltoolbench.agents.mcts.search import ReflectionJudge, flat_search, hierarchical_search
from mltoolbench.env import tools_for_stage


class _TreeSearchAgent(Agent):
    #: Default settings for this agent; keyword arguments override them.
    DEFAULTS: dict[str, Any] = {}

    def __init__(self, **options: Any):
        self.config = MCTSConfig.build(self.DEFAULTS, **options)

    def _rng(self, ctx: RunContext) -> random.Random:
        return random.Random(ctx.seed)

    def _verifier(self, ctx: RunContext, mode: str) -> StageVerifier:
        return StageVerifier(problem_type(ctx.task), mode)

    @staticmethod
    def _write(ctx: RunContext, submission) -> bool:
        if submission is None:
            return False
        submission.to_csv(ctx.submission_path, index=False)
        return True


class LATSAgent(_TreeSearchAgent):
    """Language Agent Tree Search: an LLM judge scores each step (Zhou et al., 2023)."""

    name = "lats"
    DEFAULTS = {"max_depth": 30, "step_budget": 500, "reduce_context_on_error": False, "use_feedback": False}

    def run(self, ctx: RunContext) -> dict[str, Any]:
        cfg = self.config
        tools = tools_for_stage(None)  # LATS saves its own submission with save_dataframe_to_csv
        expander = Expander(ctx, LATS_SYSTEM, bound=tools, executable=tools, parallel_calls=cfg.allow_parallel_candidates,
                            reduce_context_on_error=cfg.reduce_context_on_error)
        result = flat_search(ctx, cfg, expander, judge=ReflectionJudge(ctx), rng=self._rng(ctx))
        # A solved LATS node is usually a final text answer with no dataframe; then the
        # submission is whatever the agent saved itself with save_dataframe_to_csv.
        wrote = self._write(ctx, result.submission)
        return {**result.stats, "framework_wrote_submission": wrote}


class MCTSAgent(_TreeSearchAgent):
    """MCTS with verifiable rewards: ``reward="shaped"`` (every pipeline stage) or ``"outcome"``."""

    name = "mcts"
    DEFAULTS = {"max_depth": 50, "step_budget": 750}

    def __init__(self, reward: Literal["shaped", "outcome"] = "shaped", **options: Any):
        if reward not in ("shaped", "outcome"):
            raise ValueError(f"reward must be 'shaped' or 'outcome', got {reward!r}")
        self.reward = reward
        super().__init__(**options)

    def run(self, ctx: RunContext) -> dict[str, Any]:
        cfg = self.config
        tools = mcts_tools()
        expander = Expander(ctx, MCTS_SYSTEM, bound=tools, executable=tools, parallel_calls=cfg.allow_parallel_candidates,
                            reduce_context_on_error=cfg.reduce_context_on_error, excluded=[FRAMEWORK_WRITES_SUBMISSION])
        result = flat_search(ctx, cfg, expander, verifier=self._verifier(ctx, self.reward), rng=self._rng(ctx))
        return {**result.stats, "framework_wrote_submission": self._write(ctx, result.submission)}


class MCTSShapedAgent(MCTSAgent):
    name = "mcts_shaped"

    def __init__(self, **options: Any):
        super().__init__("shaped", **options)


class MCTSOutcomeAgent(MCTSAgent):
    name = "mcts_outcome"

    def __init__(self, **options: Any):
        super().__init__("outcome", **options)


class HierarchicalMCTSAgent(_TreeSearchAgent):
    """One MCTS per pipeline stage, with stage-specific prompts and tool masking."""

    name = "hierarchical_mcts"
    DEFAULTS = {"step_budget": 750, "allow_parallel_candidates": False}

    def run(self, ctx: RunContext) -> dict[str, Any]:
        cfg = self.config
        all_tools = mcts_tools()
        expanders = {}
        for stage in STAGES:
            stage_tools = mcts_tools(stage)
            expanders[stage] = Expander(
                ctx, stage_prompt(stage),
                bound=stage_tools if cfg.tool_masking else all_tools,
                executable=stage_tools,  # calls are always executed against the stage's tools
                parallel_calls=cfg.allow_parallel_candidates, reduce_context_on_error=cfg.reduce_context_on_error,
                excluded=[FRAMEWORK_WRITES_SUBMISSION])
        result = hierarchical_search(ctx, cfg, expanders, self._verifier(ctx, "hierarchical"), self._rng(ctx))
        return {**result.stats, "framework_wrote_submission": self._write(ctx, result.submission)}
