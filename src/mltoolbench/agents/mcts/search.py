"""Tree search loops: flat (LATS, MCTS-Outcome, MCTS-Shaped) and staged (Hierarchical MCTS)."""
from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from mltoolbench.agents.mcts.config import MCTSConfig
from mltoolbench.agents.mcts.expansion import Candidate, Expander
from mltoolbench.agents.mcts.node import Node
from mltoolbench.agents.mcts.prompts import REFLECTION_JUDGE_SYSTEM
from mltoolbench.agents.mcts.rewards import STAGES, Reflection, StageVerifier, Verdict


@dataclass
class Budget:
    limit: int
    used: int = 1  # the start step

    def take(self) -> bool:
        """Account for one more search step; False means stop and keep the best result."""
        if self.used >= self.limit:
            return False
        self.used += 1
        return True


def select(root: Node, cfg: MCTSConfig) -> Node:
    node = root
    while node.children:
        node = max(node.children, key=lambda c: c.uct(cfg.exploration_weight))
    return node


def first_frame(scratchpad: dict[str, Any]) -> pd.DataFrame | None:
    return next((v for v in scratchpad.values() if hasattr(v, "columns") and hasattr(v, "shape")), None)


def has_frame(node: Node) -> int:
    return int(first_frame(node.scratchpad) is not None)


# ============================================================================ flat search

@dataclass
class FlatResult:
    root: Node
    submission: pd.DataFrame | None
    stats: dict = field(default_factory=dict)


class ReflectionJudge:
    """LATS value function: an LLM grades each candidate trajectory 0-10."""

    def __init__(self, ctx, max_retries: int = 5):
        prompt = ChatPromptTemplate.from_messages([
            ("system", REFLECTION_JUDGE_SYSTEM), ("user", "{input}"), MessagesPlaceholder(variable_name="candidate"),
        ])
        model = ctx.chat_model().bind_tools([Reflection], tool_choice="Reflection")
        self.chain = prompt | model | PydanticToolsParser(tools=[Reflection])
        self.question = ctx.prompt()
        self.max_retries = max_retries

    def __call__(self, trajectory: list[BaseMessage], candidates: list[Candidate]) -> list[Reflection]:
        inputs = [{"input": self.question, "candidate": trajectory + c.messages} for c in candidates]
        for attempt in range(self.max_retries):
            try:
                reflections = [parsed[0] for parsed in self.chain.batch(inputs)]
                break
            except Exception:
                if attempt == self.max_retries - 1:
                    raise
        for reflection, c in zip(reflections, candidates, strict=True):
            if not isinstance(c.messages[-1], AIMessage):  # only a final answer can be a solution
                reflection.found_solution = False
        return reflections


def flat_search(ctx, cfg: MCTSConfig, expander: Expander, *, verifier: StageVerifier | None = None,
                judge: ReflectionJudge | None = None, rng: random.Random) -> FlatResult:
    """LATS when ``judge`` is given, otherwise MCTS with verifiable ``verifier`` rewards."""
    lats = judge is not None
    root = Node([], stage_solved=None if lats else "init")
    budget = Budget(cfg.step_budget)
    expansions = 0

    def done() -> bool:
        if root.is_solved or root.height > cfg.max_depth:
            return True
        return not lats and any(n.is_solved for n in root.subtree())

    while not done() and budget.take():
        leaf = select(root, cfg)
        trajectory = leaf.trajectory()
        pad = leaf.path_scratchpad()
        candidates = expander.candidates(expander.propose(trajectory), pad)
        expansions += 1
        if lats:
            for cand, refl in zip(candidates, judge(trajectory, candidates), strict=True):
                child = Node(cand.messages, leaf, cand.delta, refl.normalized_score, reflection=refl)
                if refl.found_solution:
                    child.mark_solved(ancestors=True)
        else:
            stage = verifier.next_stage(leaf.stages_solved())
            for cand in candidates:
                entry = verifier.check(stage, cand.messages, cand.delta, cand.tool_doc)
                reward = entry.reward
                if cfg.rollout_depth > 0:
                    reward = _safe_rollout(flat_rollout, entry.reward, cfg, expander, verifier, leaf, cand, entry, rng)
                messages = cand.messages + ([entry.message()] if cfg.use_feedback else [])
                child = Node(messages, leaf, cand.delta, reward - cfg.rollout_penalty * cfg.rollout_depth,
                             stage_solved=stage if entry.solved else None)
                if child.stage_solved == STAGES[-1]:
                    child.mark_solved()
        ctx.log("expansion", leaf_depth=leaf.depth, tree_height=root.height, candidates=[
            {"tool": c.tool_name, "reward": n.reward, "solved": n.is_solved}
            for c, n in zip(candidates, leaf.children[-len(candidates):], strict=True)])

    submission = _flat_submission(root, lats)
    stats = {"expansions": expansions, "tree_height": root.height, "nodes": sum(1 for _ in root.subtree()),
             "solved_nodes": sum(n.is_solved for n in root.subtree()), "budget_exhausted": budget.used >= budget.limit}
    return FlatResult(root, submission, stats)


def _flat_submission(root: Node, lats: bool) -> pd.DataFrame | None:
    nodes = list(root.subtree())
    if lats:  # best terminal solved node by value; ancestors are 'solved' too but not terminal
        best = max(nodes, key=lambda n: int(n.is_leaf and n.is_solved) * n.value)
        return first_frame(best.scratchpad) if best.is_solved else None
    solved = [n for n in nodes if n.is_solved]
    if not solved:
        return None
    best = max(solved, key=lambda n: has_frame(n) * n.trajectory_reward())
    return first_frame(best.scratchpad)


def flat_rollout(cfg, expander: Expander, verifier: StageVerifier, leaf: Node, cand: Candidate, entry: Verdict,
                 rng: random.Random) -> float:
    """Simulate up to ``rollout_depth`` further steps after ``cand``; return the summed reward."""
    messages = leaf.trajectory() + cand.messages + ([entry.message()] if cfg.use_feedback else [])
    pad = {**leaf.path_scratchpad(), **cand.delta}
    stages = leaf.stages_solved() + ([entry.stage] if entry.solved else [])
    total = entry.reward
    for _ in range(cfg.rollout_depth):
        try:
            step, pad_delta = _rollout_step(cfg, expander, messages, pad, rng)
            if step is None:
                break
            verdict = verifier.check(verifier.next_stage(stages), step.messages, pad_delta, step.tool_doc)
            messages += step.messages + ([verdict.message()] if cfg.use_feedback else [])
            pad.update(pad_delta)
            if verdict.solved and verdict.stage not in stages:
                stages.append(verdict.stage)
            total += verdict.reward
        except Exception:
            break
    return total


def _rollout_step(cfg, expander: Expander, messages, pad, rng) -> tuple[Candidate | None, dict]:
    ai = expander.propose(messages)
    if not ai.tool_calls:
        return None, {}
    call = rng.choice(ai.tool_calls) if cfg.rollout_policy == "random" else ai.tool_calls[0]
    step = expander.run_call(ai, call, pad)
    return step, step.delta


def _safe_rollout(run: Callable[..., float], fallback: float, *args: Any) -> float:
    """A failed rollout falls back to the candidate's own reward."""
    try:
        return run(*args)
    except Exception:
        return fallback


# ============================================================================ hierarchical search

@dataclass
class HierarchicalResult:
    submission: pd.DataFrame | None
    stats: dict = field(default_factory=dict)


def hierarchical_search(ctx, cfg: MCTSConfig, expanders: dict[str, Expander], verifier: StageVerifier,
                        rng: random.Random) -> HierarchicalResult:
    penalty = cfg.rollout_penalty * cfg.rollout_depth
    root_reward = 0.0
    root = Node([], reward=root_reward)
    solutions: dict[str, list[Node]] = {s: [] for s in STAGES}
    budget = Budget(cfg.step_budget)
    expansions: dict[str, int] = {s: 0 for s in STAGES}
    reached = None

    for index, stage in enumerate(STAGES):
        reached = stage
        limit = cfg.stage_depth_limits[stage]
        while True:
            if not budget.take():
                return _hier_result(solutions, expansions, reached, budget, aborted=True)
            if root.height < limit:
                _expand_stage(ctx, cfg, expanders[stage], verifier, stage, root, solutions, penalty, rng)
                expansions[stage] += 1
            if root.height >= limit:
                break
        if not solutions[stage]:
            return _hier_result(solutions, expansions, reached, budget)
        if not budget.take():
            return _hier_result(solutions, expansions, reached, budget, aborted=True)
        if index < len(STAGES) - 1:
            root = _next_stage_root(cfg, stage, solutions, root_reward, penalty, rng)
    return _hier_result(solutions, expansions, reached, budget)


def _expand_stage(ctx, cfg, expander: Expander, verifier: StageVerifier, stage: str, root: Node,
                  solutions: dict[str, list[Node]], penalty: float, rng: random.Random) -> None:
    leaf = select(root, cfg)
    trajectory = leaf.trajectory()
    pad = leaf.path_scratchpad()
    candidates = expander.candidates(expander.propose(trajectory), pad)
    use_rollout = cfg.rollout_depth > 0 and stage in cfg.rollout_stages
    for cand in candidates:
        entry = verifier.check(stage, cand.messages, cand.delta, cand.tool_doc)
        reward = entry.reward
        if use_rollout:
            reward = _safe_rollout(hierarchical_rollout, entry.reward, cfg, expander, verifier, stage, leaf, cand,
                                   entry, rng)
        solved = entry.solved
        messages = cand.messages + ([entry.message()] if cfg.use_feedback else [])
        child = Node(messages, leaf, cand.delta, reward - penalty, solved=solved)
        if solved:
            solutions[stage].append(child)
    ctx.log("expansion", stage=stage, leaf_depth=leaf.depth, tree_height=root.height, candidates=[
        {"tool": c.tool_name, "reward": n.reward, "solved": n.is_solved}
        for c, n in zip(candidates, leaf.children[-len(candidates):], strict=True)])


def hierarchical_rollout(cfg, expander, verifier, stage, leaf, cand, entry, rng) -> float:
    messages = leaf.trajectory() + cand.messages + ([entry.message()] if cfg.use_feedback else [])
    pad = {**leaf.path_scratchpad(), **cand.delta}
    total = entry.reward
    for _ in range(cfg.rollout_depth):
        try:
            step, pad_delta = _rollout_step(cfg, expander, messages, pad, rng)
            if step is None:
                break
            verdict = verifier.check(stage, step.messages, pad_delta, step.tool_doc)
            messages += step.messages + ([verdict.message()] if cfg.use_feedback else [])
            pad.update(pad_delta)
            total += verdict.reward
        except Exception:
            break
    return total


def _next_stage_root(cfg, stage: str, solutions: dict[str, list[Node]], root_reward: float, penalty: float,
                     rng: random.Random) -> Node:
    """Seed the next stage's tree with copies of this stage's solutions (and the ancestors they need)."""
    root = Node([], reward=root_reward)
    def copy(node: Node, parent: Node) -> Node:
        return Node(node.messages, parent, node.scratchpad, node.reward)

    def ancestor_below(node: Node, generations: int) -> Node:
        """Walk up while ``generations`` ancestors exist."""
        def has(n: Node) -> bool:
            for _ in range(generations):
                n = n.parent
                if n is None:
                    return False
            return True
        while has(node):
            node = node.parent
        return node

    sols = solutions[stage]
    if stage in ("data_cleaning", "feature_engineering", "split_train_test"):
        rng.shuffle(sols)
        for sol in sols[: cfg.max_carried_solutions]:
            copy(sol, root)
    elif stage in ("train_data_to_features_target", "test_data_to_features", "modeling"):
        rng.shuffle(sols)
        generations = 2 if stage == "train_data_to_features_target" else 3
        for sol in sols[: cfg.max_carried_solutions]:
            copy(sol, copy(ancestor_below(sol, generations), root))
    elif stage == "test_data_loading":
        train = max(solutions["train_data_loading"], key=lambda n: n.reward)
        copy(max(sols, key=lambda n: n.reward), copy(train, root))
    else:
        copy(max(sols, key=lambda n: n.reward), root)
    return root


def _hier_result(solutions, expansions, reached, budget, aborted=False) -> HierarchicalResult:
    final = solutions[STAGES[-1]]
    submission = None
    if final:
        best = max(final, key=lambda n: has_frame(n) * n.trajectory_reward())
        submission = first_frame(best.scratchpad)
    stats = {"stage_reached": reached, "expansions_per_stage": expansions,
             "solutions_per_stage": {s: len(v) for s, v in solutions.items()},
             "budget_exhausted": aborted}
    return HierarchicalResult(submission, stats)
