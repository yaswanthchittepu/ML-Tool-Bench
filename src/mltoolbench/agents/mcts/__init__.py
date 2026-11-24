"""Tree-search baselines (LATS, MCTS-Outcome, MCTS-Shaped, Hierarchical MCTS).

All search settings live in :class:`mltoolbench.agents.mcts.config.MCTSConfig`.
"""
from mltoolbench.agents.mcts.agents import (
    HierarchicalMCTSAgent,
    LATSAgent,
    MCTSAgent,
    MCTSOutcomeAgent,
    MCTSShapedAgent,
)
from mltoolbench.agents.mcts.config import MCTSConfig

__all__ = ["HierarchicalMCTSAgent", "LATSAgent", "MCTSAgent", "MCTSConfig", "MCTSOutcomeAgent", "MCTSShapedAgent"]
