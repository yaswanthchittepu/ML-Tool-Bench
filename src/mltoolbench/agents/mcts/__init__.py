"""Tree-search baselines (LATS, MCTS-Outcome, MCTS-Shaped, Hierarchical MCTS).

``profile="paper"`` reproduces the paper's runs; ``profile="fixed"`` (default)
corrects their bugs. See :mod:`mltoolbench.agents.mcts.config` for every flag.
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
