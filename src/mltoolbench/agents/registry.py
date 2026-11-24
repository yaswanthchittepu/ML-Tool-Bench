"""Name -> agent class, so experiments can be described in plain config files."""
from __future__ import annotations

import importlib
from typing import Any

from mltoolbench.agents.base import Agent

#: Baselines reported in the paper, then extras.
AGENTS: dict[str, str] = {
    "react": "mltoolbench.agents.react:ReActAgent",
    "lats": "mltoolbench.agents.mcts:LATSAgent",
    "mcts_outcome": "mltoolbench.agents.mcts:MCTSOutcomeAgent",
    "mcts_shaped": "mltoolbench.agents.mcts:MCTSShapedAgent",
    "hierarchical_mcts": "mltoolbench.agents.mcts:HierarchicalMCTSAgent",
    "react_code": "mltoolbench.agents.code_agent:ReActCodeAgent",
    # not in the paper
    "hierarchical_react": "mltoolbench.agents.react:HierarchicalReActAgent",
}


def build_agent(spec: str | dict[str, Any]) -> Agent:
    """Build from a registry name, ``"package.module:Class"``, or ``{"name": ..., **params}``."""
    params: dict[str, Any] = {}
    if isinstance(spec, dict):
        params = {k: v for k, v in spec.items() if k != "name"}
        spec = spec["name"]
    target = AGENTS.get(spec, spec)
    if ":" not in target:
        raise KeyError(f"Unknown agent {spec!r}. Built-in agents: {', '.join(AGENTS)}")
    module, cls = target.split(":")
    return getattr(importlib.import_module(module), cls)(**params)
