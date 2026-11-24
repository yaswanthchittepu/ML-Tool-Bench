"""Name -> agent class, so experiments can be described in plain config files."""
from __future__ import annotations

import importlib
import os
import sys
from typing import Any

from mltoolbench.agents.base import Agent

#: The baselines reported in the paper.
AGENTS: dict[str, str] = {
    "react": "mltoolbench.agents.react:ReActAgent",
    "lats": "mltoolbench.agents.mcts:LATSAgent",
    "mcts_outcome": "mltoolbench.agents.mcts:MCTSOutcomeAgent",
    "mcts_shaped": "mltoolbench.agents.mcts:MCTSShapedAgent",
    "hierarchical_mcts": "mltoolbench.agents.mcts:HierarchicalMCTSAgent",
    "react_code": "mltoolbench.agents.code_agent:ReActCodeAgent",
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
    if os.getcwd() not in sys.path:  # console scripts don't see the user's project directory
        sys.path.insert(0, os.getcwd())
    return getattr(importlib.import_module(module), cls)(**params)
