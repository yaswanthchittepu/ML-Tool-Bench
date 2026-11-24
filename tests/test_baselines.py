"""Every shipped experiment config must build its agent."""
from pathlib import Path

import pytest
import yaml

from mltoolbench.agents.registry import AGENTS, build_agent

CONFIGS = sorted(Path(__file__).parent.parent.glob("baselines/**/*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: str(p.relative_to(p.parents[1])))
def test_config_builds(path):
    cfg = yaml.safe_load(path.read_text())
    assert build_agent(cfg["agent"]).name in AGENTS


def test_every_agent_has_a_config():
    names = {yaml.safe_load(p.read_text())["agent"]["name"] for p in CONFIGS}
    assert names == set(AGENTS)
