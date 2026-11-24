"""Run agents on tasks and grade the results.

Output layout (one directory per attempt)::

    <out>/<agent>/<task_id>/run_<i>/
        submission.csv    written by the agent
        models/           anything the agent saves
        trajectory.json   steps logged by the agent
        result.json       grade, usage, timing, errors

``run_one`` is a single attempt; ``run_experiment`` sweeps tasks x runs in parallel
processes and writes ``summary.json`` with per-task aggregates.
"""
from __future__ import annotations

import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from mltoolbench.agents.base import RunContext
from mltoolbench.agents.registry import build_agent
from mltoolbench.grading import Grade, aggregate, grade_submission
from mltoolbench.llm import ModelConfig
from mltoolbench.tasks import get_task


@dataclass
class Experiment:
    agent: str | dict[str, Any]
    tasks: list[str]
    llm: ModelConfig | None
    out_dir: Path
    n_runs: int = 10
    workers: int = 1
    verbose: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def agent_name(self) -> str:
        """The agent's ``name`` attribute (``my_pkg.mod:MyAgent`` -> ``"my_agent"``)."""
        return build_agent(self.agent).name


def run_one(agent_spec: str | dict[str, Any], task_id: str, run_dir: Path, llm: ModelConfig | None,
            run_index: int = 0, verbose: bool = False) -> dict[str, Any]:
    """One attempt: build the agent, run it, save the trajectory, grade the submission."""
    task = get_task(task_id)
    ctx = RunContext(task, run_dir, llm, run_id=f"run_{run_index}", seed=run_index, verbose=verbose)
    started = time.time()
    error, stats = None, None
    try:
        stats = build_agent(agent_spec).run(ctx)
    except Exception:  # an agent crash is a failed run, never a crashed sweep
        error = traceback.format_exc()
        ctx.log("error", traceback=error)
    elapsed = time.time() - started

    grade = grade_submission(task, ctx.submission_path) if (task.private_dir / "answers.csv").exists() \
        else Grade(task.id, valid=False, reason="no answers.csv: run `mltoolbench prepare` first")
    result = {
        "task_id": task.id,
        "agent": agent_spec,
        "run": run_index,
        "model": llm.model if llm else None,
        "grade": grade.to_dict(),
        "usage": ctx.usage.total.to_dict() if ctx.usage else None,
        "max_call_usage": ctx.usage.max_call.to_dict() if ctx.usage else None,
        "wall_time_s": round(elapsed, 2),
        "agent_stats": stats,
        "error": error,
    }
    _write_json(ctx.run_dir / "trajectory.json", {"task_id": task.id, "steps": ctx.trajectory})
    _write_json(ctx.run_dir / "result.json", result)
    return result


def run_experiment(exp: Experiment) -> dict[str, Any]:
    root = Path(exp.out_dir) / exp.agent_name  # building the agent here fails fast on a bad spec
    jobs = [(t, i) for t in exp.tasks for i in range(exp.n_runs)]
    results: list[dict[str, Any]] = []

    def job_args(task_id: str, i: int):
        return exp.agent, task_id, root / task_id / f"run_{i}", exp.llm, i, exp.verbose

    if exp.workers <= 1:
        for t, i in jobs:
            results.append(run_one(*job_args(t, i)))
            _report(results[-1])
    else:
        with ProcessPoolExecutor(max_workers=exp.workers) as pool:
            futures = [pool.submit(run_one, *job_args(t, i)) for t, i in jobs]
            for future in as_completed(futures):
                results.append(future.result())
                _report(results[-1])

    summary = summarize(results, exp.n_runs)
    summary["agent_name"] = root.name
    summary["config"] = {"agent": exp.agent, "model": _public_model_config(exp.llm), "n_runs": exp.n_runs,
                         "tasks": exp.tasks, **exp.extra}
    _write_json(root / "summary.json", summary)
    return summary


def summarize(results: list[dict[str, Any]], n_runs: int) -> dict[str, Any]:
    by_task: dict[str, list[Grade]] = {}
    costs: dict[str, float] = {}
    for r in results:
        by_task.setdefault(r["task_id"], []).append(Grade(**r["grade"]))
        cost = (r.get("usage") or {}).get("cost_usd")
        if cost is not None:
            costs[r["task_id"]] = costs.get(r["task_id"], 0.0) + cost
    per_task = []
    for task_id, grades in sorted(by_task.items()):
        row = aggregate(get_task(task_id), grades, n_runs)
        row["cost_usd"] = costs.get(task_id)
        per_task.append(row)
    return {"per_task": per_task, "total_cost_usd": sum(costs.values()) if costs else None}


def _public_model_config(llm: ModelConfig | None) -> dict[str, Any] | None:
    """The model config as recorded in results: never the API key or a factory object."""
    if llm is None:
        return None
    return {k: v for k, v in asdict(llm).items() if k not in ("api_key", "factory")}


def _report(result: dict[str, Any]) -> None:
    g = result["grade"]
    status = f"metric={g['metric']:.4f} pct={g['percentile']}" if g["valid"] else f"invalid: {g['reason']}"
    print(f"[{result['task_id']} run {result['run']}] {status}" + (" (agent error)" if result["error"] else ""))


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))
