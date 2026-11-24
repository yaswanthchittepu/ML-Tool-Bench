"""Command-line interface: ``mltoolbench {list,prepare,run,grade,report}``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median
from typing import Any

import yaml

from mltoolbench.tasks import get_task, list_tasks


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    parser = argparse.ArgumentParser(prog="mltoolbench", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list tasks and agents")

    p = sub.add_parser("prepare", help="download and split competition data")
    p.add_argument("--tasks", nargs="+", default=["all"])
    p.add_argument("--from-legacy", metavar="DIR", help="import already-split data from DIR/<task>/{train,test,test_with_labels}.csv")
    p.add_argument("--no-download", action="store_true", help="split already-downloaded raw/ files")
    p.add_argument("--force", action="store_true", help="rebuild even if prepared")

    lb = sub.add_parser("leaderboards", help="(maintainers) rebuild the shipped leaderboards")
    lb.add_argument("--tasks", nargs="+", default=["all"])
    src = lb.add_mutually_exclusive_group(required=True)
    src.add_argument("--fetch", action="store_true", help="download public leaderboards via the Kaggle API")
    src.add_argument("--from", dest="from_dir", type=Path, metavar="DIR", help="import <task_id>.csv exports")

    r = sub.add_parser("run", help="run an agent on tasks and grade it")
    r.add_argument("--config", type=Path, help="YAML experiment file (see baselines/); flags override it")
    r.add_argument("--agent", help="registry name or package.module:Class")
    r.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="agent parameters, e.g. max_depth=30")
    r.add_argument("--tasks", nargs="+")
    r.add_argument("--model", help="model name (default: $OPENAI_MODEL)")
    r.add_argument("--base-url", help="OpenAI-compatible endpoint (default: $OPENAI_BASE_URL)")
    r.add_argument("--temperature", type=float)
    r.add_argument("--runs", type=int, help="attempts per task (default 10)")
    r.add_argument("--workers", type=int, help="parallel processes")
    r.add_argument("--out", type=Path, help="output directory (default: runs/<model>)")
    r.add_argument("-v", "--verbose", action="store_true")

    g = sub.add_parser("grade", help="grade one submission file")
    g.add_argument("task")
    g.add_argument("submission", type=Path)

    rep = sub.add_parser("report", help="tabulate summary.json files under a runs directory")
    rep.add_argument("runs_dir", type=Path)
    rep.add_argument("--csv", type=Path, help="also write the long-format table here")

    args = parser.parse_args(argv)
    return {"list": cmd_list, "prepare": cmd_prepare, "leaderboards": cmd_leaderboards, "run": cmd_run,
            "grade": cmd_grade, "report": cmd_report}[args.command](args)


def cmd_list(_: argparse.Namespace) -> int:
    from mltoolbench.agents.registry import AGENTS

    print("Tasks:")
    for t in map(get_task, list_tasks()):
        print(f"  {t.id:48s} {t.task_type:10s} {t.metric:16s} kaggle.com/c/{t.kaggle}")
    print("\nAgents:")
    for name in AGENTS:
        print(f"  {name}")
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    from mltoolbench.tasks.prepare import import_legacy_layout, prepare_task

    for task in map(get_task, _task_ids(args.tasks)):
        if args.from_legacy:
            import_legacy_layout(task, args.from_legacy)
        else:
            prepare_task(task, download=not args.no_download, force=args.force)
    return 0


def cmd_leaderboards(args: argparse.Namespace) -> int:
    from mltoolbench.tasks.leaderboards import fetch_leaderboard, write_leaderboard

    failed = []
    for task in map(get_task, _task_ids(args.tasks)):
        try:
            if args.fetch:
                n = fetch_leaderboard(task)
            else:
                source = args.from_dir / f"{task.id}.csv"
                if not source.exists():
                    raise FileNotFoundError(source)
                n = write_leaderboard(task, source)
            print(f"[{task.id}] {n} leaderboard entries")
        except Exception as e:  # report every task, then fail
            failed.append(task.id)
            print(f"[{task.id}] FAILED: {e}")
    return 1 if failed else 0


def cmd_run(args: argparse.Namespace) -> int:
    from mltoolbench.llm import ModelConfig
    from mltoolbench.runner import Experiment, run_experiment

    cfg: dict[str, Any] = yaml.safe_load(args.config.read_text()) if args.config else {}
    agent = cfg.get("agent", {})
    agent = {"name": agent} if isinstance(agent, str) else dict(agent)
    if args.agent:
        agent = {"name": args.agent}
    agent.update(_parse_sets(args.set))
    if "name" not in agent:
        sys.exit("error: give --agent or --config")

    model_cfg = {**cfg.get("model", {}), **{k: v for k, v in {
        "model": args.model, "base_url": args.base_url, "temperature": args.temperature}.items() if v is not None}}
    llm = ModelConfig.from_env(**model_cfg)
    exp = Experiment(
        agent=agent if len(agent) > 1 else agent["name"],
        tasks=_task_ids(args.tasks or cfg.get("tasks", ["all"])),
        llm=llm,
        out_dir=args.out or Path(cfg.get("out", "runs")) / llm.model.replace("/", "_"),
        n_runs=args.runs or cfg.get("runs", 10),
        workers=args.workers or cfg.get("workers", 1),
        verbose=args.verbose,
    )
    summary = run_experiment(exp)
    _print_table([{**row, "agent": exp.agent_name} for row in summary["per_task"]])
    return 0


def cmd_grade(args: argparse.Namespace) -> int:
    from mltoolbench.grading import grade_submission

    print(json.dumps(grade_submission(get_task(args.task), args.submission).to_dict(), indent=2))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    rows = []
    for path in sorted(args.runs_dir.rglob("summary.json")):
        summary = json.loads(path.read_text())
        name = summary.get("agent_name") or path.parent.name
        model = (summary["config"].get("model") or {}).get("model", "?")
        rows += [{**row, "agent": name, "model": model} for row in summary["per_task"]]
    if not rows:
        print(f"No summary.json under {args.runs_dir}")
        return 1
    _print_table(rows)
    if args.csv:
        import pandas as pd

        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")
    return 0


def _print_table(rows: list[dict[str, Any]]) -> None:
    """Per task: median percentile (consistency). Last row: medians across tasks, as in the paper."""
    agents = sorted({r["agent"] for r in rows})
    tasks = sorted({r["task_id"] for r in rows})
    cell = {(r["task_id"], r["agent"]): r for r in rows}
    width = max(len(a) for a in agents) + 2
    print(f"\n{'task':48s}" + "".join(f"{a:>{max(width, 16)}s}" for a in agents))
    for t in tasks:
        line = f"{t:48s}"
        for a in agents:
            r = cell.get((t, a))
            line += f"{_fmt(r):>{max(width, 16)}s}"
        print(line)
    line = f"{'OVERALL (median across tasks)':48s}"
    for a in agents:
        mine = [cell[(t, a)] for t in tasks if (t, a) in cell]
        pct = [r["median_percentile"] for r in mine if r["median_percentile"] is not None]
        overall = {"median_percentile": median(pct) if pct else None,
                   "consistency": median(r["consistency"] for r in mine)}
        line += f"{_fmt(overall):>{max(width, 16)}s}"
    print(line)
    print("\ncells: median leaderboard percentile (consistency)")


def _fmt(row: dict[str, Any] | None) -> str:
    if row is None:
        return "-"
    pct = row["median_percentile"]
    return f"{pct:.2f} ({row['consistency']:.1f})" if pct is not None else f"n/a ({row['consistency']:.1f})"


def _task_ids(names: list[str]) -> list[str]:
    ids = list_tasks() if names == ["all"] else names
    for t in ids:
        get_task(t)  # fail fast on typos
    return ids


def _parse_sets(pairs: list[str]) -> dict[str, Any]:
    out = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        out[key] = yaml.safe_load(value)  # "10" -> 10, "true" -> True, "[1, 2]" -> list
    return out


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


if __name__ == "__main__":
    sys.exit(main())
