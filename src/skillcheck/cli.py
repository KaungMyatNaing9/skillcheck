"""Command-line interface.

`skillcheck run` writes a results document. `skillcheck report` recomputes the
verdict and badge from the run records in that document. `skillcheck tasks list`
prints the tasks in a suite.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from skillcheck import __version__
from skillcheck.errors import SkillCheckError
from skillcheck.report import finalize_document, render_markdown
from skillcheck.runner import RunConfig, run_evaluation
from skillcheck.tasks import load_suite


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns a process exit code."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        return code if isinstance(code, int) else 1
    try:
        return int(args.handler(args))
    except SkillCheckError as exc:
        print(f"skillcheck: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="skillcheck",
        description=(
            "Measure whether a skill or instruction file changes a coding agent's "
            "pass rate on your tasks. The verdict is helps, hurts, or no detectable "
            "effect only when the statistics support it."
        ),
    )
    parser.add_argument("--version", action="version", version=f"skillcheck {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run a baseline vs treatment evaluation")
    run.add_argument("--skill", action="append", required=True, help="Skill file, directory, or PATH::DEST")
    run.add_argument("--tasks", required=True, help="Suite file, suite directory, or 'builtin'")
    run.add_argument("--agent", required=True, help="fake, replay, command, or openai")
    run.add_argument("--trials", type=int, required=True, help="Trials per task per condition")
    run.add_argument("--seed", type=int, default=0, help="Bootstrap seed (default: 0)")
    run.add_argument("--bootstrap", type=int, default=4000, help="Bootstrap draws (default: 4000, minimum 200)")
    run.add_argument("--confidence", type=float, default=0.95, help="Interval level (default: 0.95)")
    run.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="Negligible pass-rate band, as a fraction (default: 0.05 = 5 percentage points)",
    )
    run.add_argument("--timeout", type=float, default=None, help="Override each task timeout, in seconds")
    run.add_argument("--agent-opt", action="append", default=[], help="Adapter option key=value")
    run.add_argument("--out", default="skillcheck-results", help="Directory for results, report, and badge")
    run.add_argument("--verbose", action="store_true", help="Print each trial and keep traceback on agent crashes")
    run.add_argument("--keep", action="store_true", help="Keep sandbox directories")
    run.set_defaults(handler=_run)

    report = subparsers.add_parser("report", help="Recompute and print a report from results.json")
    report.add_argument("results", help="Path to results.json")
    report.add_argument("--badge", default=None, help="Also write a shields.io JSON badge to this path")
    report.add_argument("--format", choices=("md", "json"), default="md", help="Output format (default: md)")
    report.set_defaults(handler=_report)

    tasks = subparsers.add_parser("tasks", help="Inspect task suites")
    task_sub = tasks.add_subparsers(dest="tasks_command", required=True)
    listing = task_sub.add_parser("list", help="List tasks in a suite")
    listing.add_argument("--tasks", default="builtin", help="Suite file, directory, or 'builtin'")
    listing.set_defaults(handler=_list)
    return parser


def _run(args: argparse.Namespace) -> int:
    document = run_evaluation(
        RunConfig(
            tasks=args.tasks,
            skills=list(args.skill),
            agent=args.agent,
            trials=args.trials,
            seed=args.seed,
            confidence=args.confidence,
            bootstrap_samples=args.bootstrap,
            negligible_margin=args.margin,
            timeout=args.timeout,
            agent_options=_parse_options(args.agent_opt),
            verbose=args.verbose,
            keep=args.keep,
            output_dir=Path(args.out),
        )
    )
    verdict = document["verdict"]
    overall = document["overall"]
    if not isinstance(verdict, dict) or not isinstance(overall, dict):
        raise SkillCheckError("Internal error: malformed results.")
    pass_rate = overall["pass_rate_difference"]
    if not isinstance(pass_rate, dict):
        raise SkillCheckError("Internal error: malformed pass-rate interval.")
    print(f"Verdict: {verdict['label']}")
    print(str(verdict["reason"]))
    print(
        "Pass rate: "
        f"{float(pass_rate['estimate']) * 100:+.1f} pp "
        f"[{float(pass_rate['ci_low']) * 100:+.1f}, {float(pass_rate['ci_high']) * 100:+.1f}]"
    )
    warnings = document.get("warnings") or []
    if isinstance(warnings, list):
        for warning in warnings:
            print(f"warning: {warning}", file=sys.stderr)
    out = Path(args.out)
    print(f"Wrote {out / 'results.json'}")
    print(f"Wrote {out / 'report.md'}")
    print(f"Wrote {out / 'badge.json'}")
    return 0


def _report(args: argparse.Namespace) -> int:
    path = Path(args.results)
    if not path.is_file():
        raise SkillCheckError(f"Results file not found: {path}")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SkillCheckError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(loaded, dict):
        raise SkillCheckError(f"{path}: results must be a JSON object.")
    document = finalize_document(loaded)
    if args.badge:
        badge = document["badge"]
        if not isinstance(badge, dict):
            raise SkillCheckError("Internal error: missing badge.")
        badge_path = Path(args.badge)
        badge_path.write_text(json.dumps(badge, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.format == "json":
        print(json.dumps(document, indent=2, ensure_ascii=False))
    else:
        print(render_markdown(document), end="")
    return 0


def _list(args: argparse.Namespace) -> int:
    print(format_task_list(args.tasks), end="")
    return 0


def format_task_list(spec: str) -> str:
    """Return the human-readable task list for a suite spec."""
    suite = load_suite(spec)
    lines = [f"{suite.name} ({len(suite.tasks)} tasks)"]
    if suite.description:
        lines.append(suite.description)
    for task in suite.tasks:
        summary = task.prompt.strip().splitlines()[0]
        lines.append(f"  {task.id}: {summary}")
    return "\n".join(lines) + "\n"


def _parse_options(items: list[str] | None) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise SkillCheckError(f"Agent option must be key=value, got {item!r}.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise SkillCheckError("Agent option name is empty.")
        if key in parsed:
            raise SkillCheckError(f"Agent option {key} was given twice.")
        parsed[key] = value
    return parsed
