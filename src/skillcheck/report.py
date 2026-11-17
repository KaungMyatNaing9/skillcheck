"""Markdown, JSON, and shields.io badge output.

`finalize_document` recomputes the verdict, intervals, and badge from the run
records. A hand-edited summary or badge in a results file does not survive
`skillcheck report`.
"""

from __future__ import annotations

from dataclasses import asdict

from skillcheck.errors import SkillCheckError
from skillcheck.formatting import format_confidence, format_pp, format_signed
from skillcheck.models import RunRecord, run_from_dict
from skillcheck.stats import (
    HELPS,
    HURTS,
    NO_EFFECT,
    Interval,
    OptionalInterval,
    TaskSummary,
    Verdict,
    analyze,
)

DISCLAIMER = (
    "This verdict is the pass-rate difference on this suite, for this agent, in this run. "
    "It is not a general claim that the skill improves coding agents."
)

_DERIVED = {
    "per_task",
    "overall",
    "verdict",
    "warnings",
    "eval_cost",
    "badge",
    "disclaimer",
}

_REQUIRED = (
    "schema_version",
    "skillcheck_version",
    "created_at",
    "seed",
    "confidence",
    "bootstrap_samples",
    "negligible_margin",
    "trials",
    "suite",
    "agent",
    "skills",
    "runs",
    "eval_wall_time_seconds",
)


def finalize_document(document: dict[str, object]) -> dict[str, object]:
    """Validate a results document and replace every derived field from its runs."""
    if not isinstance(document, dict):
        raise SkillCheckError("Results must be a JSON object.")
    missing = [key for key in _REQUIRED if key not in document]
    if missing:
        raise SkillCheckError(f"Results are missing fields: {', '.join(missing)}.")
    if document["schema_version"] != 1:
        raise SkillCheckError(
            f"Unsupported results schema {document['schema_version']!r}. This SkillCheck reads schema 1."
        )
    raw_runs = document["runs"]
    if not isinstance(raw_runs, list) or not raw_runs:
        raise SkillCheckError("Results contain no runs.")
    runs = [run_from_dict(item) for item in raw_runs]
    seed = _as_int(document["seed"], "seed")
    confidence = _as_float(document["confidence"], "confidence")
    samples = _as_int(document["bootstrap_samples"], "bootstrap_samples")
    margin = _as_float(document["negligible_margin"], "negligible_margin")
    wall = _as_float(document["eval_wall_time_seconds"], "eval_wall_time_seconds")
    analysis = analyze(
        runs,
        seed=seed,
        confidence=confidence,
        bootstrap_samples=samples,
        margin=margin,
    )
    stored_label = _stored_verdict_label(document.get("verdict"))
    warnings = list(analysis.warnings)
    if stored_label and stored_label != analysis.verdict.label:
        warnings.append("The stored verdict did not match the runs and was recomputed.")
    finalized: dict[str, object] = {key: value for key, value in document.items() if key not in _DERIVED}
    finalized["runs"] = [asdict(run) for run in runs]
    finalized["per_task"] = [_task_dict(task) for task in analysis.per_task]
    finalized["overall"] = {
        "pass_rate_difference": _interval_dict(analysis.pass_rate),
        "token_difference": _optional_dict(analysis.tokens),
        "cost_difference": _optional_dict(analysis.cost),
        "wall_time_difference": _optional_dict(analysis.wall_time),
    }
    finalized["verdict"] = _verdict_dict(analysis.verdict)
    finalized["warnings"] = warnings
    finalized["eval_wall_time_seconds"] = wall
    finalized["eval_cost"] = summarize_eval_cost(runs, wall)
    finalized["badge"] = build_badge(
        analysis.verdict.label,
        analysis.pass_rate,
        confidence,
    )
    finalized["disclaimer"] = DISCLAIMER
    return finalized


def build_badge(label: str, interval: Interval, confidence: float) -> dict[str, object]:
    """Shields.io endpoint JSON derived only from the pass-rate interval and verdict.

    Directional messages are used only for helps and hurts, where the interval
    excludes zero and the outlier check agreed. Inconclusive results do not
    lead with a pass-rate claim.
    """
    level = format_confidence(confidence)
    window = f"[{level}% CI {format_pp(interval.ci_low)}, {format_pp(interval.ci_high)}]"
    if label == HELPS:
        message = f"{format_pp(interval.estimate)} pp pass rate {window}"
        color = "brightgreen"
    elif label == HURTS:
        message = f"{format_pp(interval.estimate)} pp pass rate {window}"
        color = "red"
    elif label == NO_EFFECT:
        message = f"no detectable effect {window}"
        color = "blue"
    else:
        message = f"inconclusive {window}"
        color = "yellow"
    return {"schemaVersion": 1, "label": "SkillCheck", "message": message, "color": color}


def render_markdown(document: dict[str, object]) -> str:
    """Render a finalized results document."""
    for key in ("verdict", "overall", "per_task", "eval_cost", "suite", "agent"):
        if key not in document:
            raise SkillCheckError(f"Results are missing {key}. Recompute them with skillcheck report.")
    verdict = _expect_dict(document["verdict"], "verdict")
    overall = _expect_dict(document["overall"], "overall")
    per_task = document["per_task"]
    if not isinstance(per_task, list):
        raise SkillCheckError("per_task must be a list.")
    suite = _expect_dict(document["suite"], "suite")
    agent = _expect_dict(document["agent"], "agent")
    cost = _expect_dict(document["eval_cost"], "eval_cost")
    confidence = _as_float(document["confidence"], "confidence")
    level = format_confidence(confidence)
    lines = [
        "# SkillCheck report",
        "",
        f"**Verdict: {verdict['label']}**",
        "",
        str(verdict["reason"]),
        "",
        (
            f"Suite `{suite.get('name')}` · agent `{agent.get('name')}` · "
            f"{document.get('trials')} trials per condition · seed {document.get('seed')} · "
            f"{document.get('bootstrap_samples')} bootstrap samples · "
            f"negligible band ±{_as_float(document['negligible_margin'], 'negligible_margin') * 100:.1f} pp"
        ),
        "",
        "## Overall",
        "",
        f"| Metric | Estimate (treatment - baseline) | {level}% CI |",
        "| --- | --- | --- |",
        _metric_row("Pass rate", overall["pass_rate_difference"], "pp"),
        _metric_row("Tokens", overall["token_difference"], "tokens"),
        _metric_row("Cost (USD)", overall["cost_difference"], "usd"),
        _metric_row("Wall time (seconds)", overall["wall_time_difference"], "seconds"),
        "",
        "Positive pass-rate numbers mean the treatment passed more often. "
        "Positive cost, token, and time numbers mean the treatment used more.",
        "",
    ]
    lines.extend(_side_sentences(overall, level))
    lines.extend(
        [
            "## Per task",
            "",
            "| Task | Baseline | Treatment | Pass Δ | "
            f"{level}% CI | Token Δ | Cost Δ | Turns (treatment) | Tool calls (treatment) | Lines (treatment) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for task in per_task:
        if not isinstance(task, dict):
            raise SkillCheckError("Each per-task entry must be an object.")
        lines.append(_task_row(task))
    lines.extend(["", "## Outlier check", "", _outlier_paragraph(verdict), ""])
    lines.extend(_cost_section(cost))
    warnings = document.get("warnings") or []
    if isinstance(warnings, list) and warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")
    lines.extend(
        [
            "## How the numbers were computed",
            "",
            "Each task is weighted equally. Within a task, trials are paired by index and the "
            "effect is the mean of treatment minus baseline. The interval is a percentile "
            "bootstrap that resamples trials inside each task and keeps the task list fixed. "
            "It does not claim the same effect on a new task. SkillCheck says **helps** or "
            "**hurts** only when that pass-rate interval excludes zero and the direction "
            "remains after dropping the single most extreme task. It says **no detectable "
            "effect** only when the interval lies inside the negligible band. Otherwise the "
            "result is **inconclusive**.",
            "",
            str(document.get("disclaimer") or DISCLAIMER),
            "",
        ]
    )
    return "\n".join(lines)


def summarize_eval_cost(runs: list[RunRecord], eval_wall_time_seconds: float) -> dict[str, object]:
    """Sum measured eval cost. Partial sums are labeled, not presented as totals."""
    tokens_in = 0
    tokens_out = 0
    token_runs = 0
    cost = 0.0
    cost_runs = 0
    agent_time = 0.0
    for run in runs:
        agent_time += run.agent_time_seconds
        if run.tokens_in is not None and run.tokens_out is not None:
            tokens_in += run.tokens_in
            tokens_out += run.tokens_out
            token_runs += 1
        if run.cost_usd is not None:
            cost += run.cost_usd
            cost_runs += 1
    return {
        "runs": len(runs),
        "wall_time_seconds": eval_wall_time_seconds,
        "agent_time_seconds": agent_time,
        "tokens_in": tokens_in if token_runs else None,
        "tokens_out": tokens_out if token_runs else None,
        "tokens_reported_runs": token_runs,
        "tokens_are_partial": token_runs not in {0, len(runs)},
        "cost_usd": cost if cost_runs else None,
        "cost_reported_runs": cost_runs,
        "cost_is_partial": cost_runs not in {0, len(runs)},
    }


def _side_sentences(overall: dict[str, object], level: str) -> list[str]:
    sentences = [
        _direction_sentence("token", overall.get("token_difference"), "used more tokens", "used fewer tokens", level),
        _direction_sentence("cost", overall.get("cost_difference"), "cost more", "cost less", level),
        _direction_sentence(
            "wall-time",
            overall.get("wall_time_difference"),
            "took longer",
            "took less time",
            level,
        ),
    ]
    present = [sentence for sentence in sentences if sentence]
    if not present:
        return []
    spaced: list[str] = []
    for sentence in present:
        spaced.extend([sentence, ""])
    return spaced


def _direction_sentence(
    name: str,
    raw: object,
    more: str,
    less: str,
    level: str,
) -> str | None:
    if not isinstance(raw, dict) or not raw.get("available"):
        return None
    low = raw.get("ci_low")
    high = raw.get("ci_high")
    if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
        return None
    if low > 0:
        return (
            f"The {name} interval is entirely above zero: the treatment {more}. "
            "This does not change the pass-rate verdict."
        )
    if high < 0:
        return (
            f"The {name} interval is entirely below zero: the treatment {less}. "
            "This does not change the pass-rate verdict."
        )
    return (
        f"The {name} difference is not distinguishable from zero at the {level}% level. "
        "This does not change the pass-rate verdict."
    )


def _metric_row(label: str, raw: object, kind: str) -> str:
    if not isinstance(raw, dict):
        return f"| {label} | n/a | n/a |"
    if kind != "pp" and not raw.get("available", True):
        return f"| {label} | n/a | n/a |"
    estimate = raw.get("estimate")
    low = raw.get("ci_low")
    high = raw.get("ci_high")
    if not all(isinstance(value, (int, float)) for value in (estimate, low, high)):
        return f"| {label} | n/a | n/a |"
    return (
        f"| {label} | {_format_kind(float(estimate), kind)} | "
        f"[{_format_kind(float(low), kind)}, {_format_kind(float(high), kind)}] |"
    )


def _task_row(task: dict[str, object]) -> str:
    task_id = str(task["task_id"]).replace("|", "/")
    trials = int(task["trials"])
    baseline = f"{task['baseline_passes']}/{trials}"
    treatment = f"{task['treatment_passes']}/{trials}"
    pass_rate = _expect_dict(task["pass_rate"], "pass_rate")
    tokens = task.get("tokens")
    cost = task.get("cost")
    token_cell = _optional_estimate(tokens, "tokens")
    cost_cell = _optional_estimate(cost, "usd")
    turns = task.get("mean_turns_treatment")
    tools = task.get("mean_tool_calls_treatment")
    lines = task.get("mean_diff_lines_treatment")
    return (
        f"| {task_id} | {baseline} | {treatment} | "
        f"{format_pp(float(pass_rate['estimate']))} pp | "
        f"[{format_pp(float(pass_rate['ci_low']))}, {format_pp(float(pass_rate['ci_high']))}] | "
        f"{token_cell} | {cost_cell} | {_maybe_number(turns, 1)} | {_maybe_number(tools, 1)} | "
        f"{_maybe_number(lines, 1)} |"
    )


def _outlier_paragraph(verdict: dict[str, object]) -> str:
    extreme = verdict.get("extreme_task_id")
    if verdict.get("estimate_without_extreme") is None:
        return (
            f"Most extreme task: `{extreme}`. The suite has fewer than two tasks, so leaving one out is not possible."
        )
    effect = verdict.get("extreme_task_effect")
    reduced = verdict.get("estimate_without_extreme")
    effect_text = "n/a" if not isinstance(effect, (int, float)) else f"{format_pp(float(effect))} pp"
    reduced_text = "n/a" if not isinstance(reduced, (int, float)) else f"{format_pp(float(reduced))} pp"
    flag = "yes" if verdict.get("outlier_driven") else "no"
    remaining = verdict.get("verdict_without_extreme")
    return (
        f"Most extreme task: `{extreme}` ({effect_text}). "
        f"Without it, the estimate is {reduced_text} and the unadjusted verdict would be "
        f"{remaining}. Outlier-driven: {flag}."
    )


def _cost_section(cost: dict[str, object]) -> list[str]:
    runs = cost.get("runs")
    wall = cost.get("wall_time_seconds")
    agent = cost.get("agent_time_seconds")
    lines = [
        "## Cost of this evaluation",
        "",
        f"- Runs: {runs}",
        f"- Wall time: {_seconds(wall)}",
        f"- Agent time: {_seconds(agent)}",
        f"- Tokens in / out: {_token_cost_line(cost)}",
        f"- Reported model cost: {_money_line(cost)}",
        "",
    ]
    return lines


def _token_cost_line(cost: dict[str, object]) -> str:
    reported = cost.get("tokens_reported_runs")
    total = cost.get("runs")
    if not reported:
        return f"n/a (0/{total} runs reported tokens)"
    note = "partial sum, " if cost.get("tokens_are_partial") else ""
    return f"{cost.get('tokens_in')} in / {cost.get('tokens_out')} out ({note}{reported}/{total} runs reported)"


def _money_line(cost: dict[str, object]) -> str:
    reported = cost.get("cost_reported_runs")
    total = cost.get("runs")
    if not reported:
        return f"n/a (0/{total} runs reported cost)"
    amount = cost.get("cost_usd")
    text = "n/a" if not isinstance(amount, (int, float)) else f"{float(amount):.6f} USD"
    note = "partial sum, " if cost.get("cost_is_partial") else ""
    return f"{text} ({note}{reported}/{total} runs reported)"


def _optional_estimate(raw: object, kind: str) -> str:
    if not isinstance(raw, dict) or not raw.get("available"):
        return "n/a"
    estimate = raw.get("estimate")
    if not isinstance(estimate, (int, float)):
        return "n/a"
    return _format_kind(float(estimate), kind)


def _format_kind(value: float, kind: str) -> str:
    if kind == "pp":
        return f"{format_pp(value)} pp"
    if kind == "usd":
        return format_signed(value, 4)
    if kind == "seconds":
        return format_signed(value, 3)
    return format_signed(value, 1)


def _maybe_number(value: object, digits: int) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def _seconds(value: object) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value):.2f}s"


def _task_dict(task: object) -> dict[str, object]:
    if not isinstance(task, TaskSummary):
        raise SkillCheckError("Internal error: expected a task summary.")
    return {
        "task_id": task.task_id,
        "trials": task.trials,
        "baseline_passes": task.baseline_passes,
        "treatment_passes": task.treatment_passes,
        "pass_rate": _interval_dict(task.pass_rate),
        "tokens": _optional_dict(task.tokens),
        "cost": _optional_dict(task.cost),
        "wall_time": _optional_dict(task.wall_time),
        "mean_diff_lines_baseline": task.mean_diff_lines_baseline,
        "mean_diff_lines_treatment": task.mean_diff_lines_treatment,
        "mean_turns_baseline": task.mean_turns_baseline,
        "mean_turns_treatment": task.mean_turns_treatment,
        "mean_tool_calls_baseline": task.mean_tool_calls_baseline,
        "mean_tool_calls_treatment": task.mean_tool_calls_treatment,
        "agent_errors_baseline": task.agent_errors_baseline,
        "agent_errors_treatment": task.agent_errors_treatment,
    }


def _verdict_dict(verdict: Verdict) -> dict[str, object]:
    reduced = verdict.ci_without_extreme
    return {
        "label": verdict.label,
        "unadjusted_label": verdict.unadjusted_label,
        "reason": verdict.reason,
        "outlier_driven": verdict.outlier_driven,
        "extreme_task_id": verdict.extreme_task_id,
        "extreme_task_effect": verdict.extreme_task_effect,
        "estimate_without_extreme": verdict.estimate_without_extreme,
        "ci_without_extreme": None if reduced is None else _interval_dict(reduced),
        "verdict_without_extreme": verdict.verdict_without_extreme,
        "suggested_trials_per_task": verdict.suggested_trials_per_task,
        "trials_to_detect_observed_effect": verdict.trials_to_detect_observed_effect,
        "trials_for_margin_precision": verdict.trials_for_margin_precision,
    }


def _interval_dict(interval: Interval) -> dict[str, float]:
    return {"estimate": interval.estimate, "ci_low": interval.ci_low, "ci_high": interval.ci_high}


def _optional_dict(interval: OptionalInterval) -> dict[str, object]:
    return {
        "available": interval.available,
        "estimate": interval.estimate,
        "ci_low": interval.ci_low,
        "ci_high": interval.ci_high,
        "reason": interval.reason,
    }


def _stored_verdict_label(value: object) -> str | None:
    if not isinstance(value, dict):
        return None
    label = value.get("label")
    if isinstance(label, str) and label:
        return label
    return None


def _expect_dict(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise SkillCheckError(f"Results field {field} must be an object.")
    return value


def _as_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SkillCheckError(f"Results field {field} must be an integer.")
    return value


def _as_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SkillCheckError(f"Results field {field} must be a number.")
    return float(value)
