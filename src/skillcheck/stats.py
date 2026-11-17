"""Paired, task-weighted bootstrap intervals and a conservative verdict.

Estimator
---------
For each task, pair trial *i* of the treatment with trial *i* of the baseline.
The task effect is the mean of (treatment - baseline) across those pairs.
The suite effect is the unweighted mean of the task effects, so a task with
the same trial count as the others is not double-counted, and a task with a
different trial count still has one vote.

Interval
--------
The confidence interval is a percentile bootstrap. Each draw resamples trial
pairs independently inside every task and keeps every task. Tasks are visited
in sorted task-id order, which makes the draw sequence depend only on the seed.
The interval answers: how stable is the average effect on *these* tasks across
trials? It does not by itself answer whether a different task would have
changed the result.

Outlier check
-------------
The most extreme task is the one with the largest absolute pass-rate effect
(ties break toward the smaller task id). SkillCheck recomputes the interval
without that task. If the sign of the estimate changes, or a directional
interval stops excluding zero, the result is outlier-driven and the verdict
stays inconclusive.

Verdict
-------
On the pass-rate difference only:

- ``helps`` when the interval is entirely above zero and the outlier check agrees
- ``hurts`` when the interval is entirely below zero and the outlier check agrees
- ``no detectable effect`` when the interval lies inside ±margin (default 5 pp)
  and the tasks are not cancelling each other out
- ``inconclusive`` otherwise, including fewer than 2 trials

Cost and token intervals are reported beside the verdict. They never change it.
The extra-trial suggestion assumes the standard error shrinks like 1/sqrt(n).
It is not a power simulation.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from skillcheck.errors import SkillCheckError
from skillcheck.formatting import format_confidence, format_pp
from skillcheck.models import RunRecord

MIN_BOOTSTRAP_SAMPLES = 200
HELPS = "helps"
HURTS = "hurts"
NO_EFFECT = "no detectable effect"
INCONCLUSIVE = "inconclusive"
_SIGN_EPS = 1e-9


@dataclass(frozen=True)
class Interval:
    """A point estimate and a percentile interval. `ci_low` is less than or equal to `ci_high`."""

    estimate: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class OptionalInterval:
    """An interval that may be missing because the agent did not report the metric."""

    available: bool
    estimate: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    reason: str | None = None


@dataclass(frozen=True)
class Pair:
    """One baseline trial and the treatment trial with the same index."""

    task_id: str
    trial: int
    passed_baseline: bool
    passed_treatment: bool
    tokens_baseline: int | None
    tokens_treatment: int | None
    cost_baseline: float | None
    cost_treatment: float | None
    wall_baseline: float
    wall_treatment: float
    diff_lines_baseline: int
    diff_lines_treatment: int
    turns_baseline: int | None
    turns_treatment: int | None
    tool_calls_baseline: int | None
    tool_calls_treatment: int | None
    agent_error_baseline: bool
    agent_error_treatment: bool


@dataclass(frozen=True)
class TaskSummary:
    """Per-task paired comparison."""

    task_id: str
    trials: int
    baseline_passes: int
    treatment_passes: int
    pass_rate: Interval
    tokens: OptionalInterval
    cost: OptionalInterval
    wall_time: OptionalInterval
    mean_diff_lines_baseline: float
    mean_diff_lines_treatment: float
    mean_turns_baseline: float | None
    mean_turns_treatment: float | None
    mean_tool_calls_baseline: float | None
    mean_tool_calls_treatment: float | None
    agent_errors_baseline: int
    agent_errors_treatment: int


@dataclass(frozen=True)
class Verdict:
    """The pass-rate conclusion and the numbers behind it."""

    label: str
    unadjusted_label: str
    reason: str
    outlier_driven: bool
    extreme_task_id: str | None
    extreme_task_effect: float | None
    estimate_without_extreme: float | None
    ci_without_extreme: Interval | None
    verdict_without_extreme: str | None
    suggested_trials_per_task: int | None
    trials_to_detect_observed_effect: int | None
    trials_for_margin_precision: int | None


@dataclass(frozen=True)
class Analysis:
    """Suite-level statistics computed from paired runs."""

    per_task: tuple[TaskSummary, ...]
    pass_rate: Interval
    tokens: OptionalInterval
    cost: OptionalInterval
    wall_time: OptionalInterval
    verdict: Verdict
    warnings: tuple[str, ...]
    min_trials: int
    task_count: int


def percentile(values: Sequence[float], probability: float) -> float:
    """Linear percentile (Hyndman-Fan type 7, the usual numpy default).

    ``probability`` of 0 returns the minimum and 1 returns the maximum.
    """
    if not values:
        raise SkillCheckError("Cannot take a percentile of an empty sample.")
    if probability < 0 or probability > 1:
        raise SkillCheckError("Percentile probability must be between 0 and 1.")
    ordered = sorted(values)
    if len(ordered) == 1 or probability <= 0:
        return float(ordered[0])
    if probability >= 1:
        return float(ordered[-1])
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def trials_needed(current: int, half_width: float, target_half_width: float) -> int:
    """Trials per task so a half-width would shrink to `target_half_width`.

    Assumes the standard error scales with ``1/sqrt(n)`` and the estimate does
    not move. Exact multiples stay exact. The result is never below `current`.
    """
    if current < 1:
        raise SkillCheckError("Trial count must be positive.")
    if target_half_width <= 0:
        raise SkillCheckError("Target half-width must be positive.")
    if half_width <= target_half_width or half_width == 0:
        return current
    scaled = current * (half_width / target_half_width) ** 2
    return max(current, math.ceil(scaled - 1e-12))


def pair_runs(runs: Sequence[RunRecord]) -> dict[str, list[Pair]]:
    """Group runs into per-task lists of pairs ordered by trial index."""
    grouped: dict[tuple[str, int], dict[str, RunRecord]] = {}
    for run in runs:
        if run.condition not in {"baseline", "treatment"}:
            raise SkillCheckError(f"Unknown condition {run.condition!r}.")
        slot = grouped.setdefault((run.task_id, run.trial), {})
        if run.condition in slot:
            raise SkillCheckError(f"Duplicate {run.condition} run for {run.task_id} trial {run.trial}.")
        slot[run.condition] = run
    by_task: dict[str, list[Pair]] = {}
    for task_id, trial in sorted(grouped):
        conds = grouped[(task_id, trial)]
        if "baseline" not in conds or "treatment" not in conds:
            raise SkillCheckError(f"Task {task_id} trial {trial} is missing a baseline or treatment run.")
        baseline = conds["baseline"]
        treatment = conds["treatment"]
        by_task.setdefault(task_id, []).append(
            Pair(
                task_id=task_id,
                trial=trial,
                passed_baseline=baseline.passed,
                passed_treatment=treatment.passed,
                tokens_baseline=_token_total(baseline.tokens_in, baseline.tokens_out),
                tokens_treatment=_token_total(treatment.tokens_in, treatment.tokens_out),
                cost_baseline=baseline.cost_usd,
                cost_treatment=treatment.cost_usd,
                wall_baseline=baseline.wall_time_seconds,
                wall_treatment=treatment.wall_time_seconds,
                diff_lines_baseline=baseline.diff_lines_added + baseline.diff_lines_removed,
                diff_lines_treatment=treatment.diff_lines_added + treatment.diff_lines_removed,
                turns_baseline=baseline.turns,
                turns_treatment=treatment.turns,
                tool_calls_baseline=baseline.tool_calls,
                tool_calls_treatment=treatment.tool_calls,
                agent_error_baseline=baseline.agent_error is not None,
                agent_error_treatment=treatment.agent_error is not None,
            )
        )
    if not by_task:
        raise SkillCheckError("No paired runs to analyze.")
    return by_task


def bootstrap_samples(
    by_task: dict[str, list[Pair]],
    selector: Callable[[Pair], float],
    *,
    seed: int,
    samples: int,
) -> list[float]:
    """Resample trial pairs within each task and return the suite statistic each time.

    The statistic is the unweighted mean of per-task mean differences. Task ids
    are visited in sorted order so a seed reproduces the same draws.
    """
    if samples < 1:
        raise SkillCheckError("Bootstrap sample count must be at least 1.")
    task_ids = sorted(by_task)
    if not task_ids:
        raise SkillCheckError("No tasks to bootstrap.")
    rng = random.Random(seed)
    drawn: list[float] = []
    for _ in range(samples):
        task_means: list[float] = []
        for task_id in task_ids:
            pairs = by_task[task_id]
            count = len(pairs)
            resampled = [pairs[rng.randrange(count)] for _ in range(count)]
            task_means.append(_mean([selector(pair) for pair in resampled]))
        drawn.append(_mean(task_means))
    return drawn


def bootstrap_interval(
    by_task: dict[str, list[Pair]],
    selector: Callable[[Pair], float],
    *,
    seed: int,
    samples: int,
    confidence: float,
) -> Interval:
    """Point estimate on the original pairs, interval from the bootstrap draws."""
    estimate = _suite_mean(by_task, selector)
    draws = bootstrap_samples(by_task, selector, seed=seed, samples=samples)
    return _interval_from_draws(estimate, draws, confidence)


def analyze(
    runs: Sequence[RunRecord],
    *,
    seed: int,
    confidence: float,
    bootstrap_samples: int,
    margin: float,
) -> Analysis:
    """Pair runs and compute intervals, the outlier check, and the verdict."""
    if bootstrap_samples < MIN_BOOTSTRAP_SAMPLES:
        raise SkillCheckError(f"Bootstrap samples must be at least {MIN_BOOTSTRAP_SAMPLES}. Got {bootstrap_samples}.")
    if not 0 < confidence < 1:
        raise SkillCheckError("Confidence must be between 0 and 1.")
    if not 0 < margin < 1:
        raise SkillCheckError("The negligible margin must be between 0 and 1.")
    by_task = pair_runs(runs)
    min_trials = min(len(pairs) for pairs in by_task.values())
    pass_rate = bootstrap_interval(by_task, pass_delta, seed=seed, samples=bootstrap_samples, confidence=confidence)
    tokens = _optional_interval(
        by_task,
        token_delta,
        seed=seed,
        samples=bootstrap_samples,
        confidence=confidence,
        missing_reason="Token counts were not reported for every paired trial.",
    )
    cost = _optional_interval(
        by_task,
        cost_delta,
        seed=seed,
        samples=bootstrap_samples,
        confidence=confidence,
        missing_reason="Cost was not reported for every paired trial.",
    )
    wall_time = _optional_interval(
        by_task,
        wall_delta,
        seed=seed,
        samples=bootstrap_samples,
        confidence=confidence,
        missing_reason="Wall time was missing.",
    )
    per_task = tuple(
        _summarize_task(task_id, pairs, seed=seed, samples=bootstrap_samples, confidence=confidence)
        for task_id, pairs in sorted(by_task.items())
    )
    extreme = _extreme_task(per_task)
    reduced_interval: Interval | None = None
    if len(by_task) >= 2 and extreme is not None:
        reduced_map = {key: value for key, value in by_task.items() if key != extreme.task_id}
        reduced_interval = bootstrap_interval(
            reduced_map, pass_delta, seed=seed, samples=bootstrap_samples, confidence=confidence
        )
    verdict = classify_verdict(
        trials=min_trials,
        margin=margin,
        confidence=confidence,
        full=pass_rate,
        reduced=reduced_interval,
        extreme_task_id=None if extreme is None else extreme.task_id,
        extreme_effect=None if extreme is None else extreme.pass_rate.estimate,
        task_count=len(by_task),
    )
    warnings = _warnings(
        runs=runs,
        by_task=by_task,
        pass_rate=pass_rate,
        verdict=verdict,
        bootstrap_samples=bootstrap_samples,
        min_trials=min_trials,
    )
    return Analysis(
        per_task=per_task,
        pass_rate=pass_rate,
        tokens=tokens,
        cost=cost,
        wall_time=wall_time,
        verdict=verdict,
        warnings=tuple(warnings),
        min_trials=min_trials,
        task_count=len(by_task),
    )


def classify_verdict(
    *,
    trials: int,
    margin: float,
    confidence: float,
    full: Interval,
    reduced: Interval | None,
    extreme_task_id: str | None,
    extreme_effect: float | None,
    task_count: int,
) -> Verdict:
    """Turn an interval and a leave-one-task-out interval into a verdict.

    This function does not resample. Tests use it with hand-built intervals.
    """
    if full.ci_high < full.ci_low:
        raise SkillCheckError("Confidence interval is reversed.")
    if reduced is not None and reduced.ci_high < reduced.ci_low:
        raise SkillCheckError("Leave-one-out interval is reversed.")
    unadjusted = _unadjusted_label(full, trials=trials, margin=margin)
    reduced_label = None
    if reduced is not None:
        reduced_label = _unadjusted_label(reduced, trials=trials, margin=margin)
    outlier_driven = False
    if reduced is not None:
        significance_lost = unadjusted in {HELPS, HURTS} and reduced_label not in {HELPS, HURTS}
        sign_changed = _sign(full.estimate) != _sign(reduced.estimate)
        outlier_driven = sign_changed or significance_lost
    level = format_confidence(confidence)
    half = (full.ci_high - full.ci_low) / 2
    detect: int | None = None
    margin_trials: int | None = None
    if trials >= 2 and half > 0:
        margin_trials = trials_needed(trials, half, margin)
        if abs(full.estimate) > margin:
            detect = trials_needed(trials, half, abs(full.estimate))
    blocks_trial_advice = outlier_driven and unadjusted in {HELPS, HURTS, NO_EFFECT}
    label, reason = _label_and_reason(
        unadjusted=unadjusted,
        outlier_driven=outlier_driven,
        trials=trials,
        margin=margin,
        level=level,
        full=full,
        reduced=reduced,
        extreme_task_id=extreme_task_id,
        extreme_effect=extreme_effect,
        task_count=task_count,
        detect=detect,
        margin_trials=margin_trials,
    )
    suggested: int | None = None
    if label == INCONCLUSIVE and trials >= 2 and not blocks_trial_advice:
        suggested = detect if abs(full.estimate) > margin and detect is not None else margin_trials
    return Verdict(
        label=label,
        unadjusted_label=unadjusted,
        reason=reason,
        outlier_driven=outlier_driven,
        extreme_task_id=extreme_task_id,
        extreme_task_effect=extreme_effect,
        estimate_without_extreme=None if reduced is None else reduced.estimate,
        ci_without_extreme=reduced,
        verdict_without_extreme=reduced_label,
        suggested_trials_per_task=suggested,
        trials_to_detect_observed_effect=detect,
        trials_for_margin_precision=margin_trials,
    )


def pass_delta(pair: Pair) -> float:
    """Treatment pass (1 or 0) minus baseline pass."""
    return float(pair.passed_treatment) - float(pair.passed_baseline)


def token_delta(pair: Pair) -> float | None:
    if pair.tokens_baseline is None or pair.tokens_treatment is None:
        return None
    return float(pair.tokens_treatment - pair.tokens_baseline)


def cost_delta(pair: Pair) -> float | None:
    if pair.cost_baseline is None or pair.cost_treatment is None:
        return None
    return pair.cost_treatment - pair.cost_baseline


def wall_delta(pair: Pair) -> float:
    return pair.wall_treatment - pair.wall_baseline


def _unadjusted_label(interval: Interval, *, trials: int, margin: float) -> str:
    if trials < 2:
        return INCONCLUSIVE
    if interval.ci_low > 0:
        return HELPS
    if interval.ci_high < 0:
        return HURTS
    if interval.ci_low >= -margin and interval.ci_high <= margin:
        return NO_EFFECT
    return INCONCLUSIVE


def _label_and_reason(
    *,
    unadjusted: str,
    outlier_driven: bool,
    trials: int,
    margin: float,
    level: str,
    full: Interval,
    reduced: Interval | None,
    extreme_task_id: str | None,
    extreme_effect: float | None,
    task_count: int,
    detect: int | None,
    margin_trials: int | None,
) -> tuple[str, str]:
    if trials < 2:
        return (
            INCONCLUSIVE,
            "Fewer than 2 trials were run per condition. SkillCheck will not treat "
            "a confidence interval as a verdict until every task has at least 2 paired trials. "
            "The point estimate is not a verdict.",
        )
    if outlier_driven and reduced is not None and unadjusted in {HELPS, HURTS, NO_EFFECT}:
        return (
            INCONCLUSIVE,
            _outlier_reason(level, unadjusted, full, reduced, extreme_task_id, extreme_effect),
        )
    if unadjusted == HELPS:
        return HELPS, _directional_reason("above", level, full, extreme_task_id, task_count)
    if unadjusted == HURTS:
        return HURTS, _directional_reason("below", level, full, extreme_task_id, task_count)
    if unadjusted == NO_EFFECT:
        return (
            NO_EFFECT,
            f"The {level}% CI lies inside ±{margin * 100:.1f} percentage points of zero "
            f"({format_pp(full.ci_low)} to {format_pp(full.ci_high)} pp). "
            "SkillCheck does not detect a pass-rate effect at this precision.",
        )
    return INCONCLUSIVE, _wide_reason(
        level, full, margin, detect, margin_trials, outlier_driven, extreme_task_id, reduced
    )


def _directional_reason(
    direction: str,
    level: str,
    full: Interval,
    extreme_task_id: str | None,
    task_count: int,
) -> str:
    base = (
        f"The {level}% CI for the pass-rate difference is entirely {direction} zero "
        f"({format_pp(full.ci_low)} to {format_pp(full.ci_high)} pp)"
    )
    if task_count < 2:
        return base + ". The suite has one task, so this says nothing about any other task."
    return base + f", and the direction stays the same after removing the most extreme task ({extreme_task_id})."


def _outlier_reason(
    level: str,
    unadjusted: str,
    full: Interval,
    reduced: Interval,
    extreme_task_id: str | None,
    extreme_effect: float | None,
) -> str:
    effect = "n/a" if extreme_effect is None else f"{format_pp(extreme_effect)} pp"
    return (
        f"The {level}% CI would support '{unadjusted}' "
        f"(estimate {format_pp(full.estimate)} pp, "
        f"CI {format_pp(full.ci_low)} to {format_pp(full.ci_high)} pp), "
        f"but the result changes when the most extreme task ({extreme_task_id}, {effect}) "
        f"is removed (remaining estimate {format_pp(reduced.estimate)} pp, "
        f"CI {format_pp(reduced.ci_low)} to {format_pp(reduced.ci_high)} pp). "
        "SkillCheck will not call this helps, hurts, or no detectable effect."
    )


def _wide_reason(
    level: str,
    full: Interval,
    margin: float,
    detect: int | None,
    margin_trials: int | None,
    outlier_driven: bool,
    extreme_task_id: str | None,
    reduced: Interval | None,
) -> str:
    if abs(full.estimate) <= margin:
        advice = margin_trials
        text = (
            f"The estimate is {format_pp(full.estimate)} pp, inside the ±{margin * 100:.1f} pp band, "
            f"but the {level}% CI ({format_pp(full.ci_low)} to {format_pp(full.ci_high)} pp) "
            "is wider than that band. "
        )
        if advice is not None:
            text += (
                f"If the standard error shrinks like 1/sqrt(n), about {advice} trials per task "
                "would be needed for the CI to fall inside the band. "
            )
    else:
        advice = detect
        text = (
            f"The {level}% CI ({format_pp(full.ci_low)} to {format_pp(full.ci_high)} pp) includes zero, "
            "so the evidence does not support helps or hurts. "
        )
        if advice is not None:
            text += (
                f"If the true effect equals the current estimate ({format_pp(full.estimate)} pp) "
                "and the standard error shrinks like 1/sqrt(n), about "
                f"{advice} trials per task would be needed for a {level}% CI to exclude zero. "
            )
    text += "That trial count is an approximation, not a power simulation."
    if outlier_driven and reduced is not None:
        text += f" Removing task {extreme_task_id} also changes the estimate to {format_pp(reduced.estimate)} pp."
    return text


def _optional_interval(
    by_task: dict[str, list[Pair]],
    selector: Callable[[Pair], float | None],
    *,
    seed: int,
    samples: int,
    confidence: float,
    missing_reason: str,
) -> OptionalInterval:
    for pairs in by_task.values():
        for pair in pairs:
            if selector(pair) is None:
                return OptionalInterval(available=False, reason=missing_reason)

    def required(pair: Pair) -> float:
        value = selector(pair)
        if value is None:
            raise SkillCheckError("Metric disappeared during bootstrap.")
        return value

    interval = bootstrap_interval(by_task, required, seed=seed, samples=samples, confidence=confidence)
    return OptionalInterval(
        available=True,
        estimate=interval.estimate,
        ci_low=interval.ci_low,
        ci_high=interval.ci_high,
    )


def _summarize_task(
    task_id: str,
    pairs: list[Pair],
    *,
    seed: int,
    samples: int,
    confidence: float,
) -> TaskSummary:
    alone = {task_id: pairs}
    return TaskSummary(
        task_id=task_id,
        trials=len(pairs),
        baseline_passes=sum(1 for pair in pairs if pair.passed_baseline),
        treatment_passes=sum(1 for pair in pairs if pair.passed_treatment),
        pass_rate=bootstrap_interval(alone, pass_delta, seed=seed, samples=samples, confidence=confidence),
        tokens=_optional_interval(
            alone,
            token_delta,
            seed=seed,
            samples=samples,
            confidence=confidence,
            missing_reason="Token counts were not reported for every paired trial.",
        ),
        cost=_optional_interval(
            alone,
            cost_delta,
            seed=seed,
            samples=samples,
            confidence=confidence,
            missing_reason="Cost was not reported for every paired trial.",
        ),
        wall_time=_optional_interval(
            alone,
            wall_delta,
            seed=seed,
            samples=samples,
            confidence=confidence,
            missing_reason="Wall time was missing.",
        ),
        mean_diff_lines_baseline=_mean([float(pair.diff_lines_baseline) for pair in pairs]),
        mean_diff_lines_treatment=_mean([float(pair.diff_lines_treatment) for pair in pairs]),
        mean_turns_baseline=_mean_optional([pair.turns_baseline for pair in pairs]),
        mean_turns_treatment=_mean_optional([pair.turns_treatment for pair in pairs]),
        mean_tool_calls_baseline=_mean_optional([pair.tool_calls_baseline for pair in pairs]),
        mean_tool_calls_treatment=_mean_optional([pair.tool_calls_treatment for pair in pairs]),
        agent_errors_baseline=sum(1 for pair in pairs if pair.agent_error_baseline),
        agent_errors_treatment=sum(1 for pair in pairs if pair.agent_error_treatment),
    )


def _extreme_task(tasks: tuple[TaskSummary, ...]) -> TaskSummary | None:
    if not tasks:
        return None
    largest = max(abs(task.pass_rate.estimate) for task in tasks)
    candidates = [task for task in tasks if abs(task.pass_rate.estimate) == largest]
    return min(candidates, key=lambda task: task.task_id)


def _warnings(
    *,
    runs: Sequence[RunRecord],
    by_task: dict[str, list[Pair]],
    pass_rate: Interval,
    verdict: Verdict,
    bootstrap_samples: int,
    min_trials: int,
) -> list[str]:
    warnings: list[str] = []
    if len(by_task) == 1:
        warnings.append(
            "Only one task is in this suite. The result applies to that task, not to a broader set of tasks."
        )
    if min_trials >= 2 and pass_rate.ci_low == pass_rate.ci_high and abs(pass_rate.estimate) > _SIGN_EPS:
        warnings.append(
            "Every paired trial agreed, so the pass-rate interval has zero width. "
            "That is expected when the agent is deterministic. For a stochastic agent, "
            "a zero-width interval at a small trial count can happen by chance; add trials "
            "before treating the verdict as stable."
        )
    if bootstrap_samples < 1000:
        warnings.append(
            f"Only {bootstrap_samples} bootstrap samples were drawn. "
            "Intervals are coarse below 1000 samples; the default is 4000."
        )
    token_runs = sum(1 for run in runs if run.tokens_in is not None and run.tokens_out is not None)
    if token_runs not in {0, len(runs)}:
        warnings.append(
            f"Token totals are partial ({token_runs}/{len(runs)} runs reported both input and "
            "output tokens). The token interval was not computed."
        )
    cost_runs = sum(1 for run in runs if run.cost_usd is not None)
    if cost_runs not in {0, len(runs)}:
        warnings.append(
            f"Cost is partial ({cost_runs}/{len(runs)} runs reported cost). The cost interval was not computed."
        )
    error_runs = sum(1 for run in runs if run.agent_error)
    if error_runs:
        warnings.append(
            f"{error_runs} run(s) recorded an agent error. Verification still ran on the "
            "workspace the agent left behind, and a failed verification counts as a failed trial."
        )
    if verdict.outlier_driven and verdict.extreme_task_id:
        warnings.append(
            f"Outlier check: removing task {verdict.extreme_task_id} changes the pass-rate "
            "conclusion. The headline verdict stays inconclusive."
        )
    return warnings


def _suite_mean(by_task: dict[str, list[Pair]], selector: Callable[[Pair], float]) -> float:
    means = [_mean([selector(pair) for pair in by_task[task_id]]) for task_id in sorted(by_task)]
    return _mean(means)


def _interval_from_draws(estimate: float, draws: Sequence[float], confidence: float) -> Interval:
    alpha = (1 - confidence) / 2
    low = percentile(draws, alpha)
    high = percentile(draws, 1 - alpha)
    if high < low:
        low, high = high, low
    return Interval(estimate=estimate, ci_low=low, ci_high=high)


def _token_total(tokens_in: int | None, tokens_out: int | None) -> int | None:
    if tokens_in is None or tokens_out is None:
        return None
    return tokens_in + tokens_out


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise SkillCheckError("Cannot average an empty list.")
    return float(sum(values) / len(values))


def _mean_optional(values: Sequence[int | None]) -> float | None:
    if any(value is None for value in values):
        return None
    numbers = [float(value) for value in values if value is not None]
    return _mean(numbers)


def _sign(value: float) -> int:
    if value > _SIGN_EPS:
        return 1
    if value < -_SIGN_EPS:
        return -1
    return 0
