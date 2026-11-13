"""Bootstrap intervals, verdicts, and the outlier check."""

from __future__ import annotations

import random

import pytest
from tests.conftest import make_run, runs_for

from skillcheck.errors import SkillCheckError
from skillcheck.stats import (
    HELPS,
    HURTS,
    INCONCLUSIVE,
    NO_EFFECT,
    Interval,
    analyze,
    bootstrap_samples,
    classify_verdict,
    pair_runs,
    pass_delta,
    percentile,
    trials_needed,
)


def test_percentile_matches_type7_examples() -> None:
    assert percentile([0, 1, 2, 3, 4], 0) == 0
    assert percentile([0, 1, 2, 3, 4], 1) == 4
    assert percentile([0, 1, 2, 3, 4], 0.5) == 2
    assert percentile([0, 10], 0.5) == 5
    assert percentile([0, 10, 20], 0.25) == 5


def test_trials_needed_scales_with_one_over_sqrt_n() -> None:
    assert trials_needed(10, 0.2, 0.1) == 40
    assert trials_needed(10, 0.2, 0.05) == 160
    assert trials_needed(10, 0.02, 0.05) == 10
    assert trials_needed(10, 0.0, 0.05) == 10


def test_bootstrap_matches_an_independent_resampling() -> None:
    by_task = pair_runs(runs_for({"a": [1, 0], "b": [0, 0]}))
    actual = bootstrap_samples(by_task, pass_delta, seed=0, samples=500)
    rng = random.Random(0)
    deltas = {"a": [1.0, 0.0], "b": [0.0, 0.0]}
    expected: list[float] = []
    for _ in range(500):
        means = []
        for task_id in ("a", "b"):
            drawn = [deltas[task_id][rng.randrange(2)] for _ in range(2)]
            means.append(sum(drawn) / 2)
        expected.append(sum(means) / 2)
    assert actual == expected


def test_constant_effect_has_a_zero_width_interval() -> None:
    analysis = analyze(
        runs_for({"a": [1, 1], "b": [1, 1]}),
        seed=0,
        confidence=0.95,
        bootstrap_samples=200,
        margin=0.05,
    )
    assert analysis.pass_rate.estimate == 1
    assert analysis.pass_rate.ci_low == 1
    assert analysis.pass_rate.ci_high == 1
    assert analysis.verdict.label == HELPS
    assert analysis.verdict.outlier_driven is False


def test_tasks_are_weighted_equally_when_trial_counts_differ() -> None:
    analysis = analyze(
        runs_for({"a": [1], "b": [0, 0, 0, 0]}),
        seed=0,
        confidence=0.95,
        bootstrap_samples=200,
        margin=0.05,
    )
    assert analysis.pass_rate.estimate == 0.5
    assert analysis.verdict.label == INCONCLUSIVE
    assert "2 trials" in analysis.verdict.reason or "Fewer than 2" in analysis.verdict.reason


def test_outlier_task_blocks_a_directional_verdict() -> None:
    analysis = analyze(
        runs_for({"a": [1, 1], "b": [0, 0], "c": [0, 0]}),
        seed=0,
        confidence=0.95,
        bootstrap_samples=200,
        margin=0.05,
    )
    assert analysis.pass_rate.estimate == pytest.approx(1 / 3)
    assert analysis.verdict.unadjusted_label == HELPS
    assert analysis.verdict.outlier_driven is True
    assert analysis.verdict.extreme_task_id == "a"
    assert analysis.verdict.estimate_without_extreme == 0
    assert analysis.verdict.label == INCONCLUSIVE


def test_uniform_positive_effect_is_not_an_outlier() -> None:
    analysis = analyze(
        runs_for({"a": [1, 1, 1, 1, 1], "b": [1, 1, 1, 1, 1], "c": [1, 0, 0, 0, 0]}),
        seed=0,
        confidence=0.95,
        bootstrap_samples=200,
        margin=0.05,
    )
    assert analysis.pass_rate.estimate == pytest.approx((1 + 1 + 0.2) / 3)
    assert analysis.verdict.outlier_driven is False
    assert analysis.verdict.label == HELPS


def test_null_effect_is_no_detectable_effect() -> None:
    analysis = analyze(
        runs_for({"a": [0, 0], "b": [0, 0]}),
        seed=0,
        confidence=0.95,
        bootstrap_samples=200,
        margin=0.05,
    )
    assert analysis.pass_rate.estimate == 0
    assert (analysis.pass_rate.ci_low, analysis.pass_rate.ci_high) == (0, 0)
    assert analysis.verdict.label == NO_EFFECT
    assert analysis.verdict.outlier_driven is False


def test_uniform_negative_effect_hurts() -> None:
    analysis = analyze(
        runs_for({"a": [-1, -1], "b": [-1, -1]}),
        seed=0,
        confidence=0.95,
        bootstrap_samples=200,
        margin=0.05,
    )
    assert analysis.verdict.label == HURTS
    assert analysis.pass_rate.estimate == -1


def test_cancellation_is_not_called_no_effect() -> None:
    analysis = analyze(
        runs_for({"a": [1, 1], "b": [-1, -1]}),
        seed=0,
        confidence=0.95,
        bootstrap_samples=200,
        margin=0.05,
    )
    assert analysis.pass_rate.estimate == 0
    assert analysis.verdict.unadjusted_label == NO_EFFECT
    assert analysis.verdict.outlier_driven is True
    assert analysis.verdict.label == INCONCLUSIVE


def test_wide_interval_suggests_more_trials() -> None:
    verdict = classify_verdict(
        trials=5,
        margin=0.05,
        confidence=0.95,
        full=Interval(0.2, -0.05, 0.45),
        reduced=Interval(0.2, -0.05, 0.45),
        extreme_task_id="a",
        extreme_effect=0.2,
        task_count=2,
    )
    assert verdict.label == INCONCLUSIVE
    assert verdict.outlier_driven is False
    # half-width is 0.25, target is the 0.2 estimate: ceil(5 * (0.25/0.2)^2) = 8
    assert verdict.suggested_trials_per_task == 8
    assert "approximation" in verdict.reason


def test_near_zero_but_wide_interval_targets_the_margin() -> None:
    verdict = classify_verdict(
        trials=10,
        margin=0.05,
        confidence=0.95,
        full=Interval(0.0, -0.2, 0.2),
        reduced=Interval(0.0, -0.2, 0.2),
        extreme_task_id="a",
        extreme_effect=0.0,
        task_count=2,
    )
    assert verdict.label == INCONCLUSIVE
    assert verdict.suggested_trials_per_task == 160


def test_analyze_is_reproducible_for_a_noisy_sample() -> None:
    runs = runs_for({"a": [1, -1, 1, -1, 0, 1], "b": [0, 1, -1, 0, 1, -1]})
    kwargs = {"seed": 7, "confidence": 0.9, "bootstrap_samples": 300, "margin": 0.05}
    first = analyze(runs, **kwargs)
    second = analyze(runs, **kwargs)
    assert first.pass_rate == second.pass_rate
    assert first.verdict.label == second.verdict.label


def test_duplicate_and_unpaired_runs_error() -> None:
    runs = runs_for({"a": [1, 0]})
    with pytest.raises(SkillCheckError, match="Duplicate"):
        analyze(
            [*runs, make_run(task_id="a", condition="baseline", trial=1)],
            seed=0,
            confidence=0.95,
            bootstrap_samples=200,
            margin=0.05,
        )
    with pytest.raises(SkillCheckError, match="missing"):
        analyze(
            [make_run(task_id="a", condition="baseline", trial=1)],
            seed=0,
            confidence=0.95,
            bootstrap_samples=200,
            margin=0.05,
        )


def test_missing_tokens_and_partial_cost() -> None:
    runs = runs_for({"a": [1, 1], "b": [1, 1]}, tokens=None, cost=(1.0, 3.0))
    analysis = analyze(runs, seed=0, confidence=0.95, bootstrap_samples=200, margin=0.05)
    assert analysis.tokens.available is False
    assert analysis.cost.available is True
    assert analysis.cost.estimate == 2
    assert analysis.cost.ci_low == 2
    assert analysis.cost.ci_high == 2
    partial = runs_for({"a": [0, 0]})
    partial[0] = make_run(
        task_id="a",
        condition="baseline",
        trial=1,
        cost_usd=1.0,
        tokens_in=1,
        tokens_out=1,
    )
    # The treatment pair for trial 1 has no cost, so the metric is unavailable
    # and the partial coverage warning fires.
    partial_analysis = analyze(partial, seed=0, confidence=0.95, bootstrap_samples=200, margin=0.05)
    assert partial_analysis.cost.available is False
    assert any("partial" in warning.lower() for warning in partial_analysis.warnings)


def test_bootstrap_sample_floor() -> None:
    with pytest.raises(SkillCheckError, match="200"):
        analyze(
            runs_for({"a": [0, 0]}),
            seed=0,
            confidence=0.95,
            bootstrap_samples=10,
            margin=0.05,
        )


def test_zero_width_and_single_task_warnings() -> None:
    analysis = analyze(
        runs_for({"only": [1, 1]}),
        seed=0,
        confidence=0.95,
        bootstrap_samples=200,
        margin=0.05,
    )
    text = " ".join(analysis.warnings)
    assert "one task" in text
    assert "zero width" in text
    assert analysis.verdict.label == HELPS
