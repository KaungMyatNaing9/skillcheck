"""Reports and badges are computed from run records."""

from __future__ import annotations

from tests.conftest import runs_for

from skillcheck.formatting import format_pp
from skillcheck.report import build_badge, finalize_document, render_markdown
from skillcheck.stats import HELPS, INCONCLUSIVE, NO_EFFECT, Interval


def _document(runs: list[object], **overrides: object) -> dict[str, object]:
    document: dict[str, object] = {
        "schema_version": 1,
        "skillcheck_version": "0.1.0",
        "created_at": "2026-01-01T00:00:00+00:00",
        "seed": 0,
        "confidence": 0.95,
        "bootstrap_samples": 200,
        "negligible_margin": 0.05,
        "trials": 2,
        "suite": {
            "name": "synthetic",
            "description": "Synthetic runs for the report test.",
            "source": "synthetic",
            "path": "synthetic",
            "tasks": [],
        },
        "agent": {"name": "fake", "options": {}},
        "skills": [],
        "runs": [run.__dict__ if hasattr(run, "__dict__") else run for run in runs],
        "eval_wall_time_seconds": 1.25,
    }
    document.update(overrides)
    finalized = finalize_document(document)
    return finalized


def test_helps_badge_uses_the_interval() -> None:
    badge = build_badge(HELPS, Interval(0.12, 0.04, 0.20), 0.95)
    assert badge["schemaVersion"] == 1
    assert badge["label"] == "SkillCheck"
    assert badge["color"] == "brightgreen"
    assert badge["message"] == f"{format_pp(0.12)} pp pass rate [95% CI {format_pp(0.04)}, {format_pp(0.20)}]"
    assert badge["message"] == "+12.0 pp pass rate [95% CI +4.0, +20.0]"


def test_inconclusive_badge_does_not_lead_with_a_pass_rate_claim() -> None:
    badge = build_badge(INCONCLUSIVE, Interval(0.33, 0.33, 0.33), 0.95)
    assert str(badge["message"]).startswith("inconclusive")
    assert "pp pass rate" not in str(badge["message"])
    assert badge["color"] == "yellow"
    none = build_badge(NO_EFFECT, Interval(0.0, -0.02, 0.02), 0.8)
    assert str(none["message"]).startswith("no detectable effect")
    assert "80% CI" in str(none["message"])
    assert none["color"] == "blue"


def test_markdown_contains_the_task_and_disclaimer() -> None:
    from tests.conftest import runs_for as build

    document = _document(build({"alpha": [1, 1], "beta": [1, 1]}))
    text = render_markdown(document)
    assert "Verdict: helps" in text
    assert "alpha" in text
    assert "beta" in text
    assert "not a general claim" in text
    assert document["badge"]["color"] == "brightgreen"
    assert document["overall"]["pass_rate_difference"]["estimate"] == 1


def test_tampered_badge_is_recomputed_from_runs() -> None:
    document = _document(runs_for({"alpha": [0, 0], "beta": [0, 0]}))
    assert document["verdict"]["label"] == "no detectable effect"
    tampered = dict(document)
    tampered["verdict"] = {"label": "helps", "reason": "invented"}
    tampered["badge"] = {
        "schemaVersion": 1,
        "label": "SkillCheck",
        "message": "+12% pass rate [95% CI +4, +20]",
        "color": "brightgreen",
    }
    tampered["overall"] = {"pass_rate_difference": {"estimate": 0.12, "ci_low": 0.04, "ci_high": 0.20}}
    recomputed = finalize_document(tampered)
    assert recomputed["verdict"]["label"] == "no detectable effect"
    assert "+12%" not in str(recomputed["badge"]["message"])
    assert recomputed["badge"]["color"] == "blue"
    assert any("recomputed" in warning for warning in recomputed["warnings"])
    text = render_markdown(recomputed)
    assert "no detectable effect" in text
    assert "+12%" not in text


def test_cost_sentence_does_not_change_the_verdict() -> None:
    document = _document(
        runs_for({"alpha": [1, 1], "beta": [1, 1]}, cost=(1.0, 2.0)),
    )
    text = render_markdown(document)
    assert "Verdict: helps" in text
    assert "cost more" in text
    assert "does not change the pass-rate verdict" in text
    assert document["overall"]["cost_difference"]["estimate"] == 1
