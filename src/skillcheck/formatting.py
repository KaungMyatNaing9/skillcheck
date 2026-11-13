"""Shared number formatting for reports and verdict text."""

from __future__ import annotations


def format_confidence(level: float) -> str:
    """Format 0.95 as '95' and 0.975 as '97.5'."""
    percent = level * 100
    if abs(percent - round(percent)) < 1e-9:
        return str(round(percent))
    return f"{percent:.1f}"


def format_signed(value: float, digits: int) -> str:
    """Format a number with an explicit sign. Tiny residuals become +0."""
    rounded = round(float(value), digits)
    if abs(rounded) < 0.5 * 10 ** (-digits):
        rounded = 0.0
    return f"{rounded:+.{digits}f}"


def format_pp(fraction: float) -> str:
    """Format a pass-rate difference as signed percentage points, one decimal."""
    return format_signed(fraction * 100, 1)
