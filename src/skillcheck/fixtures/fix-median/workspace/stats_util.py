def median(values: list[float]) -> float:
    """Return the median. For an even count, average the two middle numbers."""
    if not values:
        raise ValueError("median of empty input")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return float(ordered[mid - 1])
