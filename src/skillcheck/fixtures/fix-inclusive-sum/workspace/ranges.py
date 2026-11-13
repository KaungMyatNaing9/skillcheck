def sum_inclusive(start: int, end: int) -> int:
    """Sum the integers from start through end, inclusive."""
    total = 0
    for i in range(start, end):
        total += i
    return total
