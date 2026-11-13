"""User-facing errors."""


class SkillCheckError(Exception):
    """A problem the caller can fix. The CLI prints this without a traceback."""
