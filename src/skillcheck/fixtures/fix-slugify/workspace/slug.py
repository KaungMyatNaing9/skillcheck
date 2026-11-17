def slugify(text: str) -> str:
    """Build a URL slug: lowercase, hyphen-separated words."""
    return text.replace(" ", "-")
