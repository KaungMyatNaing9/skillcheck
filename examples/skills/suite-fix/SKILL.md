# Suite fix guide

This file is a fixture for SkillCheck's fake adapter. The fenced JSON is how that
deterministic adapter applies edits during tests and dry runs. A real model is not
expected to follow it, and these edits are not a measured result for any live agent.

```skillcheck-fake
{
  "tasks": {
    "fix-inclusive-sum": {
      "actions": [
        {
          "op": "replace",
          "file": "ranges.py",
          "find": "for i in range(start, end):",
          "replace": "for i in range(start, end + 1):"
        }
      ]
    },
    "fix-median": {
      "actions": [
        {
          "op": "replace",
          "file": "stats_util.py",
          "find": "    return float(ordered[mid - 1])",
          "replace": "    return (ordered[mid - 1] + ordered[mid]) / 2"
        }
      ]
    },
    "fix-slugify": {
      "actions": [
        {
          "op": "replace",
          "file": "slug.py",
          "find": "def slugify(text: str) -> str:",
          "replace": "import re\n\n\ndef slugify(text: str) -> str:"
        },
        {
          "op": "replace",
          "file": "slug.py",
          "find": "    return text.replace(\" \", \"-\")",
          "replace": "    slug = text.strip().lower()\n    slug = re.sub(r\"[^a-z0-9]+\", \"-\", slug)\n    return slug.strip(\"-\")"
        }
      ]
    }
  }
}
```
