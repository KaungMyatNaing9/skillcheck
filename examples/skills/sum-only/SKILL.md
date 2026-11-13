# Sum-only fix

Fake-adapter fixture that fixes only the inclusive-sum task. On the built-in suite
the other two tasks stay broken, so a directional verdict would be driven by one task.
SkillCheck should call that inconclusive.

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
    }
  }
}
```
