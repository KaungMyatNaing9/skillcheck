# SkillCheck

SkillCheck measures whether a skill or instruction file changes a coding agent's results on a task suite you choose. It runs the same tasks with the file absent (baseline) and with the file installed (treatment), then reports the pass-rate difference with a bootstrap confidence interval.

A verdict of **helps**, **hurts**, or **no detectable effect** is used only when that interval supports it. Otherwise the result is **inconclusive**. One extreme task cannot carry a directional claim.

Author: Kaung Myat Naing ([KaungMyatNaing9](https://github.com/KaungMyatNaing9)). License: MIT.

This repository is a local evaluation tool. It does not publish results, call external services, or contact anyone on its own.

## Install

Requires Python 3.10 or newer. The tests in this checkout were run on Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick start

No API key and no third-party agent CLI are required. The `fake` adapter applies edits only when a skill contains a fenced `skillcheck-fake` JSON block. It is a deterministic stand-in for the pipeline, not a model.

```bash
skillcheck tasks list
skillcheck run \
  --tasks builtin \
  --skill examples/skills/suite-fix \
  --agent fake \
  --trials 4
skillcheck report skillcheck-results/results.json
```

`examples/skills/null-style` is a negative control: prose only, no scripted edits. `examples/skills/sum-only` fixes one of the three built-in tasks so the outlier check can fire.

## What a verdict means

The headline number is the pass rate of the treatment minus the pass rate of the baseline, in percentage points. Each task is weighted equally. Trials are paired by index.

| Verdict | When SkillCheck uses it |
| --- | --- |
| `helps` | The pass-rate interval is entirely above zero, and dropping the most extreme task does not remove that direction. |
| `hurts` | The interval is entirely below zero, and the outlier check agrees. |
| `no detectable effect` | The interval lies inside the negligible band (default ±5 percentage points) and the tasks are not cancelling each other out. |
| `inconclusive` | Anything else, including fewer than 2 trials, a wide interval, or an effect that disappears when one task is removed. |

Cost, tokens, and time are reported beside the verdict. They do not change it. When the result is inconclusive because the interval is wide, the report includes an approximate trial count based on the standard error shrinking like `1/sqrt(n)`. That number is not a power simulation.

The interval is a percentile bootstrap that resamples trials inside each task and keeps the task list fixed. It describes stability on these tasks. It does not, by itself, say what would happen on a new task. The leave-one-task-out check is the guard against a single task driving the suite.

## Tasks

A suite is YAML or JSON with a `tasks` list. Paths are relative to the suite file. `skillcheck run --tasks builtin` uses the packaged suite. A directory is accepted when it contains `suite.yaml`, `suite.yml`, or `suite.json`.

```yaml
name: my-suite
tasks:
  - id: fix-add
    workspace: workspace          # directory copied into the sandbox
    hidden: hidden                # copied in only after the agent finishes
    # or: repo: /path/to/repo
    #     rev: HEAD               # git archive; uncommitted files are not included
    #     hide: ["tests/private"] # removed before the agent runs, restored before verify
    prompt: |
      Fix add so it returns the sum of its arguments.
    verify: "{python} -m unittest discover -s tests -q"
    timeout_seconds: 60
```

`{python}` is replaced with the interpreter running SkillCheck. The verify command is trusted shell, executed in the sandbox. See `examples/minimal-task/` for a complete copy.

Built-in tasks are three small Python bugs (`fix-inclusive-sum`, `fix-median`, `fix-slugify`). Each starts broken. Hidden tests are not in the agent workspace. They are not a security boundary: a command can still read the host filesystem. The files are only kept out of the workspace tree.

`workspace` copies files and skips `.git` and tool caches. `repo` plus `rev` uses `git archive`.

## Skills

`--skill` may be repeated. Every file is installed only in the treatment sandbox. Both conditions have those destination paths removed first, so the comparison is absence versus the files you passed.

| Input | Installed at |
| --- | --- |
| `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `.cursorrules` | that filename at the workspace root |
| `SKILL.md` | `.claude/skills/<parent-directory>/SKILL.md` |
| a directory containing `SKILL.md` | `.claude/skills/<directory-name>/` |
| `*.mdc` | `.cursor/rules/<name>` |
| `.cursor/rules/**` inside a directory | the same relative path |
| `skillcheck.yaml` / `.yml` / `.json` | the manifest's `install: [{source, dest}, ...]` list |
| `PATH::relative/dest` | the explicit destination |

Destinations cannot be absolute or contain `..`.

The fake adapter reads `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `.cursorrules`, `.cursor/rules/**`, and `.claude/skills/**/SKILL.md` inside the workspace. A fence like this is applied only by that adapter:

````markdown
```skillcheck-fake
{"tasks": {"fix-add": {"actions": [
  {"op": "replace", "file": "app.py", "find": "return left - right", "replace": "return left + right"}
]}}}
```
````

`tasks` selects actions by task id. A top-level `actions` list applies to every task. `write` and `replace` stay inside the sandbox. The fake adapter refuses `run`, so a skill file cannot start a shell. The replay adapter can, because you supplied the transcript.

## Agents

| `--agent` | What it does | Tested here |
| --- | --- | --- |
| `fake` | Applies `skillcheck-fake` blocks. No network. | Yes, including end-to-end studies. |
| `replay` | Plays a JSON transcript (`--agent-opt script=PATH`). | Yes. |
| `command` | Runs a shell template. | Yes, with a local Python process. Not with Claude, Codex, or Aider. |
| `openai` | Tool loop (list, read, write, edit, run, done) against an OpenAI-compatible chat completions API. | Yes, against an in-process fake client and a local HTTP stub. **No live provider has been called. There is no API key in this environment.** |

Command placeholders, each shell-quoted: `{prompt}`, `{prompt_file}`, `{workspace}`. The prompt is also written to `.skillcheck/prompt.txt`. If the process prints a line

```text
SKILLCHECK_METRICS {"tokens_in": 1, "tokens_out": 2, "turns": 1, "tool_calls": 1, "cost_usd": 0.01}
```

those fields are recorded. Otherwise tokens and cost stay unreported. They are not guessed.

OpenAI-compatible settings:

| Purpose | Preferred | Fallback |
| --- | --- | --- |
| API key | `SKILLCHECK_API_KEY` | `OPENAI_API_KEY` |
| Model | `SKILLCHECK_MODEL` | `OPENAI_MODEL` (required; there is no default model) |
| Base URL | `SKILLCHECK_BASE_URL` | `OPENAI_BASE_URL`, then `https://api.openai.com/v1` |
| USD per million input tokens | `SKILLCHECK_INPUT_USD_PER_MILLION` | |
| USD per million output tokens | `SKILLCHECK_OUTPUT_USD_PER_MILLION` | |

Cost is filled in only when both prices are set and the response includes `usage`. Usage is summed across responses. A provider that reports cumulative usage would be over-counted. That has not been checked against a live API.

Do not put secrets in `--agent-opt`. Options other than `api_key` are copied into `results.json`.

## CLI

```text
skillcheck run --skill PATH [--skill PATH::DEST] --tasks SUITE|builtin \
  --agent fake|replay|command|openai --trials N \
  [--seed N] [--bootstrap N] [--confidence 0.95] [--margin 0.05] \
  [--timeout SEC] [--agent-opt key=value] [--out DIR] [--verbose] [--keep]

skillcheck report results.json [--badge badge.json] [--format md|json]
skillcheck tasks list [--tasks SUITE|builtin]
```

`run` writes `results.json`, `report.md`, and `badge.json`. Exit code 0 means the evaluation finished, including when the verdict is `hurts` or `inconclusive`. `report` recomputes the verdict and the badge from the run records. A hand-edited summary in the JSON does not survive that command.

`--trials` may be 1, for a smoke test. The verdict stays inconclusive until every task has at least 2 paired trials. `--bootstrap` must be at least 200. The default is 4000.

## Reports and badges

The badge is shields.io endpoint JSON (`schemaVersion`, `label`, `message`, `color`). It is built from the pass-rate interval of the run that produced it. `helps` and `hurts` messages include the estimate and the interval. `no detectable effect` and `inconclusive` lead with that word, not with a pass-rate claim. Colors are brightgreen, red, blue, and yellow. "pp" means percentage points of pass rate (treatment minus baseline), not a relative percent change. The badges in the next section are from the fake adapter, not from a live model.

## Example output

The following is the real stdout of the fake adapter on the built-in suite in this environment. It is not a result from a live coding agent. The `suite-fix` skill contains the scripted edits that adapter applies, so a pass-rate of +100 pp means those edits made the hidden tests pass. The zero-width warning is expected: every trial agreed.

```text
warning: Every paired trial agreed, so the pass-rate interval has zero width. That is expected when the agent is deterministic. For a stochastic agent, a zero-width interval at a small trial count can happen by chance; add trials before treating the verdict as stable.
Verdict: helps
The 95% CI for the pass-rate difference is entirely above zero (+100.0 to +100.0 pp), and the direction stays the same after removing the most extreme task (fix-inclusive-sum).
Pass rate: +100.0 pp [+100.0, +100.0]
```

Badge written by that run:

```json
{"schemaVersion": 1, "label": "SkillCheck", "message": "+100.0 pp pass rate [95% CI +100.0, +100.0]", "color": "brightgreen"}
```

The same command with `examples/skills/null-style`:

```text
Verdict: no detectable effect
The 95% CI lies inside ±5.0 percentage points of zero (+0.0 to +0.0 pp). SkillCheck does not detect a pass-rate effect at this precision.
Pass rate: +0.0 pp [+0.0, +0.0]
```

The same command with `examples/skills/sum-only` (only the inclusive-sum task is fixed):

```text
Verdict: inconclusive
The 95% CI would support 'helps' (estimate +33.3 pp, CI +33.3 to +33.3 pp), but the result changes when the most extreme task (fix-inclusive-sum, +100.0 pp) is removed (remaining estimate +0.0 pp, CI +0.0 to +0.0 pp). SkillCheck will not call this helps, hurts, or no detectable effect.
Pass rate: +33.3 pp [+33.3, +33.3]
```

On that same fake-adapter run, token totals use a 4-characters-per-token heuristic, not a model tokenizer. The fix skill's report showed a token difference of +342.3 because the skill text and the edits are counted. Cost was `n/a` because the fake adapter does not invent a price. The null skill does not change pass rate, but it still increases the heuristic token count, because its text is part of the treatment prompt. That does not change the verdict.

## What is tested, and what is not

Tested without a network and without an API key:

- YAML and JSON task loading, including the built-in suite and `examples/minimal-task`
- sandbox copies, skill install paths, hidden-file restore, and `git archive` at a revision
- fake and replay adapters, including refusal to run shell from a skill and refusal to write outside the workspace
- the command adapter, including the metrics trailer, timeout, and a study where the command edits only when `AGENTS.md` is present
- the OpenAI-compatible tool loop against a scripted client and a local HTTP stub (auth header, URL, usage, HTTP 401, path jail)
- diff sizes, bootstrap intervals on synthetic data (including an independent reimplementation of the resample), verdict rules, and the outlier check
- end-to-end studies: scripted skill → `helps`; null skill → `no detectable effect`; one-task skill → `inconclusive` and outlier-driven
- `skillcheck report` recomputes a tampered badge from the run records

Not tested:

- any live model API
- `claude`, `codex`, `aider`, or any other third-party agent CLI
- that hidden tests stay secret from an agent that can read outside the workspace
- Windows

A live run needs `SKILLCHECK_API_KEY` (or `OPENAI_API_KEY`) and `SKILLCHECK_MODEL` (or `OPENAI_MODEL`). Until that has been executed, treat the OpenAI adapter as implemented and untested against a real provider.

## Development

```bash
ruff check .
ruff format --check .
pytest
```

## License

MIT. See [LICENSE](LICENSE).
