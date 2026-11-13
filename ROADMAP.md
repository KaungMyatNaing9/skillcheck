# Roadmap

Milestone 1 is the local A/B harness in this repository. Later milestones are not started. None of them should ship a score that drops the confidence interval, the trial count, or the outlier check.

## Milestone 1 — Local A/B harness (this release)

- CLI: `run`, `report`, `tasks list`
- Task suites in YAML or JSON, plus three built-in fixture tasks with hidden tests
- Isolated sandboxes; skill files installed only on the treatment side
- Adapters: `fake`, `replay`, `command`, and an OpenAI-compatible tool loop
- Pass/fail, tokens, cost when reported, wall time, turns, tool calls, diff size
- Paired trials, equal task weights, percentile bootstrap, leave-one-task-out outlier check
- Verdicts: `helps`, `hurts`, `no detectable effect`, `inconclusive`
- Markdown, JSON, and a shields.io badge computed from the run records

Live model APIs and third-party agent CLIs are not part of what milestone 1 verified.

## Milestone 2 — Live agents on real repositories

Run the adapters that milestone 1 only stubbed.

- One recorded transcript against a real OpenAI-compatible endpoint, checked into the test suite as a fixture after secrets are stripped
- Command-adapter templates for Claude, Codex, and Aider that have actually been executed, with the flags that were used written down only after that run
- A cost figure compared with a provider invoice for one model, instead of the optional USD-per-million inputs alone
- Notes on timeout, tool-call drift, and how API keys are kept out of `results.json`

Until that exists, README examples that mention those CLIs stay templates, not claims.

## Milestone 3 — Stronger inference

The milestone 1 interval treats the task list as fixed. That is the right question for "these tasks," and it is the wrong question for "tasks like these."

- A task-cluster bootstrap, reported separately, for anyone who wants to generalize past the suite
- A pre-registered trial count, and a sequential rule that does not peek and then stop on a favorable interval
- A correction when several skills are compared in one family of runs, so a leaderboard cannot crown noise
- A power simulation to replace the `1/sqrt(n)` trial suggestion
- A cross-check of the percentile interval against a permutation test or a BCa interval on synthetic data

## Milestone 4 — GitHub Action

A workflow that runs SkillCheck when a pull request changes a skill, and posts the markdown report.

- Cache the task checkout
- Fail the check only when the user opts in and the verdict is `hurts`
- Never fail on `inconclusive`
- Do not upload a badge that the report command would refuse to regenerate

## Milestone 5 — Verified badges

The badge file is already derived from run records. A later badge that other people are asked to trust needs a provenance stamp in the linked report:

- SkillCheck version, suite hash, agent name, model id, trial count, seed
- The interval, not only the point estimate
- No badge when the verdict is `inconclusive` or outlier-driven, if the badge is meant to be cited as a result

## Milestone 6 — Opt-in public leaderboard

Only if someone chooses to publish a results file.

- Publish the raw results, not a single score
- Show the interval next to every number
- Rank nothing whose interval includes zero or whose outlier check fired
- Require the provenance stamp from milestone 5
- No "top skill" headline

## Milestone 7 — Harder isolation

Milestone 1 keeps hidden tests out of the workspace directory. It does not stop a shell from reading the host.

- A container or bubblewrap sandbox with the hidden tests outside the mount
- Network off during a trial unless the task opts in
- Parallel trials and a resume log so a long study can continue after a crash
