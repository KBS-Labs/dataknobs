# Lint Policy — the config decides what is an error

## The Rule

The repository's ruff and mypy configuration is the sole authority on what
is a lint or type error. Before reporting or "fixing" a finding, ask:

```bash
bin/quality-contract.py explain <CODE> [<path>]
```

Do not report a finding from a rule the config declines. If a decline looks
wrong, that is a config change with a written argument — not a finding.

## Why the question needs a command

A finding is suppressed through four independent channels, and which one
applies is not visible from the finding:

| Channel | Where it is stated |
|---|---|
| the rule is not selected | `[tool.ruff.lint] select` |
| the rule is declined repo-wide | `[tool.ruff.lint] ignore`, with a `[category]` |
| the file is waived | `[tool.ruff.lint.per-file-ignores]` |
| the line carries a directive | `# noqa: <CODE>` / `# type: ignore[<code>]` |

Answering by reading is a five-hundred-line TOML file plus knowing which
mechanism applies. `explain` answers in one line, and it asks **ruff** for
the enabled set rather than deriving it: `select` lists the legacy `TCH`
selector while the rules it enables are spelled `TC00x`, so prefix-matching
the declared families reports enforced rules as unselected. Only the
*reason* is parsed out of the config, because reasons are comments and no
tool can read them.

`explain` is read-only and exits 0 on every verdict, including the ones
meaning the finding is real. `check` owns the pass/fail role; a second
command that can also fail is how a gate ends up with two verdicts able to
disagree.

## Never narrow `--select` when fixing

A narrowed run is for **counting one rule's exposure** and nothing else.

Parity fails in both directions, and the second is worse. A worker running
narrower than the gate misses real findings **and can delete live
suppressions**: `ruff check --select X --fix` reads a `# noqa` naming a rule
outside the narrowing as *unused* and removes it. Measured, not
hypothetical — a `--select RUF100 --fix` run in this repository removed five
directives where four were predicted, the fifth being a live `# noqa: B027`.

Fixes go through `bin/validate.sh -f`, which runs the whole configured rule
set with `--no-unsafe-fixes`. Never a narrowed `--fix`.

The same applies to measurement runs: every measuring command must be
read-only. A `--fix-only` inside a measurement loop once silently mutated
225 files here, detected only because a re-measurement returned lower
numbers. If a count moves between two runs, check `git status` first and
conclude second.

### `RUF100` cannot be counted by a narrowed run at all

The opening sentence of this section has one exception, and it is the rule
most likely to be measured alone.

`RUF100` reports a `# noqa` that suppresses nothing. What a directive
suppresses depends on which rules are *enabled*, so narrowing the rule set
does not narrow `RUF100`'s output — it manufactures findings, by disabling
the rules the directives were live against. The narrowed number is not a
subset of the real one and is not an under-count of it; it is an answer to a
different question.

Measured over the seven test cells promoted in the pass that found this: the
full config reports **one** dead directive, and `--select RUF100` reports
**twenty-one**. The twenty extras name `ASYNC251`, `N815`, `F821`, `E731`,
`N803`, `UP031`, `F401` and `E402`, and every one of them is live.

So a narrowed `RUF100` run is wrong in both modes. With `--fix` it deletes
live suppressions, which the paragraph above already prohibits. Read-only it
still reports directives that are doing their job, and acting on that report
by hand deletes exactly the same directives one at a time. Count this rule
under the full config or not at all.

## Changing a decline

Declines carry a `[category]` marker, checked by `tests/test_lint_policy.py`:

| Category | Claim | What it must carry |
|---|---|---|
| `presentational` | no finding can be a behaviour difference | if ruff marks any fix *unsafe*, why that unsafety is not a behaviour difference here |
| `covered-elsewhere` | another tool enforces the property | which tool, or which rule — and that rule has to be one ruff actually enforces |
| `behavioural` | findings can be real; the decline is argued | the argument, on the continuation lines |
| `provisional` | findings can be real and the decline is **not** argued | the measured count, compared against what ruff reports today |

The fourth exists so that an unargued decline is *countable*. The total of
those counts has a target of zero.

**The category is a property of what a fix would do, not of what a rule is
named.** Read the findings before assigning one. Two entries here were filed
by their names and were wrong about their own sites: one described a loop as
`async for` when it was not, and one gave a reason that misstated what the
rule detects — which is why neither rule's findings had ever been read.

**A rule declined repo-wide for a handful of findings is an accommodation,
not a posture.** The threshold applied here: **three or fewer findings ⇒
un-decline**, unless the family carries a written posture that binds its
members. Three such postures are in the config today, and the list is not
closed — a fourth is written the day a family needs one:

| Family | The posture that survives a low count |
|---|---|
| `PTH` | a gradual transition, declined as a whole while it runs |
| `ARG00x` | sibling coherence: an unused argument is part of a signature its siblings share |
| `TC00x` | the hazard is a property of the *library*, not of the count — today's three findings are `import pytest`, but the next could be one that resolves annotations at runtime |

`TC00x` is the one that shows why the count alone cannot decide. It sits at
exactly three, and un-declining on that basis would trade a bounded backlog
for an unbounded runtime failure. A site that genuinely needs an exception
takes a per-line `# noqa` naming the rule and the reason.

Prefer per-line over per-file. A per-file waiver also unflags a **future**
finding of that code in that file — the cost
`.claude/rules/async-transport.md` names for the per-file form, paid
deliberately, per file, with the file read first.
