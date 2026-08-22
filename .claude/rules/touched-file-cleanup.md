# Touched-File Cleanup — clear what the file you opened owes

## The rule

> **A change that opens a file clears the type-checker findings that file
> owes, and lowers the ceiling in the same pull request.**
>
> **Unless the file owes more than 25** — then leave it, say so in the pull
> request, and lower nothing.

Ask what a file owes; do not estimate:

```bash
uv run python bin/quality-contract.py charge --tool mypy <path>
```

A clean file answers `nothing outstanding`, which is the whole point of the
command: `0` means *paid*, so the rule is self-checkable rather than a matter
of conscience.

## Why a convention rather than a check

The backlog this drains is not distributed the way a gate could exploit. It
sits in the cells the contract marks `transitional`, and slightly more than
half the files those cells track are already clean — so a rule keyed to *files
a change happens to open* charges most pull requests nothing, and charges the
rest an amount that falls as the backlog it is drawn from shrinks. A
standing per-change tax against a static backlog is a different and worse
shape: it never stops charging, because nothing it charges for goes away.

The cost is therefore front-loaded and self-extinguishing, and the median
change pays zero. What makes that bearable is the cap, and what makes the cap
honest is that `charge` settles any individual claim in one second.

**This is not a schedule.** Nothing is gated on the backlog reaching any
particular number, and no file is anyone's assignment. The convention clears
what ordinary editing touches; what ordinary editing never touches is exactly
the residue a scheduled pass would still have to collect, and it is smaller
than the whole.

## Scope — the cells the contract marks `transitional`

The rule applies to **mypy findings, in the cells
`.dataknobs/quality-contract.json` gives tier `transitional`**, and to nothing
else. The declaration is where that is written; `census` prints it alongside
what each cell currently measures:

```bash
uv run python bin/quality-contract.py census --tool mypy
```

Everything outside that is already stricter than this convention could make
it. **Every ruff cell and every formatting cell stands at ceiling 0**, as does
every `strict` mypy cell: there, a new finding fails the gate unconditionally,
today, with nothing to opt into. This document adds nothing for those and
should not be read as softening them.

**The `unchecked` cells are silent, and the silence is not a loophole.**
`packages/*/tests` is outside the type checker's target set, so a change there
encounters nothing and owes nothing. That is deliberate: it is what keeps this
rule cheap enough to adopt now. Widening the measured surface is a separate
decision with its own cost, and pre-empting it by hand — clearing findings in a
cell nothing measures — is work nobody asked for and no ceiling records.

## Lowering the ceiling is not optional

Clearing findings without lowering the ceiling leaves headroom, and headroom is
a regression budget: findings a later change can reintroduce with every run in
between reporting green, because nothing the check compares will have moved.

`check` therefore fails a cell measuring **below** its ceiling exactly as it
fails one measuring above, and prints the command that fixes it:

```bash
uv run python bin/quality-contract.py update-baseline --tool mypy --cell <cell>
```

So the second half of the rule is enforced even though the first half is not.
A change that clears findings and stops there does not pass — which also means
you cannot forget it silently, and that the contract's diff is where your
progress gets recorded.

**Partial payment is fine.** A change clearing three findings lowers the
ceiling by three. The cap governs what a change *owes*; the ratchet records
what it *paid*. They do not interact, and a change that clears nothing measures
at ceiling and passes.

## Fix, don't suppress

A finding silenced with `# type: ignore[...]`, a `[[tool.mypy.overrides]]`
section that relaxes the rule for the module, or an annotation widened to
`Any` **satisfies the ceiling while defeating the point**. It
is a lower count and a worse tree, and it is worse in the specific way that
matters here: the ceiling stops being a measurement of how much is wrong.

| | |
|---|---|
| **Clears** | the finding is gone because the code or the type is right |
| **Suppresses** | the finding is gone because something was told not to report it |

Suppression is legitimate in its own right — see
[`lint-policy-authority.md`](./lint-policy-authority.md) for who decides what is
an error, and `docs/development/mypy-configuration.md` for why a directive must
name the code it suppresses. What is illegitimate is
reaching for it *because a convention asked you to clear something*. If the
honest answer is that the finding needs a suppression, write the suppression
with its reason and note in the pull request that the ceiling fell for that
reason rather than because the code improved.

## Reproduce-first still applies

**A quarter of this backlog is judgement work.** `union-attr` and
`attr-defined` together are about 25% of it — the same share as
`no-untyped-def`, which is rote — and they are judgement about code the person
clearing them usually did not write.

**That is the hazard this clause exists for.** A `union-attr` finding cleared
quickly, in unfamiliar code, during an unrelated change, is precisely where a
real latent bug gets papered over with a narrowing cast or an `assert` that
happens to be false in production. The type checker was right and the fix
silenced it.

So: if clearing a finding means changing behaviour, or asserting something
about a value the code did not previously assert, **it is a bug fix and the
reproduce-first workflow applies** — a failing test first, then the diagnosis,
then the change. If clearing it is genuinely mechanical, it is not. Deciding
which is the work; the convention does not license skipping that decision
because the finding arrived incidentally.

See what a cell is actually made of before assuming which kind you are looking
at:

```bash
uv run python bin/quality-contract.py census --tool mypy --cell <cell>
```

## A companion pull request is acceptable

Expect about one change in ten to open a file whose charge is real extra scope.
**Moving the cleanup to its own pull request is fine and always was** — this is
a claim about the work being done, not about it landing in one diff. Say which
change carries it, so the reviewer of the first one is not left wondering.

What is not fine is the third option: clearing nothing, lowering nothing, and
saying nothing. The rule survives collisions with deadlines by being explicit
about them, and dies the first time it is quietly dropped instead.

## Scheduled cleanup carries a trailer

Once ceilings follow measurements down, **a falling ceiling stops meaning one
thing**. It means either that this convention did its work incidentally, or
that someone sat down and drained a cell on purpose. Those are different
claims, and the contract's history cannot tell them apart after the fact.

So a **leg** — a commit whose *purpose* is clearing backlog, as opposed to one
that cleared some on its way past — carries a trailer naming the cell it
drained:

```
Quality-Leg: packages/data/src
```

- **Presence is the discriminator.** A commit without it that lowers a ceiling
  is incidental cleanup, which is the common case and stays ceremony-free.
- **The value is a cell path, spelled exactly as `.dataknobs/quality-contract.json`
  spells it.** Not a planning identifier: this repository's commits do not carry
  those, because they mean nothing to a reader without access to the planning
  system and they age instantly. A cell path also has the property an identifier
  does not — it can be checked against the declaration.
- **Repeat the trailer** once per cell if one commit drained several.

Ordinary work needs no trailer. Deliberate cleanup is rare, so the asymmetry
puts the ceremony where it is cheap.

## The record

`.dataknobs/quality-contract.json` is committed, so `git log` over it *is* the
time series — what fell, when, and in which cell. No separate artifact records
this and none should: a per-run file would be a snapshot where the question is
a series, and one that changes on every run conflicts on every run.

Read it with:

```bash
uv run python bin/quality-contract.py ledger --tool mypy

# A fixed window, opening after a named boundary rather than at the beginning
uv run python bin/quality-contract.py ledger --tool mypy --since <sha>
```

That history is why the rule can be evaluated rather than merely believed. If
it turns out to charge more than it drains, the evidence for that is already
being collected, by the ratchet, with nobody doing anything extra — and the
`Quality-Leg:` trailer is what keeps the two populations apart in the reading,
so the result can be stated as a pair rather than as a total.

**Read the convention's row, not the total.** The command prints a mean and an
idle fraction for each population separately, because a drain achieved entirely
by legs would clear the bar the total is quoted against while showing nothing
at all about what this rule costs.

## Related

- [`lint-policy-authority.md`](./lint-policy-authority.md) — the config decides
  what is an error, and `explain` answers why a finding is or is not reported
- [`code-validation.md`](./code-validation.md) — running the checks
- `docs/development/mypy-configuration.md` — tiers, ceilings, and how
  `bin/validate.sh` reaches a verdict
- `docs/development/quality-checks.md` — the contract's commands in full
