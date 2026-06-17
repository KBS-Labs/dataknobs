# Consumer Extensibility

## The Mindset

DataKnobs is shared infrastructure consumed by many projects. We're
building **general tools for others to extend**, not tight production
systems that solve one use case. Every "we'll add it when someone
asks" decision pushes work onto consumers — they hit the gap, write
a workaround, file a request, wait for our turnaround, then migrate.
That cost belongs on us, not them.

## The Test

When evaluating whether to add an extension point, ask:

> **"Would we want this flexibility if we were a consumer of our own project?"**

If yes, ship it — even if no consumer has asked yet. Including a
small extension point is almost always cheaper than N consumers each
writing a workaround while waiting.

## Bias Toward Extension Points

When a problem can be solved by either a special-case knob OR a
general extension surface for ~the same cost, prefer the surface:

| Tighter | More General |
|---|---|
| One-off config field for a specific behaviour | Registry consumers can extend with their own entries |
| Hard-coded vocabulary / defaults | Constructor-injectable vocab + dotted-path overrides |
| Special-case dispatch branch in the runtime | New hook following an existing pattern (`on_enter`, `on_exit`, ...) |
| Implicit "we'll never need this" assumption | Documented protocol with "here's how to add your own" |
| Subclass-only flexibility | Constructor injection / registry (subclassing is fine but adds a cliff) |

Often the dataknobs feature you're shipping becomes the **first
adopter** of the extension surface — the same code with one less
hard-coded coupling.

## Prudence

"General tools" does NOT mean "go overboard." Don't speculate
without need:

- **No new abstractions without a concrete trigger.** Evaluate
  seriously; don't expand for its own sake.
- **No new dependencies for hypothetical use cases.** Premature
  flexibility costs maintenance.
- **No protocol changes that lock in a guess.** If the right shape
  isn't clear, ship the simpler thing and add the extension point
  when a real second consumer surfaces the right API.

## Decision Framework for "Defer or Ship"

When a request surfaces an adjacent opportunity, ask:

1. **Consumer cost of NOT shipping** — what workaround will they
   write? How long until they hit it? How many consumers will?
2. **Our cost of shipping** — design complexity, surface size,
   maintenance burden.
3. **Is the right API shape clear, or am I guessing?**

Verdict:

- Shipping leverages existing infrastructure + shape is clear →
  **bias to ship**
- Shape is unclear OR requires a brand-new abstraction layer that
  may get reshaped → **defer + capture**
- Cost is large but consumer pain is real → **scope down, ship the
  smaller version + capture the larger follow-up**

## The Deferral Bar — The Toolkit Standard

DataKnobs is a toolkit consumers grab off the shelf and **run to
production** — not grab off the shelf, then stop and ask us to
supply the rest of what their use case needs. That standard sets a
hard bar on what may be deferred.

**"We shipped the Protocol; consumers can subclass / implement it"
is NOT a sufficient reason to defer.** A reference implementation of
a standard production behavior is part of the product, not an
optional extra. If a reasonable production consumer would
predictably need a behavior and we ship only the abstract seam,
we've shipped a homework assignment, not a tool. The Protocol pins
the *shape*; the reference impl delivers the *capability* — both
ship together unless one of the categories below applies.

An item may stay deferred **only** if it falls into one of these
five categories. Name the category explicitly in the deferral
rationale:

1. **Blocked on unbuilt substrate** — depends on infrastructure
   that does not exist yet (a later wave, an unbuilt framework).
   Ships when the substrate lands.
2. **Genuinely speculative shape, no consumer blocked** — the API
   form is a real guess AND no production consumer is currently
   forced into a workaround by its absence. (If a consumer *is*
   blocked, the shape is no longer speculative — they've shown you
   the use case.)
3. **Premature optimization** — a performance/efficiency refinement
   with no demonstrated need (e.g. per-field caching before any
   profile shows the cost).
4. **Internal / maintainer tooling** — adoption lints, scaffolding,
   things consumers never touch.
5. **New abstraction needing its own design pass** — not a missing
   reference impl of an existing seam, but a genuinely new construct
   whose design space warrants its own brief.

If none of these fit, the honest disposition is **ship it** — even
absent a specific request. "A consumer could write it themselves"
disqualifies an item from deferral; that *is* the work we're
supposed to have done for them.

Note the interaction with the Maturity Caveat below: that section
keeps a CAPABILITY in scope even without current demand; this bar
says that once a capability *is* in scope, shipping its reference
impl alongside the Protocol is the default, and withholding it
needs one of the five categories — not just "the seam exists."

## Maturity Caveat — SHAPE vs CAPABILITY

Consumer surveys cut cleanly along the SHAPE dimension. CAPABILITY
scope needs separate treatment when adopters are at early project
maturity.

### The principle

**Absence of consumer demand for a capability is a weak signal when
adopters are at early project maturity.** Consumers don't ask for
what they haven't seen. A capability that's industry-standard for
production-grade systems (observability hooks, distributed tracing,
calibrated confidence, retry / circuit-breaker, external pub-sub
fan-out, multi-tenant resource coordination, structured-output LLM
modes, etc.) is known-needed regardless of whether current
spike-stage / early-build-out adopters have hit the wall yet.

Consumer surveys help pin **SHAPE decisions** where consumers have
made concrete architectural choices (sync vs async, payload shape,
resolver pattern, state-mutation contract, error semantics —
deliberate convention choices that wouldn't change with maturity).
Surveys do NOT inform **CAPABILITY scope** where absence of demand
is more likely a maturity artifact than a true signal that the
capability is dispensable.

### How to apply

When deciding what stays in a follow-up's design pass:

- **SHAPE decisions** legitimately defer until 2–3 concrete
  adopters surface to pin the right API. Adopter survey evidence is
  the correct base.
- **CAPABILITY scope** retains industry-pattern anchors even
  without current concrete demand. The "would we want this
  flexibility if we were a consumer of our own project?" lens (see
  "The Test" above) is the correct base — applied beyond
  current-demand evidence.

When writing the follow-up's rationale column:

- If the defer is shape-driven, say so explicitly: e.g. "Why
  defer (SHAPE pinning only — not capability scope)."
- Enumerate the CAPABILITY anchors that stay in scope as a
  separate item, so the next person picking up the design pass has
  the full surface to work from rather than relitigating "did
  anyone ask for this?" against current adopter snapshots.

### Reinvention as evidence

When a consumer needs a capability and dataknobs doesn't provide
an accessible primitive, they build their own — parallel
implementation, subclass + monkey-patch, fork. That reinvention is
direct evidence the capability is needed; discoverability /
accessibility / shape failures can drive the gap even when a
related primitive exists. The fix is BOTH the primitive AND
prominent docs / worked examples / typed-facet surfaces so the
**next** consumer adopts rather than reinvents.

Signals to watch for in adopter surveys:

- **Subclasses that override a single method** — the override IS
  the hook the surface should expose.
- **Monkey-patches with documented "TEMPORARY MITIGATION" markers**
  — the marker is the workaround's lifespan acknowledgment;
  treat it as a flagged consumer request.
- **Parallel implementations citing dataknobs as "spirit-inspired
  but not using"** — accessibility / shape failure of the existing
  primitive, even though one is already shipped.

### Anti-pattern: confusing absence-of-current-pain with absence-of-need

If a survey of early-stage consumer projects shows "nobody is
building per-step observability for their reasoning pipeline," do
NOT conclude "per-step observability is dispensable." Conclude
"per-step observability is the next wall consumers will hit at
production maturity, and we should have the infrastructure ready
when they arrive."

Industry pattern + first-principles reasoning + the "would we want
this if we were a consumer of our own project?" lens are the
correct evidence base for **CAPABILITY scope**; consumer-survey
evidence is the correct base for **SHAPE pinning only**.

## Capture, Don't Dismiss

Every brief / impl-plan should have a "Follow-up items captured for
later investigation" section with rationale per deferral. The
following are NOT valid deferral reasons in isolation:

- "No concrete consumer asked yet" (pair with "API shape unclear"
  or "interacts with feature X we haven't designed". For
  industry-pattern CAPABILITY items, see the "Maturity Caveat —
  SHAPE vs CAPABILITY" section above — capability scope retains
  these even without current demand).
- "Premature abstraction" (pair with "design space too large to
  commit without seeing a second use case").
- "Defer until requested" (without other justification).

When you do defer, give the item a stable ID (e.g. `162-FU1`,
`163-FU3`) so it can be lifted into a tracker row + context brief
when prioritized.

## Anti-Patterns

| Don't | Do |
|---|---|
| Drop "out of scope" items without rationale or capture | Capture as numbered follow-up with cost/shape analysis |
| Ship a single-purpose knob when a small extension surface costs the same | Ship the surface; make the knob the first adopter |
| Hide flexibility behind subclassing-only patterns | Prefer constructor injection / registries |
| Add abstraction layers "just in case" | Evaluate seriously — the principle is about lens, not maximalism |
| Treat consumer migration cost as their problem | Treat it as our problem; design to minimize it |
| Drop CAPABILITY scope on absence-of-demand grounds when adopters are at early project maturity | Apply the SHAPE-vs-CAPABILITY distinction — surveys inform SHAPE pinning; capability scope retains industry-pattern anchors per the maturity caveat |
| Treat a consumer's subclass / monkey-patch / parallel reimplementation as "they solved it themselves" | Treat it as a flagged consumer request — the workaround IS the missing surface signal |
| Defer a reference impl because "the Protocol ships and consumers can subclass it" | Ship the reference impl with the Protocol unless one of the five deferral-bar categories applies; the seam alone is a homework assignment, not a tool |
| Defer on "no one asked + a consumer could write it themselves" | "Could write it themselves" is the work we owe them — name a deferral-bar category or ship it |

## Examples in Recent Work

- **Stage-primitive synthesizer registry** (Item 163): ships
  `intent_confirm:` as the first adopter, but consumers register
  their own primitives (`vendor_select:`, `policy_review:`) without
  dataknobs turnaround.
- **Turn-lifecycle hook surface** (Item 164): the inbox bridge is
  one auto-registered hook; the surface itself extends for any
  pre/post-turn logic.
- **`StructuredConfigConsumer.components` pass-through** (Item 162):
  engages existing mixin infrastructure so consumer composing
  strategies inherit the pattern.
- **Two-adopter survey + maturity caveat** (Item 164 v4): surveyed
  two consumer projects to pin SHAPE decisions (sync/async, payload
  shape, resolver pattern, state-mutation contract — all PINNED via
  concrete consumer evidence) but explicitly retained CAPABILITY
  scope for industry-pattern needs the surveys didn't validate
  (per-step observability hooks, external pub-sub fan-out via
  EventBus substrate). Reinvention signals (ayler's parallel
  `CompilationEventPublisher` reimplementing dataknobs's
  `InMemoryEventBus`-shape pattern) treated as flagged consumer
  requests, not "they solved it themselves." Industry-pattern
  anchors enumerated in the follow-up rationale so the next person
  picking up the design pass works from the full surface, not from
  current-adopter-snapshot evidence alone.
