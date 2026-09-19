# Placing a Phrase Against a Vocabulary

`CascadingResolver` turns a string into ranked entities, each carrying the
reason it was produced. It asks its rungs in order and stops when it has
enough — so an exact hit outranks a fuzzier one because of *where its rung
sits*, not because of a weight anywhere. The composition is the policy, and
the composition is a list.

Nothing here opens a connection, reads a file or awaits. A cascade over a
vocabulary someone typed is dictionary lookups all the way down, which is why
the synchronous flavour is the native one and the asynchronous is the
accommodation.

## Where the names live

On the package door, and on this family's own. `dataknobs_common` re-exports
these names, and `dataknobs_common.entity_resolution` is where they are
defined — the same names either way. What the family still imports nothing
from is the *vocabulary* package, which is the property that keeps a rung
free of it; [Where this package sits](#where-this-package-sits) says why that
direction is the one that matters.

```python
from dataknobs_common.entity_resolution import (
    AliasSignal,
    CascadingResolver,
    EvidenceKind,
    ExactNormalizedSignal,
    LexicalSignal,
    ScanningSignal,
)
```

## The whole input

Hand-edited, no tooling, no second file. The same vocabulary the
[ontology guide](https://kbs-labs.github.io/dataknobs/packages/common/ontology/)
publishes, at the version that declares a breed and the breed it is a kind of —
which is what makes the sentence below carry two forms at overlapping spans:

<!-- worked-input -->

```yaml
# mammals.yaml
ontology:
  id: mammals
  version: "1.1"

  entity_types:
    - id: Species
      attributes:
        - {name: latin_name, type: string, required: true}
        - {name: lifespan_years, type: number, field_type: float}
    - id: Breed
      isa: Species                        # <-- the TYPE lattice
      attributes:
        - {name: akc_group, type: string}

  relation_types:
    - id: isa
      transitive: true

  entities:
    - {id: mammal, type: Species, name: Mammal,
       description: "Warm-blooded, milk-producing vertebrates."}
    - {id: dog, type: Species, name: Dog, aliases: [Canine, "Domestic dog"],
       description: "A domesticated carnivoran."}
    - {id: retriever, type: Breed, name: Retriever}
    - {id: golden_retriever, type: Breed, name: Golden Retriever, aliases: [Goldie]}
    - {id: beagle, type: Breed, name: Beagle, aliases: [Beagles],
       source: {source_id: clinic_db, table: species, key: "sp-2291"}}

  assertions:
    - {subject: dog, relation: isa, object: mammal}            # <-- the INSTANCE
    - {subject: retriever, relation: isa, object: dog}         #     lattice, same
    - {subject: golden_retriever, relation: isa, object: retriever}   # relation id
    - {subject: beagle, relation: isa, object: dog}
    # an attribute value is an assertion whose object is a Literal
    - {subject: dog, relation: lifespan_years, object: 12}

  taxonomies:
    - {id: species, name: Species, relation: isa}
```

Reading a `.yaml` path needs PyYAML, which `dataknobs-common` does not install
by default — `pip install dataknobs-common[yaml]`. Nothing else on this page
does: `load_ontology` also takes a `.json` path or a plain mapping, and either
runs on the base install.

## The worked call site

A sentence rather than a phrase, which is the case this page is for: the caller
does not know where the forms are, or how many, or what the vocabulary missed.
Every line below runs against the file above, exactly as written.

<!-- worked-call-site -->

```python
from pathlib import Path

from dataknobs_common.entity_resolution import CascadingResolver, ScanningSignal
from dataknobs_common.ontology import load_ontology

onto = load_ontology(Path("mammals.yaml"))  # -> Ontology
# No database. No embedder. No event loop.

resolver = CascadingResolver([ScanningSignal(onto.entities)], onto.entities)
result = resolver.resolve("my golden retriever has been limping", k=5)

# (1) which declared forms the sentence carries, longest first
[c.entity_id for c in result.candidates]  # ["golden_retriever", "retriever"]

# (2) where each one sat -- half-open offsets into `result.query`
result.explain("golden_retriever")[0].span  # (3, 19)
result.explain("retriever")[0].span  # (10, 19)
result.explain("golden_retriever")[0].matched_text  # "golden retriever"

# (3) what the vocabulary did not account for
result.coverage.matched  # ((3, 19),) -- the union: (10, 19) is inside it
result.coverage.unmatched  # ((0, 2), (20, 36))
result.unmatched_text()  # ("my", "has been limping")
```

That block is executed as written by a workspace test, and the test asserts it
is character-identical to the fence above. If this page and the code ever
disagree, the suite goes red rather than the page going quietly wrong.

`ScanningSignal` is constructed by name rather than reached through
`build_resolver`, and that is the subject rather than a convenience: a page
showing a reader how to find declared forms inside a sentence should name the
rung that does it. [The door, and what it builds](#the-door-and-what-it-builds)
is the other route, and a document declaring no `resolver:` section gets this
rung too — last, behind the two that compare the whole query.

## The door, and what it builds

```python
from pathlib import Path

from dataknobs_common.ontology import build_resolver, load_ontology

onto = load_ontology(Path("mammals.yaml"))
resolver = build_resolver(Path("mammals.yaml"), onto)   # no store, no loop

result = resolver.resolve("beagles", k=5)
assert result.candidates[0].entity_id == "beagle"
```

Two functions rather than one, because an `Ontology` is a **value**: it owns no
lifecycle and has nowhere to put a runtime. A caller wanting both makes two
calls and holds two objects.

A document that declares no `resolver:` section gets the declared order over
what the value can serve — `ExactNormalizedSignal`, then `AliasSignal`, then
`ScanningSignal`. The scan is there because silence has to build something a
caller can use: without it the default compares the whole query and nothing
else, so a sentence comes back with no candidates and the whole string
unmatched. It sits **last** because the three never disagree — they read one
index two ways, and the composition returns the same candidates at the same
spans whichever end the scan sits at. What the position decides is which rung
is of *record*, and last leaves that `exact` for a caller whose string already
*is* the phrase. A document that declares `rungs: []` gets a cascade that
misses everything. Those are different answers on purpose: silence is silence,
and an empty list is a composition somebody wrote.

## Every candidate carries its evidence

*Why did this win?* is answerable without re-running the query.

```python
top = result.candidates[0]

assert [e.signal for e in top.evidence] == ["exact", "alias", "scan"]
assert top.evidence[0].kind is EvidenceKind.DECLARED
assert top.declared                                   # some evidence is DECLARED

assert [e.signal for e in result.explain("beagle")] == ["exact", "alias", "scan"]
```

`beagles` is `beagle`'s declared alias, so all **three** default rungs produce
it: the exact rung through the folded form index, the alias rung through the
alias index, and the scan through the folded index again — a one-word query is
a single token slice, so the scan probes it whole. The candidate appears
**once**, positioned by the earliest rung, carrying three pieces of evidence —
a duplicate appends evidence and moves nothing.

`explain()` returns the evidence tuple rather than a mapping of signal to
score. A mapping cannot carry `kind`, `scoring`, `matched_text` or `span`, and
since a rung's name is an open string it would silently drop one of two rungs
that happened to share one. Asking about an id that is not a candidate raises
`KeyError`: an empty tuple is what a candidate that *inherited* its match
legitimately carries, so it cannot also mean *absent*.

## What a score is, and what it is not

A cascade's candidates are heterogeneous by construction — an exact `1.0` and
a cosine `0.83` are not the same measurement — so the kind travels with each
piece of evidence rather than being declared once per result.

```python
from dataknobs_common.entity_resolution import Scoring

assert top.evidence[0].scoring is Scoring.DECLARED     # 1.0 by fiat, not a distance
assert result.as_distribution() is None                # DECLARED does not normalize
```

`as_distribution()` returns `None` rather than an approximation, ever. It
refuses on a corpus whose compatibility verdict is `INCOMPATIBLE`, and on any
candidate whose rung of record produced a kind that does not normalize.

`ranked()` is the other half: order survives every scoring kind, because a
cascade positions by rung rather than by number. It does *not* survive an
`INCOMPATIBLE` corpus, so a caller putting candidates in front of a person
reads `compatibility` first.

## When the query does not spell the form

Every rung above answers a **lookup**: the form is in the vocabulary or it is
not. So a query carrying a typo reaches none of them, and a cascade of all
three returns nothing at all.

`LexicalSignal` is the rung for that case. It compares each window of the query
against every form the vocabulary declares, and proposes the ones that came
close — so it is the first rung here whose score is a **measurement** rather
than a `1.0` by fiat.

<!-- worked-near-spelling -->

```python
from dataknobs_common.entity_resolution import (
    AliasSignal,
    ExactNormalizedSignal,
    LexicalSignal,
    ScanningSignal,
)
from dataknobs_common.ontology import Entity, MappingEntitySource

breeds = MappingEntitySource(
    {
        "golden_retriever": Entity(id="golden_retriever", type="Breed", name="Golden Retriever"),
        "retriever": Entity(id="retriever", type="Breed", name="Retriever"),
    }
)
typo = "my goldne retriver has been limping"

for rung in (ExactNormalizedSignal, AliasSignal, ScanningSignal):
    print(f"{rung.__name__:21} {rung(breeds).candidates(typo, k=5)}")

for candidate in LexicalSignal(breeds).candidates(typo, k=5):
    found = candidate.evidence[0]
    print(
        f"{candidate.entity_id:17}{found.matched_text!r:19}{found.score:.3f}"
        f"  {found.kind.value}/{found.scoring.value}  {found.span}"
    )
```

```
ExactNormalizedSignal []
AliasSignal           []
ScanningSignal        []
retriever        'retriver'         0.941  inferred/native  (10, 18)
golden_retriever 'goldne retriver'  0.903  inferred/native  (3, 18)
```

Every block in this section is executed, and the output block under each one is
compared against what running it prints. The numbers here are the whole reason
the section exists, so a number the code stopped producing takes the suite red
rather than leaving the page quietly wrong.

The span points into the **query** — at the words the caller actually typed —
rather than at the form they were reaching for. That is the half a consumer
cannot recover: they already have the vocabulary, and what they do not have is
where in their own sentence the near-miss sat.

`kind` is `INFERRED` because the entity was *proposed* rather than found: the
vocabulary carries nothing the query said. `scoring` is `NATIVE` even though a
ratio is already in `[0, 1]`, because the number means whatever the configured
scorer means — two rungs with different scorers produce incomparable `0.8`s, so
it stays out of `as_distribution()` by construction.

A near-spelling hit scoring `0.94` still sits **behind** a declared hit scoring
`1.0`, and not because `0.94` is the smaller number: a cascade positions by the
first rung that produced an id. Put this rung after the declared ones.

### What it does not change: coverage

This is the first rung here whose evidence is `INFERRED` *and* carries a span.
Before it, *inferred* meant *unlocated* by construction — a cosine neighbour
has no position in the utterance it neighbours — so `Coverage` could test for
a span and get the right answer. A near-spelling proposal is located, so that
implication became a condition: **`Coverage` counts `DECLARED` spans.**

`DECLARED` is the half kept because *what the vocabulary accounted for* is the
question both coverage fields are read for, and a near-spelling proposal is
the rung reporting that the vocabulary accounts for **none** of what the query
said. Two things follow:

- **A phrase this rung resolved is still reported as uncovered.** For the typo
  query above, `unmatched_text()` is the whole sentence with this rung and
  without it. That is what a maintainer wants: the vocabulary does not carry
  `goldne retriver`, and now they also get a candidate naming the entry it was
  probably reaching for.
- **An overreaching window cannot widen `matched`.** The rung reports every
  window that cleared the threshold rather than choosing one, so on the
  *correctly spelled* sentence it also proposes `golden_retriever` across
  `my golden retriever` (`0.914`) and `golden retriever has` (`0.889`). Merged
  positionally those would report `my` and `has` as covered. They are not.

So adding this rung to a cascade adds candidates and never coverage. Its own
spans are not hidden — they are on the evidence, where
`result.explain(entity_id)` hands them over with each piece's `kind`, and
`EntityCandidate.declared` answers the same question per candidate. A consumer
wanting *everywhere any rung read something* builds it from those; coverage
answers the narrower question, which is the one that is hard to reconstruct.

### The threshold, and what a lower one buys

<!-- worked-threshold -->

```python
for threshold in (0.60, 0.75, 0.85, 1.00):
    found = [
        evidence.matched_text
        for candidate in LexicalSignal(breeds, threshold=threshold).candidates(typo, k=9)
        for evidence in candidate.evidence
    ]
    widest = max(found, key=len, default="-")
    print(f"{threshold:.2f}  {len(found):2} hit(s)   widest window: {widest!r}")
```

```
0.60  11 hit(s)   widest window: 'my goldne retriver has been'
0.75   5 hit(s)   widest window: 'goldne retriver has'
0.85   2 hit(s)   widest window: 'goldne retriver'
1.00   0 hit(s)   widest window: '-'
```

The default is `0.85`, and the table is why. What a lower threshold buys is not
*more entities* — it is the **same two entities at spans that run past them**.
At `0.75` the rung reports `retriever` across `retriver has`, reaching into a
word the vocabulary never matched; at `0.60` it reports `golden_retriever`
across most of the sentence. The entity is right and the offsets are wrong,
which is the worse of the two failures for a caller who highlights what the
span points at.

`1.00` is a real request rather than a mistake — *only an exact fold* — and is
kept for it.

The threshold is also what decides how far the probe reaches, so those are one
effect and not two. A window can only score `t` against a form of length `|F|`
while it is no longer than `|F| × (2 − t) / t` characters, so the rung stops
widening there — and every window that bound removes is one that would have
answered *below the threshold* for every form. It is arithmetic rather than a
budget, the way `ScanningSignal`'s token bound is, and it is **not** that bound:
a scored probe can match a window wider than the longest declared form, which
is what the typo class *a space where the vocabulary has none* requires.

And `0.60` is not an arbitrary low number: it is `difflib`'s own default cutoff,
and it is right where that default is used, on long configuration keys a caller
has already half-typed. A vocabulary carries three-letter forms, and any
three-letter word is within one edit of many others:

<!-- worked-cutoff -->

```python
sentence = "the log fell over"
animals = MappingEntitySource({"dog": Entity(id="dog", type="Species", name="Dog")})
for threshold in (0.60, 0.85):
    rung = LexicalSignal(animals, threshold=threshold)
    print(
        f"{threshold:.2f}  "
        f"{[(str(c.entity_id), c.evidence[0].matched_text) for c in rung.candidates(sentence, k=5)]}"
    )
```

```
0.60  [('dog', 'log')]
0.85  []
```

### What it costs, and the seam for when that is too much

The rung scores **every window against every form**, so the cost is the product
rather than the vocabulary size. Measured on one laptop, over a nine-token query
and a vocabulary of two-word names — indicative, not a promise, and the shape is
the point rather than the milliseconds:

| entities | forms | per query |
|---|---|---|
| 100 | 200 | ~3 ms |
| 1,000 | 2,000 | ~30 ms |
| 10,000 | 20,000 | ~315 ms |

Linear in the vocabulary, which means there is a size at which the standard
library stops being free. `scorer` is the seam for that: any
`Callable[[str, str], float]`, and `rapidfuzz.fuzz.ratio` is the usual one.
Measured 2026-09-17 with `rapidfuzz` installed, the two **agree to two decimal
places on every case tried**, so taking the dependency costs no answers — which
is exactly why it is a seam here rather than a dependency in `pyproject.toml`.
A consumer whose vocabulary is small never installs anything, and one who needs
the fast end writes `LexicalSignal(breeds, scorer=fuzz.ratio)` and takes the
dependency in their own tree. That line is prose rather than a fence precisely
because this repository does not have `rapidfuzz` installed: a block the suite
cannot run is a block nothing checks, and every other block in this section is
executed.

The cost also grows with the **query**, and nothing about the vocabulary
bounds that. The threshold decides how *wide* a window may be; how many there
are is the caller's token count, and each one costs a scorer call per declared
form where a scan's costs a dictionary lookup. Measured: linear in the token
count, and a nine-hundred-token paste over a five-hundred-entity vocabulary is
two seconds of one CPU. `max_query_tokens` is the cap for text a caller did not
write, and it is off by default because it is the one parameter here that can
cost an answer:

A query over the cap is **refused rather than truncated**. Probing the first
*n* tokens of a paste and answering from them is a plausible-looking answer to
a question nobody asked — the entity is as likely to be in the tail as in the
head — so the refusal names the count and leaves the repair, chunk the text or
raise the cap, to the caller.

<!-- worked-cap -->

```python
from dataknobs_common.exceptions import ValidationError

capped = LexicalSignal(breeds, max_query_tokens=4)
try:
    capped.candidates(typo, k=5)
except ValidationError as refused:
    print(refused)
```

```
this query carries 6 tokens and max_query_tokens is 4. A near-spelling rung scores every window against every declared form, so the work grows with the caller's own text -- chunk the text, or raise the cap for a vocabulary small enough to afford it.
```

**Do not pass `rapidfuzz.fuzz.partial_ratio`.** It scores any substring `1.0`,
so over a vocabulary whose forms contain one another — which is this one —
`gold retriever` matches `retriever` at `1.00` and picks the wrong entity. Its
"it can locate" property is bought with exactly the discrimination this rung
exists to have, and it also breaks the inequality the probe's bound is derived
from, so it silently shortens the probe as well.

### What the source has to be able to do

This is the one rung here that needs something `EntitySource` does not publish.
Every member there takes a form and answers with ids; a query spelling no form
has nothing to hand them. So the source must also satisfy
`SurfaceFormCatalog` — one member, `surface_forms()`, answering with every form
the vocabulary declares:

<!-- worked-catalogue -->

```python
from dataknobs_common.entity_resolution import SurfaceFormCatalog

print(isinstance(breeds, SurfaceFormCatalog))     # structural: nothing to register
```

```
True
```

`MappingEntitySource` satisfies it for free, because its index is already keyed
by the folded form. A source that does **not** is refused when the rung is
constructed, rather than yielding a rung that matches nothing — which is where
this differs from `AliasSignal`, whose fallback is *this vocabulary declares no
aliases*. A vocabulary may legitimately declare no aliases; none has no forms,
so an empty answer there is a fact and here would be a misconfiguration wearing
the costume of one.

It is a separate protocol rather than a member on `EntitySource` for the reason
`AliasFormSource` is: `EntitySource` is `@runtime_checkable` and consumers
satisfy it structurally, so a member added to it turns every implementation we
never see from conforming into non-conforming, silently and at once.

Both sides of the comparison are folded, and with the same function — so this
rung's `normalizer` defaults to `default_normalizer` where every other rung here
defaults to no fold at all. The others hand a string to the index and the index
folds it; this one *is* the comparison, so an unfolded window scored against a
folded form would read a capital letter as a misspelling.

Your own `normalizer` must not **shorten** as its input grows. The probe's
bound is spent in folded characters and it stops widening at the first window
over it, so a fold that deleted more from a longer slice than from a shorter
one could stop the probe early. Every character-wise fold satisfies this —
anything built from `strip`, `casefold`, `lower` or `replace`, including one
that deletes separators outright.

**And the source's flavour is checked, not just its shape.** Both catalogues
spell the member `surface_forms`, and `isinstance` against a runtime-checkable
protocol compares member *names* — so a structural check alone accepts either
one, and a synchronous source reaching `AsyncLexicalSignal` builds a rung whose
every query raises from inside the scan. The flavour is asked separately and
the refusal says which one it found:

<!-- worked-flavour -->

```python
from dataknobs_common.entity_resolution import AsyncLexicalSignal

try:
    AsyncLexicalSignal(breeds)
except ValidationError as refused:
    print(refused)
```

```
MappingEntitySource publishes surface_forms() as a synchronous member, and this rung needs the asynchronous one. The two protocols spell the member identically, so a structural check cannot tell them apart -- pass the asynchronous source, or the rung of the other flavour.
```

That matters because a configuration is where the mistake gets made:
`entities:` is resolved before either registry sees it, so nothing between the
two flavours' factories would otherwise catch it.

## Where a match sat

Evidence carries a **span**: half-open offsets into `result.query`, or `None`
where the rung cannot locate one.

```python
assert result.explain("beagle")[0].span == (0, 7)      # into `result.query`

padded = resolver.resolve("  Beagles  ", k=5)
assert padded.explain("beagle")[0].span == (2, 9)      # the fold strips
assert padded.explain("beagle")[0].matched_text == "Beagles"
```

`(0, len(query))` would be the easy answer for that second one and is not the
honest one: the fold strips, so two of those characters took no part in the
match. `matched_text` is the slice the span points at, so the two agree by
construction — and it is the caller's own casing rather than the index's.

A rung that compares the whole query has one span to report and the base class
reports it; a rung that *locates* forms inside the query reports one per place
it found something. Those are the same rung as far as a caller is concerned,
which is the point: a caller who already knows the phrase and one who hands
over a whole sentence take the same path.

`span` is `None` where there is no position to report — a cosine neighbour over
an embedded utterance has none in the utterance, and a candidate that inherited
its match rather than making one has none either.

## What the query left unaccounted for

Coverage is that positional account, rolled up over every candidate.

```python
miss = resolver.resolve("wombat", k=5)

assert not miss.candidates                             # a miss is an empty tuple
assert miss.coverage.unmatched == ((0, 6),)
assert miss.unmatched_text() == ("wombat",)
```

There is no outcome enum: `bool(candidates)` is the miss. Coverage is a report
rather than a verdict — an absence is not a falsehood — and it is how a
vocabulary gets maintained: the phrases users ask about and the vocabulary does
not carry are the next entries somebody should add.

`matched` is the union of the evidence spans, merged and ordered; `unmatched`
is the residue, each interval trimmed of the whitespace that bounded it.
`matched_text()` and `unmatched_text()` slice them back out of `query` for you.
Spans rather than strings because the text is derivable from the position and
the position is not derivable from the text — a phrase occurring twice has one
string and two places, and a report naming the string cannot say which.

**Evidence that located nothing covers nothing**, and that is the reading
rather than a gap in it. A resolution whose only hits came from a vector rung
comes back with `matched` empty and the whole query `unmatched`: a
neighbourhood guess is being offered and no declared form was found in the
text. That is the single strongest line a consumer maintaining a vocabulary can
act on, and one the older all-or-nothing rule could not state — it reported a
vector-only hit and an exact one the same way, because both had produced a
candidate.

Its third field, `beyond_authority`, holds entity **ids** rather than offsets,
and belongs to the scope path — see [Scoping a
resolution](#scoping-a-resolution).

## Scoping a resolution

`within` takes a set id, a collection of them, or a mapping from a scope axis
to either. On this path the sets are the entity types the source declares —
which is what the default axis is named for. `ENTITY_TYPE_KEY` holds
`Entity.type`, so `Breed` and `Species` belong there and a taxonomy's own id
does not: `onto.taxonomies` is a different id space, and a value from it under
this axis matches nothing.

```python
assert [c.entity_id for c in resolver.resolve("beagle", within="Breed").candidates] == [
    "beagle"
]
assert resolver.resolve("beagle", within="Species").candidates == ()
```

The mapping form is `AND`-ed **across keys** and unioned **within one**, so a
consumer whose scope is a kind *and* a state can say so. The bare forms are
sugar for the default axis and mean exactly what they always meant.

**The axis name is load-bearing.** A candidate is admitted on an axis only if
it declares something there, so a candidate silent on a named axis is excluded
rather than passing every filter on it.

That reading is right for an entity and wrong for a typo, which is why a scope
naming an axis the source does not publish is **refused** rather than answered
with nothing: an empty result is what a correctly spelled scope over a
vocabulary holding none of that type returns too, and nothing in it says which
happened.

```python
from dataknobs_common.entity_resolution import ENTITY_TYPE_KEY, within_axis_names

assert within_axis_names(onto.entities) == frozenset({ENTITY_TYPE_KEY})
assert resolver.resolve("beagle", within={ENTITY_TYPE_KEY: "Breed"}).candidates

resolver.resolve("beagle", within={"habitat": "forest"})   # ValidationError
```

An ordinary entity source publishes exactly one axis. A source that knows more
than a type about its entities publishes more by satisfying
`MembershipOracle` — `memberships()` says what one entity *is*, and `axes()`
says which names may be asked, which is what makes the second axis both
answerable and spell-checkable:

```python
class Habitats(MappingEntitySource):
    def memberships(self, entity):
        return {ENTITY_TYPE_KEY: entity.type, "habitat": entity.metadata["habitat"]}

    def axes(self):
        return frozenset({ENTITY_TYPE_KEY, "habitat"})
```

**The cascade decides this, not the rungs.** It holds the entity source it was
built with, and that source is the single authority: a rung answering with an
entity its own backing calls a `Breed` is overruled if the cascade's source
disagrees. `MatchSignal` does not require two rungs to share a backing, so
under a rung-side drop two rungs could disagree and nothing would notice.

Not sharing that backing also means a rung can answer with an id the authority
does not carry at all — an index gone stale against the vocabulary, which is an
operational fact rather than a bug. Such a candidate is **dropped**, because
nothing can show it is inside the scope; but the drop is **reported**, because
an empty result whose `unmatched` spans the query is also what a correctly
spelled scope over a vocabulary lacking the phrase returns, and only one of
those is the caller's to fix.

```python
from dataknobs_common.entity_resolution import CascadingResolver, ExactNormalizedSignal
from dataknobs_common.ontology import Entity, MappingEntitySource

# A rung reading an index that still carries an entity the vocabulary does not.
stale = MappingEntitySource({"quokka": Entity(id="quokka", type="Breed", name="Quokka")})
across = CascadingResolver([ExactNormalizedSignal(stale)], onto.entities)

scoped = across.resolve("quokka", k=5, within="Breed")
assert scoped.candidates == ()                          # dropped: nothing can check it
assert scoped.unmatched_text() == ("quokka",)           # true of a plain miss too
assert scoped.coverage.beyond_authority == ("quokka",)  # only this says which id

assert across.resolve("quokka", k=5).coverage.beyond_authority == ()
```

That last line is the rule the field turns on: **empty wherever nothing was
scoped.** An unscoped resolution asks the authority nothing, so there is no
check to have failed, and reporting one that was never made is the same kind of
claim the field exists to refuse. A candidate the source *does* carry and the
scope rejects stays out of it too — that drop was made, not skipped.

Refusing instead was considered and rejected. An unknown *axis* is refused
because it has no innocent reading; an unknown *id* has one, and refusing would
fail every scoped query until somebody rebuilt an index.

A rung that declares `narrows()` is still *offered* the scope, and may use it
to return `k` candidates already inside it — otherwise a rung asked for `k`
hands back `k` unscoped candidates that the cascade then thins, and the query
comes back short.

That narrowing is an optimisation, but **only in one direction**. A rung may
**over-admit freely and must never under-admit** — it is a superset filter, not
a second reading of the scope. The cascade rules on what a rung *produced*, so
it can only remove: too much is corrected, too little is lost. A candidate the
rung withholds is one nothing downstream can recover, so when in doubt a rung
should return it and let the cascade decide.

The general form is worth stating on its own, because it is not specific to
this pair: **a downstream filter cannot recover what an upstream one removed,
so in any two-stage filter the upstream stage must be a superset filter.** The
asymmetry survives even when both stages call the same functions — and here
they do: rung and cascade both apply `within_admits` over
`within_memberships`, so there is one reading of a scope and one projection of
what an entity is, rather than copies that agree today.

## Both flavours, and one core

`AsyncCascadingResolver` is the twin for rungs that reach for data. It is not a
second implementation: `CascadeState`, `merge_rung` and `finish` are shared,
and the only difference is the `await` around each rung.

```python
from dataknobs_common.entity_resolution import (
    AsyncAliasSignal,
    AsyncCascadingResolver,
    AsyncExactNormalizedSignal,
)
from dataknobs_common.ontology import async_build_resolver, async_load_ontology

onto = await async_load_ontology(Path("mammals.yaml"))
resolver = await async_build_resolver(Path("mammals.yaml"), onto)
result = await resolver.resolve("beagles", k=5)
```

Where a synchronous call site has no choice — the rungs it needs are
asynchronous and the caller cannot await — `BridgedEntityResolver` forwards
across a private event loop on a daemon thread. It is callable from inside a
running loop without deadlocking, but it still *blocks*: the calling thread
waits for the whole cascade. From async code, await the resolver directly.

<!-- worked-bridge -->

```python
import asyncio
from pathlib import Path

from dataknobs_common.entity_resolution import AsyncEntityResolver, BridgedEntityResolver
from dataknobs_common.ontology import async_build_resolver, async_load_ontology


async def cascade() -> AsyncEntityResolver:
    onto = await async_load_ontology(Path("mammals.yaml"))
    return await async_build_resolver(Path("mammals.yaml"), onto)


async_resolver = asyncio.run(cascade())

# Synchronous from here down, which is the point: no `await`, no loop of your
# own, and no rewriting the cascade you already have.
with BridgedEntityResolver(async_resolver) as bridged:
    result = bridged.resolve("beagles", k=5)

assert [c.entity_id for c in result.candidates] == ["beagle"]
assert [e.signal for e in result.explain("beagle")] == ["exact", "alias", "scan"]
```

It costs one daemon thread for the object's lifetime, so build one and keep it
rather than one per call.

That block runs as written too, against the same `mammals.yaml`, and the same
workspace test holds it character-identical to the copy it executes. The three
signals on the one candidate are the default composition the door builds for a
document that declares no `resolver:` section — exact, then alias, then the
scan.

**The rungs it wraps need not be this package's.** `BridgedEntityResolver`
takes any `AsyncEntityResolver`, so a cascade whose rungs read a vector store
or an authority stack bridges the same way. What the bridge does not change is
the cost: whatever the rungs reach for, the calling thread waits for all of it.

## Writing your own rung

**Start from `DeclaredSignal`.** Set `key`, implement `_hits`, and everything
else arrives with it: the constructor, `name`, `narrows()`, the batch loop, and
the rung-side narrowing — the last of these already a superset filter, which is
the one part of a rung that is easy to get wrong in the direction nothing
downstream can recover.

```python
from dataknobs_common.entity_resolution import DeclaredSignal, signal_backends


class MyRung(DeclaredSignal):
    key = "my_rung"

    def _hits(self, query: str) -> frozenset[str]:
        # `query` is already folded by this rung's normalizer.
        # Return everything that matches; ordering, the cut to `k` and the
        # scope are decided above you.
        return self._entities.by_surface_form(query)


signal_backends.register("my_rung", MyRung)
```

`AsyncDeclaredSignal` is the same for a rung that reaches for data — one
`async def _hits`, everything else shared.

**To measure rather than look up, set `kind` and `scoring` too.** They are class
attributes on both bases, defaulting to `DECLARED`/`DECLARED` — which is what
every rung above means — and a rung proposing an entity the query did not spell
sets `EvidenceKind.INFERRED` and `Scoring.NATIVE`, then carries its number on
each `FormHit`. `LexicalSignal` is the one in this package that does, and
[When the query does not spell the form](#when-the-query-does-not-spell-the-form)
is what it looks like from the outside.

**To scan rather than compare, override `_located` instead.** `_hits` answers
with a `frozenset[str]`, which has nowhere to put an offset; `_located` answers
with `FormHit`s — an id and where its form sat — in the order the rung wants
them proposed.

**If a near-spelling match is what you want, it ships too.** `LexicalSignal`,
in the section linked above — and note that its probe is bounded by
*characters* rather than by tokens, which the next paragraph is the reason for.

**If an n-gram scan is what you want, it ships — construct it.** That is
`ScanningSignal`, at the top of this page: every span of consecutive tokens
looked up as the slice it covers, longest first, at most twenty-one lookups for
a six-token utterance — and no window wider than the longest form the
vocabulary declares, since no declared form could fill one, so usually fewer.
Writing it again is the one thing this section should not talk you into.

That bound holds because `default_normalizer` keeps token boundaries where it
finds them. A vocabulary loaded with a fold that *deletes* them — one
squashing `C.D.C.` to `cdc` by dropping every non-alphanumeric — has no such
bound: a one-token key is then reachable from a window of any width, so the
source reports that it cannot bound and the scan enumerates in full rather
than quietly stopping short of a declared form. `max_window` is where a caller
who knows their own queries puts a number back.

What is worth writing yourself is a rung that asks the index a question the
shipped one cannot, and there is a concrete one. A probe is a slice *between*
token boundaries, so a declared form whose first or last character is not
alphanumeric — `(beagle)`, `C.D.C.` — is never probed at all. A vocabulary
carrying forms like those wants a different boundary:

<!-- worked-rung -->
```python
import re

from dataknobs_common.entity_resolution import DeclaredSignal, FormHit


class PunctuatedFormRung(DeclaredSignal):
    key = "punctuated"

    def _located(self, query: str) -> list[FormHit]:
        found: list[FormHit] = []
        for chunk in re.finditer(r"\S+", query):  # "(beagle)" is one chunk
            hits = self._entities.by_surface_form(self._fold(chunk.group()))
            found += [FormHit(entity_id=i, span=chunk.span()) for i in self._order(hits)]
        return found
```

Five lines, and every one of them is the hook rather than the policy. Note what
the rung is responsible for: `self._order(hits)` because a set has no stable
order and the rung is the layer that decides what the order means,
`self._fold` because the slice is what reaches the index, and a span that points
at **the query** — `chunk.span()` rather than an offset into anything the rung
computed for itself. Order matters where the score does not: a declared hit is
`1.0` by fiat, so the only order a caller can read is the one the rung
publishes — and `_order` is the hook that publishes it on both paths, the
`_hits` one and this one, with alphabetical as the stable default that means
nothing more than that.

That block is executed: `tests/worked_punctuated_rung.py` is the same text, run
against a vocabulary declaring `(beagle)`, `C.D.C.` and `K-9`, so the three
forms named below are measured rather than asserted here.

**And note what it gives up.** A chunk is whitespace-delimited, so this rung
reaches a form whose edges are punctuated and reaches *no multi-word form at
all* — `golden retriever` and `domestic dog` are two chunks each, and nothing
here ever joins two. That is the trade the boundary buys, not an oversight:
the same test measures it, so a reader copying this block for a vocabulary
that carries both kinds of form knows before they run it. A vocabulary needing
both composes this rung with `ScanningSignal` rather than choosing between
them, which is what the section below is about.

`token_spans` is the boundary policy the shipped scan uses, and it is a
different question from the fold: `default_normalizer` strips and case-folds,
which is what a whole-string lookup wants and is not what stops a scan finding
`beagle` inside `unbeagleable`. Whichever boundary a rung picks, probe the
**slice** rather than a join, so no offset ever points at a string the text does
not contain.

Overlapping forms are all returned. `"my golden retriever has been limping"`
hits `golden_retriever` at `(3, 19)` and `retriever` at `(10, 19)` — the same
two the executed block at the top of this page reports; choosing one is a
verdict, and the offsets make the containment visible so the consumer can make
it instead.

`k` counts entities rather than places: one entity named twice is one candidate
carrying two pieces of evidence, which is what the cascade does when two rungs
produce the same id.

**Compose rather than choose.** `ScanningSignal` and `ExactNormalizedSignal`
read the same index and neither subsumes the other: the scan finds a declared
form sitting inside a sentence, which a whole-string comparison misses, and the
whole-string rung finds a form whose own edges are punctuated, which no scan
probes. A cascade carrying both asks the index twice and pays two dictionary
lookups for it — which is what the default composition does, so a document
that configures nothing already asks it both ways.

The bare `MatchSignal` protocol stays the escape hatch, for a rung the base
cannot serve: one whose backing is not a dictionary lookup, or one that already
has a superclass. Satisfying it means a `name`, a `narrows()` and the two
`candidates` members — and writing the batch loop and the narrowing yourself,
correctly, including that `narrows()` promise. Either way, register it under
the key consumers will write as `kind:`; the evidence it produces carries that
same key, so a caller reading `evidence.signal` can correlate a hit back to
what they configured.

**What it does not mean is assembling the candidates yourself.** Find your
hits; `declared_candidates` turns them into what the cascade expects:

```python
from dataknobs_common.entity_resolution import declared_candidates


class MyRung:
    name = "my_rung"

    def narrows(self) -> bool:
        return False

    def candidates(self, query, k, *, filter=None):
        return declared_candidates(self._located(query), k, signal=self.name, query=query)
```

It groups hits by entity so `k` counts **entities** rather than places, keeps
the order you returned them in, and slices `matched_text` out of the query so
the text and the span agree by construction rather than because you computed
both. A rung that narrows passes the ids its filter left standing as
`admitted=`; one that does not — like the example above — leaves it out.

Each piece of evidence is `DECLARED` and scored `1.0` unless you say otherwise,
which is what a rung over declared forms means. A rung that **measures** passes
`kind=` and `scoring=`, and puts its number on each `FormHit` — the hit's score
where it has one, `1.0` where it does not, and the candidate's own score is the
best of its hits'.

This is shared with `DeclaredSignal` rather than parallel to it: the base runs
the same function, so a rung written against the protocol produces the same
evidence shape as a shipped one instead of a copy that agrees until one of them
changes.

Registering also clears an *unavailable* mark. A rung that exists in only one
flavour is declared in the other with a reason, so asking the synchronous
registry for it says why rather than `unknown key`:

```python
assert signal_backends.unavailable_reason("semantic") == (
    "SemanticSignal has no synchronous form"
)
```

`load_ontology` reads that fact and composes the refusal, naming the rung, its
kind and the loader that *can* build it. The refusal is computed from the
declared kind, so it holds with nothing constructed — and supplying the rung is
what makes the door accept it.

A rung that ships in **another distribution** is declared the same way, for a
different reason. `kind: "authority"` reads an authority stack and lives in
`dataknobs-xization`, which this package does not depend on and must not, so
both registries carry the key with a sentence naming the distribution and the
module that registers it:

```python
assert signal_backends.get_metadata("authority")["requires_install"] == (
    "pip install dataknobs-xization"
)
```

It matches text a vocabulary *describes* — a pattern — as well as text it
enumerates, and it keeps less overlap than `ScanningSignal` does. Both
directions are written up at
<https://kbs-labs.github.io/dataknobs/packages/xization/entity-resolution/>.

## Where this package sits

`dataknobs_common.entity_resolution` imports nothing from
`dataknobs_common.ontology`, and nothing from `dataknobs_data`. The rungs take
an entity source as a *protocol*; the value types they construct live in this
package; the normalizer they fold with lives in `dataknobs_common.text`.

That is a property worth keeping rather than a coincidence. An edge back would
close a cycle through `ontology/__init__`, which imports the loader, which
builds a cascade — and it would fail on import **order**, so a suite that
happens to import one side first would stay green while the other was broken.

It is a property of the **module graph**, not of your process. `dataknobs_common`
publishes the vocabulary on its own door, so importing anything from the package
imports `dataknobs_common.ontology` too. What the sentence above buys is that
this family's modules can be read, moved or depended on without the vocabulary —
not that a running interpreter holding a rung has never loaded it.
