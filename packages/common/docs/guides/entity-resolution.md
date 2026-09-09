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

Imported by module path, like the rest of this family, and on no package door
yet.

```python
from dataknobs_common.entity_resolution import (
    AliasSignal,
    CascadingResolver,
    EvidenceKind,
    ExactNormalizedSignal,
)
```

## The door, and what it builds

```python
from pathlib import Path

from dataknobs_common.ontology import load_ontology
from dataknobs_common.ontology.loader import build_resolver

onto = load_ontology(Path("mammals.yaml"))
resolver = build_resolver(Path("mammals.yaml"), onto)   # no store, no loop

result = resolver.resolve("beagles", k=5)
assert result.candidates[0].entity_id == "beagle"
```

Two functions rather than one, because an `Ontology` is a **value**: it owns no
lifecycle and has nowhere to put a runtime. A caller wanting both makes two
calls and holds two objects.

A document that declares no `resolver:` section gets the declared order over
what the value can serve — `ExactNormalizedSignal`, then `AliasSignal`. A
document that declares `rungs: []` gets a cascade that misses everything.
Those are different answers on purpose: silence is silence, and an empty list
is a composition somebody wrote.

## Every candidate carries its evidence

*Why did this win?* is answerable without re-running the query.

```python
top = result.candidates[0]

assert [e.signal for e in top.evidence] == ["exact", "alias"]
assert top.evidence[0].kind is EvidenceKind.DECLARED
assert top.declared                                   # some evidence is DECLARED

assert [e.signal for e in result.explain("beagle")] == ["exact", "alias"]
```

`beagles` is `beagle`'s declared alias, so **two** rungs produce it: the exact
rung through the folded form index, the alias rung through the alias index.
The candidate appears **once**, positioned by the earlier rung, carrying two
pieces of evidence — a duplicate appends evidence and moves nothing.

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

## What the query left unaccounted for

```python
miss = resolver.resolve("wombat", k=5)

assert not miss.candidates                             # a miss is an empty tuple
assert miss.coverage.unmatched == ("wombat",)
```

There is no outcome enum: `bool(candidates)` is the miss. Coverage is a report
rather than a verdict — an absence is not a falsehood — and it is how a
vocabulary gets maintained: the phrases users ask about and the vocabulary does
not carry are the next entries somebody should add.

## Scoping a resolution

`within` takes a set id, a collection of them, or a mapping from a scope axis
to either. On this path the sets are the entity types the source declares.

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
it declares something there, so a scope naming an axis the source does not
publish admits nothing — rather than the forgiving reading, where a candidate
silent on an axis passes every filter on it. What a source publishes is
`describe().declares`, and an entity source publishes one axis:

```python
from dataknobs_common.entity_resolution import TAXONOMY_ID_KEY

assert resolver.resolve("beagle", within={TAXONOMY_ID_KEY: "Breed"}).candidates
assert not resolver.resolve("beagle", within={TAXONOMY_ID_KEY: "Breed",
                                              "habitat": "forest"}).candidates
```

**The cascade decides this, not the rungs.** It holds the entity source it was
built with, and that source is the single authority: a rung answering with an
entity its own backing calls a `Breed` is overruled if the cascade's source
disagrees. `MatchSignal` does not require two rungs to share a backing, so
under a rung-side drop two rungs could disagree and nothing would notice.

A rung that declares `narrows()` is still *offered* the scope, and may use it
to return `k` candidates already inside it — otherwise a rung asked for `k`
hands back `k` unscoped candidates that the cascade then thins, and the query
comes back short. That is an optimisation and is allowed to be wrong. Both it
and the cascade apply the same published function, `within_admits`, so there
is one reading of a scope rather than two that agree today.

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
from dataknobs_common.ontology import async_load_ontology
from dataknobs_common.ontology.loader import async_build_resolver

onto = await async_load_ontology(Path("mammals.yaml"))
resolver = await async_build_resolver(Path("mammals.yaml"), onto)
result = await resolver.resolve("beagles", k=5)
```

Where a synchronous call site has no choice — the rungs it needs are
asynchronous and the caller cannot await — `BridgedEntityResolver` forwards
across a private event loop on a daemon thread. It is callable from inside a
running loop without deadlocking, but it still *blocks*: the calling thread
waits for the whole cascade. From async code, await the resolver directly.

```python
from dataknobs_common.entity_resolution import BridgedEntityResolver

with BridgedEntityResolver(async_resolver) as bridged:
    result = bridged.resolve("beagles", k=5)
```

It costs one daemon thread for the object's lifetime, so build one and keep it
rather than one per call.

## Writing your own rung

A rung is anything satisfying `MatchSignal`: a `name`, a `narrows()`, and the
two `candidates` members. Register it under the key consumers will write as
`kind:`, and the evidence it produces carries that same key — so a caller
reading `evidence.signal` can correlate a hit back to what they configured.

```python
from dataknobs_common.entity_resolution import signal_backends

signal_backends.register("my_rung", MyRung)
```

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

## Where this package sits

`dataknobs_common.entity_resolution` imports nothing from
`dataknobs_common.ontology`, and nothing from `dataknobs_data`. The rungs take
an entity source as a *protocol*; the value types they construct live in this
package; the normalizer they fold with lives in `dataknobs_common.text`.

That is a property worth keeping rather than a coincidence. An edge back would
close a cycle through `ontology/__init__`, which imports the loader, which
builds a cascade — and it would fail on import **order**, so a suite that
happens to import one side first would stay green while the other was broken.
