# Resolving Against an Authority Stack

`dataknobs_common` ships three resolution rungs that read an entity source and
nothing else. They match text your vocabulary **enumerates**: a name, an alias,
an id. This package ships a fourth that matches text your vocabulary
**describes** as well — a chip number, an account code, a date — by putting
`dataknobs_xization.authorities.Authority` behind the same protocol.

```python
from dataknobs_xization import AuthoritySignal
```

Importing `dataknobs_xization` also registers it under `kind: "authority"` in
both of `dataknobs_common`'s rung registries, so a configured cascade reaches
it without naming the class.

## What it buys, and what it costs

It is an upgrade in one direction and a downgrade in the other, and both are
worth knowing before you choose it.

| | `ScanningSignal` (`dataknobs-common`) | `AuthoritySignal` (this package) |
|---|---|---|
| a form the vocabulary **lists** | found | found |
| a form the vocabulary **describes** — a pattern | never | found |
| two overlapping forms, one authority per axis | **both** returned | the **containing** form only |
| two overlapping forms, one authority per form | both | both |
| honours a `filter` | yes | **no** — `narrows()` is `False` |
| dependencies | none | pandas, and this package |

The overlap row is the one that surprises people, so it has a section of its
own below. Neither answer is wrong; they are answers to different questions.

## The worked example

A vocabulary with two entities whose surface forms overlap, plus a pattern no
enumeration could carry:

<!-- worked-authority-rung -->
```python
import re

import pandas as pd

from dataknobs_xization import AuthoritySignal
from dataknobs_xization.authorities import AuthorityData, AuthoritiesBundle, RegexAuthority
from dataknobs_xization.lexicon import DataframeAuthority, LexicalExpander

# The dictionary arm: forms the vocabulary lists. The frame's **index** is the
# entity id -- see "The ids are yours" below.
breeds = DataframeAuthority(
    "breed",
    LexicalExpander(None, str.lower),
    AuthorityData(
        pd.DataFrame(
            {"breed": ["Golden Retriever", "Retriever"]},
            index=["golden_retriever", "retriever"],
        ),
        "breed",
    ),
)

# The regex arm: a form the vocabulary describes. `canonical_fn` computes the
# entity id from the matched text.
chips = RegexAuthority("chip", re.compile(r"\bK-\d{3}\b"), lambda text, _group: text)

rung = AuthoritySignal(AuthoritiesBundle("clinic", auths=[breeds, chips]))

found = rung.candidates("my golden retriever K-901 has been limping", k=5)

for candidate in found:
    for evidence in candidate.evidence:
        print(candidate.entity_id, evidence.span, repr(evidence.matched_text))
```

```
golden_retriever (3, 19) 'golden retriever'
K-901 (20, 25) 'K-901'
```

Every hit carries where it sat, as half-open offsets into the query you passed,
and `matched_text` is that slice rather than a second copy of it.

The chip resolves to `K-901` rather than to `chip`: `chip` is the *authority's*
name, and the entity id is what `canonical_fn` returned. For a pattern that is
usually the matched text itself — the vocabulary describes a shape, and each
text matching it is its own entity.

## The overlap difference, in one query

`"golden retriever"` contains `"retriever"`, and a vocabulary that declares
both is declaring two entities that can match the same words.

An `Authority` suppresses a form contained by one it has already matched:
`TokenAligner` marks a match's tokens consumed, and `re.finditer` is
non-overlapping. That suppression is **within** one authority, so it is a
consequence of how you loaded your vocabulary rather than of the rung:

<!-- worked-overlap -->
```python
QUERY = "my golden retriever has been limping"

# One authority per *axis* -- the ordinary way to load a vocabulary.
per_axis = AuthoritySignal(breeds)

# The same two forms, one authority each.
per_form = AuthoritySignal(
    AuthoritiesBundle(
        "breeds",
        auths=[
            RegexAuthority(
                entity_id,
                re.compile(re.escape(form), re.IGNORECASE),
                lambda _text, _group, found=entity_id: found,
            )
            for entity_id, form in [
                ("golden_retriever", "Golden Retriever"),
                ("retriever", "Retriever"),
            ]
        ],
    )
)

for label, configured in [("per axis", per_axis), ("per form", per_form)]:
    spans = [
        (candidate.entity_id, evidence.span)
        for candidate in configured.candidates(QUERY, k=5)
        for evidence in candidate.evidence
    ]
    print(label, spans)
```

```
per axis [('golden_retriever', (3, 19))]
per form [('golden_retriever', (3, 19)), ('retriever', (10, 19))]
```

`dataknobs_common`'s `ScanningSignal` returns both in either case, because it
probes every span of consecutive tokens independently and never consumes one.
If your consumers need to see that a longer form contained a shorter one — a
containment the offsets make visible rather than a verdict — either load one
authority per form, or compose both rungs and let the cascade merge them.

## The ids are yours

A rung answers in **your ontology's id space**, and this one reads whatever the
authority wrote as its value id. Nothing in the rung rewrites or coerces it.

| arm | the value id is |
|---|---|
| `DataframeAuthority` | the backing frame's **index** — `AuthorityData.lookup_values` reads `df.index` for one |
| `RegexAuthority` | whatever `canonical_fn(matched_text, group_name)` returns |

A frame left on its default `RangeIndex` therefore resolves every query to a
**row number**. That is a plausible-looking id and never the right one, and no
layer above can detect it — which is why it is stated here rather than guarded
in code.

## Reaching it by configuration

`dataknobs_common` declares the key and cannot implement it: `common` does not
depend on `xization`, and must not. So a process that has **not** imported this
module is answered with a sentence rather than with an unknown key —

> `rung kind 'authority' is not available here: AuthoritySignal ships in
> dataknobs-xization; import dataknobs_xization.entity_resolution to register
> it`

— and the registration that happens at this module's import clears that mark.
This page has already imported it, so here the registry reports the other side
of the same fact:

<!-- worked-registry -->
```python
from dataknobs_common.entity_resolution import signal_backends

print(signal_backends.is_registered("authority"))
print(signal_backends.unavailable_reason("authority"))
print(signal_backends.get_metadata("authority")["requires_install"])
```

```
True
None
pip install dataknobs-xization
```

`requires_install` outlives the mark deliberately: a reader asking what a rung
costs to obtain gets the same answer whether or not it happens to be installed.
Building one through the registry takes the stack under the `authorities` key:

<!-- worked-factory -->
```python
built = signal_backends.create("authority", {"authorities": breeds})
print(type(built).__name__, built.name, built.narrows())
```

```
AuthoritySignal authority False
```

`narrows()` is `False` because an authority stack holds no index of which types
an entity belongs to, so a scope handed to it could only be ignored or guessed
at. The cascade rules on this rung's candidates itself.

## Both flavours

`AsyncAuthoritySignal` is the same rung for an asynchronous cascade. It exists
for **composition, not concurrency**: an authority stack is regular expressions
and in-memory frames, so there is nothing to await and nothing worth offloading.
What it buys is that a cascade whose other rungs really do reach a store can
hold this one without a bridge.

## What this rung is not

It matches **declared** forms — listed or described — so its evidence is
`EvidenceKind.DECLARED` with a score of `1.0` by fiat, exactly like the rungs in
`dataknobs_common`. It finds no typos, and no near-spellings: a query has to
carry a form the vocabulary declares, or a form matching a pattern it declares.

A rung that proposes an entity the query did **not** spell is a different kind
with a real number underneath it — `EvidenceKind.INFERRED`, scored
`Scoring.NATIVE`. The key `lexical` is reserved for it and is deliberately
unregistered; asking for it today reports an unknown rung kind, which is the
honest answer.
