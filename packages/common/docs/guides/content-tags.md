# Content Tags

A vector row written from a vocabulary carries three keys saying what it is
about: which vocabulary, which axis, and which node or nodes.
`read_node_tags(metadata)` reads them back out of the row's own metadata
mapping, and `read_node_tags_many` does it for a hit set.

**The read holds no vocabulary**, which is the property the rest of this page
turns on. It is a pure function over a mapping: it opens nothing, awaits
nothing, and cannot tell a node that exists from one that does not. Judging an
id needs an ontology, and having one is your next line rather than this
function's — so the read works on rows out of a store this package has never
heard of, which is the only kind of store there is on the other side of a
retrieval.

**Nobody here writes the tags.** A row is tagged by whatever built the index,
and that is a component this package does not own. These keys are published so
that the writer and the reader spell them the same way, and the read is
published beside them so that a writer can see what will be done with what they
wrote.

## Where the names live

On the ontology door, with the vocabulary they describe:

```python
from dataknobs_common.ontology import (
    NODE_ID_KEY,
    ONTOLOGY_ID_KEY,
    TAXONOMY_ID_KEY,
    MalformedRow,
    NodeTag,
    TagReading,
    read_node_tags,
    read_node_tags_many,
)
```

`ALIAS_FORMS_KEY` and `read_alias_forms` are the fourth key of the same family
and its read, on the same door. It says which surface forms a row's entity is
known by, which is a different question from which node the row is placed on,
so it is not one of the three `read_node_tags` wants — but the rule for reading
it is the same one, and [it reads through the same
core](#the-fourth-key-reads-the-same-way). A fifth — the model that produced
the vector — lives in `dataknobs_data.vector.content`, because it is about an
embedder rather than about identity.

## The whole input

The same hand-edited vocabulary the [anchored view](anchored-view.md) uses:
two entity types, five entities, the edges between them, and a taxonomy naming
the relation those edges are made of.

<!-- worked-input -->

```yaml
# mammals.yaml
ontology:
  id: mammals
  version: "1.1"

  entity_types:
    - id: Species
      attributes:
        - {name: latin_name, type: string}
        - {name: lifespan_years, type: number, field_type: float}
    - id: Breed
      isa: Species                      # the TYPE lattice
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
    - {subject: dog, relation: isa, object: mammal}
    - {subject: retriever, relation: isa, object: dog}
    - {subject: golden_retriever, relation: isa, object: retriever}
    - {subject: beagle, relation: isa, object: dog}

  taxonomies:
    - {id: species, name: Species, relation: isa}
```

## The worked call site

Four hits, of which one is about two nodes, one touched no vocabulary at all,
one names an axis this file does not declare, and one was written wrong. That
is not a contrived set: it is what a page of results off a real corpus looks
like, and the read is only worth publishing if it says something useful about
each of them.

<!-- worked-call-site -->

```python
from pathlib import Path
from typing import Any

from dataknobs_common.exceptions import ValidationError
from dataknobs_common.ontology import (
    NODE_ID_KEY,
    ONTOLOGY_ID_KEY,
    TAXONOMY_ID_KEY,
    load_ontology,
    qualify,
    read_node_tags,
    read_node_tags_many,
)

# What the consumer's own retrieval handed back. This page wrote none of it and
# names no corpus: four rows out of somebody else's store, read back.
hits: list[dict[str, Any]] = [
    {
        "text": "Spay recovery in retrievers is usually uneventful.",
        "metadata": {
            ONTOLOGY_ID_KEY: "mammals",
            TAXONOMY_ID_KEY: "species",
            NODE_ID_KEY: ["golden_retriever", "beagle"],
            "model_name": "all-MiniLM-L6-v2",
        },
    },
    {
        "text": "Invoice 2291 paid in full.",
        "metadata": {"invoice_id": "2291"},
    },
    {
        "text": "Double coats need more grooming in spring.",
        "metadata": {
            ONTOLOGY_ID_KEY: "mammals",
            TAXONOMY_ID_KEY: "coat",  # an axis this vocabulary does not declare
            NODE_ID_KEY: ["double_coat"],
        },
    },
    {
        "text": "Vaccination schedule, revised.",
        "metadata": {
            ONTOLOGY_ID_KEY: "mammals",  # a writer got this one wrong:
            NODE_ID_KEY: ["beagle"],  # no dk_taxonomy_id
        },
    },
]

# (1) what is this row about? -- a pure read of what the row declares. No store,
#     no embedder, no event loop, and no vocabulary either.
tags = read_node_tags(hits[0]["metadata"])
[(t.ontology_id, t.taxonomy_id, t.node_id) for t in tags]
# [("mammals", "species", "golden_retriever"), ("mammals", "species", "beagle")]

read_node_tags(hits[1]["metadata"])  # () -- a row that touched no vocabulary.
# NOT an error, and NOT the same answer as a row whose tag is half written.

# (2) the whole hit set at once, because one row a writer got wrong must not
#     take the readable ones with it
reading = read_node_tags_many([hit["metadata"] for hit in hits])
len(reading.tags)  # 4 -- one entry per position, in the order asked
[len(row) for row in reading.tags]  # [2, 0, 1, 0] -- () for BOTH kinds of nothing
[bad.row for bad in reading.malformed]  # [3] -- which of them was refused, and:
reading.malformed[0].reason
# "a content row carrying 'dk_ontology_id', 'dk_node_id' declares no
#  'dk_taxonomy_id': a row carrying any of the three tag keys carries all three"

# The refusal is a member rather than the stance -- ask for it when you want it.
try:
    reading.require_readable()
except ValidationError as refusal:
    refused = str(refusal)  # names EVERY malformed position, once

# (3) name the vocabulary, and keep the tags it is about. The filter is yours: a
#     name off a row is not a name you typed.
onto = load_ontology(Path("mammals.yaml"))
mine = [
    t
    for row in reading.tags
    for t in row
    if t.ontology_id == onto.id and t.taxonomy_id in onto.taxonomies
]
[t.node_id for t in mine]  # ["golden_retriever", "beagle"] -- the coat row named
# an axis this vocabulary does not declare, and dropping it is a branch you took
# rather than a refusal you caught.

# (4) the tie-back-in: the id the row carries, judged by the vocabulary
axis = onto.taxonomy(mine[0].taxonomy_id)
here = axis.at(onto.localize(mine[0].qualified_id))
here.node  # "golden_retriever"
here.exists()  # True -- the tag names a node this vocabulary actually carries
placed = here.entity()  # -> Entity | None
assert placed is not None  # an id that misses is a typo, not a result
placed.name  # "Golden Retriever"
here.ancestors()  # retriever, dog, mammal

# (5) and the refusal, which is why the PAIR travels rather than a bare id
try:
    onto.localize(qualify("procedures", "spay"))
except ValidationError as refusal:
    foreign = str(refusal)  # names both ontologies
```

That block is executed as written by a workspace test, and the test asserts it
is character-identical to the fence above. If this page and the code ever
disagree, the suite goes red rather than the page going quietly wrong.

## The three keys travel together

```python
from dataknobs_common.ontology import NODE_ID_KEY, ONTOLOGY_ID_KEY, TAXONOMY_ID_KEY

assert (ONTOLOGY_ID_KEY, TAXONOMY_ID_KEY, NODE_ID_KEY) == (
    "dk_ontology_id",
    "dk_taxonomy_id",
    "dk_node_id",
)
```

**The vocabulary is written beside the node rather than composed into it**, so
that neither the writer nor a filter over the index has to parse anything. A
node id without its vocabulary addresses nothing: two ontologies may both carry
`beagle`, and a reader that cannot tell them apart resolves a hit to the wrong
entity while reporting success.

**The axis is required with the other two.** An entity id is
ontology-scoped and can be looked up with no axis at all, but what a reader
wants from a tag is a *cursor* — somewhere to walk up from — and a cursor needs
an axis to walk on.

**`dk_` because the keys are written into a store we do not own.** A consumer's
own metadata is in the same namespace, and a family spelled two ways is the
defect a published family exists to prevent. Each value is its constant's name
less `_KEY`, lower-cased, which is what makes the whole derivation checkable
rather than remembered.

## Nothing, and the other nothing

A row carrying none of the three keys answers `()`. A row carrying some of them
and not all raises:

```python
from dataknobs_common.exceptions import ValidationError
from dataknobs_common.ontology import (
    NODE_ID_KEY,
    ONTOLOGY_ID_KEY,
    TAXONOMY_ID_KEY,
    read_node_tags,
)

assert read_node_tags({"invoice_id": "2291"}) == ()

# A store that materialises what nothing wrote is saying the same thing.
assert read_node_tags({ONTOLOGY_ID_KEY: None, TAXONOMY_ID_KEY: None, NODE_ID_KEY: None}) == ()

try:
    read_node_tags({NODE_ID_KEY: ["beagle"]})
except ValidationError as refusal:
    assert "dk_ontology_id" in str(refusal)
    assert "dk_taxonomy_id" in str(refusal)
```

The two are different states and the difference is who can act on them. **Most
of a corpus touched no vocabulary** — invoices, transcripts, anything indexed
before a vocabulary existed — and that is an ordinary answer rather than a
fault. A caller who had to tell an exception from an answer on the common path
would wrap every call in a `try`, and a `try` around the common path catches
the uncommon one too.

**A half-written tag is a writer that got the contract wrong**, and silence
there lets one mistake scale to a whole corpus before anybody reads a row back.
So it refuses, and the refusal names the keys that are absent, because the
writer it is addressed to has not read this page.

**A `None` under a key is a key nothing wrote.** The argument is a row *as
your own store handed it back*, and only some stores omit — a relational or
columnar one hands back a column nobody wrote as a null. Reading that as a
half-written tag would refuse an untagged corpus in its entirety, which is the
opposite of the distinction above; so presence is read off the value. A null
under *one* key of three still refuses, and with the more useful of the two
messages: *declares no `dk_node_id`* rather than *`dk_node_id` is NoneType*.

## A bare string is one node

```python
from dataknobs_common.ontology import (
    NODE_ID_KEY,
    ONTOLOGY_ID_KEY,
    TAXONOMY_ID_KEY,
    read_node_tags,
)

row = {ONTOLOGY_ID_KEY: "mammals", TAXONOMY_ID_KEY: "species"}

assert [t.node_id for t in read_node_tags({**row, NODE_ID_KEY: "beagle"})] == ["beagle"]
assert read_node_tags({**row, NODE_ID_KEY: []}) == ()
```

The key is list-valued because a row may be about several nodes at once, and a
bare string satisfies *iterate it* — which is what makes the first line worth
asserting. Without it, `"beagle"` reads as six nodes named `b`, `e`, `a`, `g`,
`l`, `e`, and the symptom is an empty `ancestors()` three calls later, with
nothing having raised.

**`[]` is not a malformation.** All three keys are present and every entry is
well typed, vacuously; a row saying *this row is about no node of this axis* is
a statement a writer can make, and `()` is what it means.

## What the read refuses, and why here

Everything else that satisfies *iterate it* is refused, naming the key and the
type found:

```python
from dataknobs_common.exceptions import ValidationError
from dataknobs_common.ontology import (
    NODE_ID_KEY,
    ONTOLOGY_ID_KEY,
    TAXONOMY_ID_KEY,
    read_node_tags,
)

row = {ONTOLOGY_ID_KEY: "mammals", TAXONOMY_ID_KEY: "species"}

for written in (42, b"beagle", memoryview(b"beagle"), {"beagle": 1}, {"beagle"}):
    try:
        read_node_tags({**row, NODE_ID_KEY: written})
    except ValidationError as refusal:
        assert "dk_node_id" in str(refusal)
        assert type(written).__name__ in str(refusal)

try:
    read_node_tags({**row, NODE_ID_KEY: ["golden_retriever", None, "beagle"]})
except ValidationError as refusal:
    assert "1 of 3 entries" in str(refusal)

# The pair is joined with `:`, so a `:` in the ontology id would compose an id
# a DIFFERENT vocabulary accepts. This is the last frame that can see that.
try:
    read_node_tags({**row, ONTOLOGY_ID_KEY: "mammals:evil", NODE_ID_KEY: ["beagle"]})
except ValidationError as refusal:
    assert "dk_ontology_id" in str(refusal)
```

A `bytes` is a sequence of integers, so it reaches the six-tags-for-six-letters
failure through a second door — and **the exclusion is the buffer property, not
a list of types**. A list is the types somebody thought of: `memoryview` is what
a driver hands back for a binary column, it is registered on `Sequence` too, and
an empty one would otherwise answer `()` — *this row is about no node* — over a
value that says nothing of the kind. A `Mapping` reads as its keys and would
otherwise *succeed*, silently, on a shape nobody meant. A scalar that is not a
string would reach `TypeError` from the iteration rather than the class this
contract documents, so the type is tested before anything is iterated: a caller
holding the `Raises:` should not have to catch two classes for the commonest
serialization accident there is.

**A `:` in the ontology id is refused for the same reason one frame further
on.** `qualified_id` joins the pair with that character and `localize` splits on
the first one, so `mammals:evil` would compose `mammals:evil:beagle` — which the
vocabulary `mammals` *accepts*, localizing it to `evil:beagle`. The loader
already refuses a colon in an id it mints; a value off a foreign row is the one
door where that invariant is not already held.

**This is the last frame that can still see a type.** One call later the value
has been rendered into a string, one call after that it is a well-formed
qualified id, and at the cursor it is a miss indistinguishable from a stale
id — the bucket reserved for things that are merely *out of date*. A writer's
mistake arriving there is a writer's mistake nobody will ever attribute.

**One bad entry refuses the row whole**, and the message names every offending
position rather than the first. A row declaring three nodes of which one is
wrong would otherwise count as evidence for two, and a caller reading that has
been told something false about what the row is about. The extent of a tagger's
bug should be learnable in one read.

## The fourth key reads the same way

`dk_alias_forms` is not one of the three. It says which surface forms the
row's *entity* is known by rather than which node the row is placed on, so a
row carrying it and nothing else is an ordinary alias row and not a
half-written tag — there is no companion key for it to be absent against:

```python
from dataknobs_common.ontology import ALIAS_FORMS_KEY, read_alias_forms

assert read_alias_forms({ALIAS_FORMS_KEY: ["ACME", "ACME widget"]}) == (
    "ACME",
    "ACME widget",
)
assert read_alias_forms({ALIAS_FORMS_KEY: "ACME"}) == ("ACME",)
assert read_alias_forms({"invoice_id": "2291"}) == ()

# `aliases_key` on the writer is configurable, so the reader takes it too.
assert read_alias_forms({"forms": ["ACME"]}, key="forms") == ("ACME",)
```

**The second line is the whole reason this is published.** A bare `"ACME"` is
one form, not four one-character ones — the same rule the node key states, and
the one its reader had not applied: a consumer-written `"ACME"` became four
rows, all under the entity's id, leaving one holding `"E"`. The rule was
implemented here the whole time and implemented *privately*, so the reader that
had the bug was a reader writing the rule again. Both keys now read through one
function, which is what keeps them from drifting apart a second time.

**`key` is a parameter, not the constant.**
`EntitySourceIndexSource.aliases_key` is a field defaulting to
`ALIAS_FORMS_KEY`, so a consumer who configured their own key writes rows this
read still has to open. A reader hard-coded to the constant would be the
one-key-two-ends failure this family exists to prevent, arriving at the read
end.

## Over a corpus, the report is the answer

`read_node_tags_many` calls the row reader and re-implements nothing, so what
*malformed* means is defined in exactly one place. What it adds is a stance
about the batch:

```python
from dataknobs_common.ontology import NODE_ID_KEY, ONTOLOGY_ID_KEY, TAXONOMY_ID_KEY
from dataknobs_common.ontology import read_node_tags_many

rows = [
    {ONTOLOGY_ID_KEY: "mammals", TAXONOMY_ID_KEY: "species", NODE_ID_KEY: ["beagle"]},
    {"invoice_id": "2291"},
    {ONTOLOGY_ID_KEY: "mammals", NODE_ID_KEY: ["dog"]},
]

reading = read_node_tags_many(rows)

assert len(reading.tags) == len(rows)
assert reading.tags[1] == () and reading.tags[2] == ()
assert [bad.row for bad in reading.malformed] == [2]
assert "dk_taxonomy_id" in reading.malformed[0].reason
```

**One entry per position, in the order asked**, which is this package's rule for
every positional reply: a row with no answer contributes an empty sequence
rather than being dropped, because dropping one shifts every later row onto the
wrong position.

**Two fields, and the second is why the first is readable.** An untagged row
holds `()` and a refused row holds `()`, so a reading with one field would have
spelled both the same way — and *touched no vocabulary* and *was written wrong*
are opposite facts about a corpus. `malformed` is what tells them apart, and
`MalformedRow` carries the row reader's own message verbatim rather than a
re-composed one, so the batch caller is told no less than the per-row caller.

**A row with an empty node list also holds `()`, and `malformed` does not
separate that one.** It is the same fact as the untagged row — *this row is
about no node* — said explicitly rather than by omission, and neither is
something anybody can be told about. The field separates the answers that
differ in *who can fix them*. If you need the other distinction you have the
row: it is whether `dk_node_id` is in it.

**Raising would have been the quieter option, not the louder one.** Over ten
thousand rows with one bad row in the middle, a loop that lets the refusal out
returns nothing at all and names one position; the reading returns 9,999
readable rows and names the one. The refusal is still there when you want it:

```python
from dataknobs_common.exceptions import ValidationError
from dataknobs_common.ontology import ONTOLOGY_ID_KEY, NODE_ID_KEY, read_node_tags_many

reading = read_node_tags_many([{ONTOLOGY_ID_KEY: "mammals", NODE_ID_KEY: ["dog"]}])

try:
    reading.require_readable()
except ValidationError as refusal:
    assert "row 0" in str(refusal)

assert read_node_tags_many([]).tags == ()
assert read_node_tags_many([]).malformed == ()
```

`require_readable()` returns `tags` unchanged when nothing was refused, and
raises once naming every malformed position when something was. Which of the
two stances you took is then visible at the call site — whatever you hand the
tags to takes `reading.tags` to proceed over what is readable and
`reading.require_readable()` to refuse the batch — rather than being a policy
buried in a reader.

## Only the vocabulary can judge the id

A tag carries the pair. The door that judges an id takes one string, so the tag
composes it:

```python
from pathlib import Path

from dataknobs_common.exceptions import ValidationError
from dataknobs_common.ontology import NodeTag, load_ontology, qualify

onto = load_ontology(Path("mammals.yaml"))
tag = NodeTag(ontology_id="mammals", taxonomy_id="species", node_id="beagle")

assert tag.qualified_id == qualify(tag.ontology_id, tag.node_id) == "mammals:beagle"
assert onto.localize(tag.qualified_id) == "beagle"

try:
    onto.localize(qualify("procedures", "spay"))
except ValidationError as refusal:
    assert "procedures" in str(refusal) and "mammals" in str(refusal)
```

**`node_id` is a rendered id, not a key.** Over a vocabulary whose keys are
strings the distinction is invisible, because the rendering is the identity.
Over a vocabulary with a key of its own — a frozen dataclass, a tuple — the
read still answers strings and `localize` is what turns one back into a key, so
the two are different types in the same expression. That is why the tag hands
`localize` a string and takes back whatever `at()` wants.

**The refusal is the reason the pair travels rather than a bare node id.** An
id qualified by another vocabulary is refused *naming both*, which is a
diagnosis; a bare `spay` handed to this vocabulary is simply a node it does not
carry, which is the same answer as a typo.

## What a tag naming an unknown axis is

Two things can be unknown, they are one level apart, and neither is an error:

```python
from pathlib import Path

from dataknobs_common.exceptions import NotFoundError
from dataknobs_common.ontology import load_ontology

onto = load_ontology(Path("mammals.yaml"))
axis = onto.taxonomy("species")

assert axis.at("no_such_node").exists() is False

try:
    onto.taxonomy("coat")
except NotFoundError as refusal:
    assert "species" in str(refusal)
```

**A node the axis does not carry is reported, not refused** — the view answers
`exists() is False`, and *nothing below this node* and *this node is not here*
stay opposite answers.

**An axis the vocabulary does not declare is yours to filter.** `onto.taxonomy`
refuses a name it does not know, and it should: a caller who typed the name is
one edit away from the right one. But a name that came off a *row* is not a
name anybody typed — it is an ordinary drifted corpus, written when the
vocabulary had an axis it has since dropped, and there is nobody to tell. So
the call site filters before it asks, and the filter builds nothing:

```python
tags = [t for row in reading.tags for t in row]
mine = [t for t in tags if t.ontology_id == onto.id and t.taxonomy_id in onto.taxonomies]
```

`taxonomies` is a mapping the vocabulary already holds, so the membership test
costs a lookup. Making the accessor stop refusing would have been the other
way to get here, and it would have taken the refusal away from the caller who
does want it.

## What this is not

**Not a writer.** Nothing here puts a tag onto a row. The tagging belongs to
whatever builds your index, and these keys are a contract that an outside
writer can satisfy without reading this package's source — which they could not
do if the only way to write a tag were a member of ours.

**Not a resolver.** A `NodeTag` holds three strings and nothing from a
vocabulary: no entity, no view, no ontology. It is read from a mapping by a
function that has none of those, and a tag that resolved would need one.

**No asynchronous twin.** There is nothing to await: both functions are pure
over a mapping and a sequence of mappings. `qualify` and `split_qualified` have
no twins either, for the same reason.
