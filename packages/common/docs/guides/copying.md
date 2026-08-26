# Copying Structures

`copy_structure` sits between `dict(value)` and `copy.deepcopy(value)`:
it rebuilds nested dicts and lists, and passes every other value through
unchanged.

```python
from dataknobs_common import copy_structure

handed_out = copy_structure(source)
```

## Choosing between the three

| | Isolates | Cost | Use when |
|---|---|---|---|
| `dict(v)` / `list(v)` | the top level only | ~0.1 µs | the values are immutable, or aliasing them is the point |
| **`copy_structure(v)`** | every dict and list | ~6 µs | you are handing out a structure a caller may adjust |
| `copy.deepcopy(v)` | everything, including leaf objects | ~14 µs | you genuinely need independent copies of the objects too |

The figures are one wizard stage config on one machine, and the absolute
values move with how deep the structure is — a flatter one measures well
under half of these. What holds across shapes is the ordering, and that
`deepcopy` costs roughly twice what `copy_structure` does.

## Why not a shallow copy

A shallow copy isolates one level. Every nested container in the result
is still the source's own object, so a consumer adjusting a nested
section writes back into a structure that outlives the hand-out:

```python
source = {"schema": {"type": "object"}}

shallow = dict(source)
shallow["schema"]["type"] = "array"
assert source["schema"]["type"] == "array"        # reached the source

structural = copy_structure(source)
structural["schema"]["type"] = "array"
assert source["schema"]["type"] == "object"       # did not
```

This is the gap a "returns a copy" docstring most often turns out to
have. It is worth stating which of the two a method means.

## Why not `deepcopy`

A structure assembled in Python may hold a live object — a connection
pool, a prebuilt provider, a lock. `deepcopy` duplicates it, silently
giving its owner a second one; and a value that cannot be pickled raises
out of what was meant to be an ordinary read.

```python
import threading

lock = threading.Lock()
source = {"resource": {"lock": lock}}

handed_out = copy_structure(source)
assert handed_out["resource"] is not source["resource"]   # container rebuilt
assert handed_out["resource"]["lock"] is lock             # leaf shared
```

So the boundary is: **containers are rebuilt, leaves are shared.**
Mutating the returned structure never reaches the source; mutating a leaf
object reachable from it still does, because that object was never
copied. When the leaves are immutable — the ordinary case for
configuration loaded from YAML or JSON — the result is full isolation at
a fraction of the price.

Only `dict` and `list` are rebuilt. A `tuple` is immutable, so its
identity is not a hazard — but a dict *inside* one is not reached.

## The memo

`copy_structure` accepts an optional `seen` memo, and keeps one
internally when you omit it. It does two things:

- **A structure that refers to itself terminates** instead of raising
  `RecursionError`.
- **A subtree shared between two keys stays shared.** Sharing on the way
  in is sharing on the way out.

Pass one memo across several calls when you are assembling a *single*
hand-out from several values, so the result's sharing reflects the
source's:

```python
seen: dict[int, Any] = {}
config = copy_structure(resources[name], seen)
for key, value in defaults.items():
    config.setdefault(key, copy_structure(value, seen))
```

Omit it when copying one value. Two separate calls without a shared memo
produce independently copied subtrees, which is usually what you want.

The memo also holds a reference to every source it has seen — as
`copy.deepcopy` does through `_keep_alive`, and for the same reason. It
is keyed on `id()`, and an id is only unique among *live* objects.
Without that reference a source freed between two calls sharing one memo
can have its id reused by the next, and the memo then answers for an
unrelated object. Two successive calls passing a temporary dict literal
are enough to reproduce it.
