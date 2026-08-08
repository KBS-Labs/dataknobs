# Dotted-Path Resolution

Turning a dotted string from configuration into a live Python object —
`middleware: {class: "myapp.middleware.AuditLog"}`,
`on_enter: ["myapp.hooks:log_entry"]`, `custom_class: "myapp.transforms.ToId"`.

Four functions in `dataknobs_common.imports`, one policy.

## Why one implementation

The operation looks like two lines and contains four decisions: which separator
to accept, what to raise, whether to check the target's shape before or after
constructing it, and whether a typo is fatal at all. Written once it is a
policy. Written nine times it is nine policies that agree until they do not —
which is the state this module replaced. Three of those copies accepted only
`:`, four only `.`, and two either; they raised four different exception types
between them and two did not raise at all; two checked the target's shape
before constructing it and two after.

None of that was carelessness. Each copy was written because the existing one
was not findable — exported from no `__init__`, filed under a name that read as
something else, reached by every caller through a function-local deep import.
Hence this module's location, name, and export.

## The family

```python
from dataknobs_common import (
    resolve_dotted,            # any object, no shape check
    resolve_callable,          # + callable()
    resolve_class,             # + issubclass(base) — returns the CLASS
    resolve_optional_callable, # None-tolerant, names the config site
)
```

```python
fn = resolve_callable("myapp.hooks:on_save")

cls = resolve_class("myapp.middleware.AuditLog", Middleware)
instance = cls(**params)          # you construct it — see below

hook = resolve_optional_callable(
    config.get("dedup_key"), field_name="dedup_key", owner=source_name
)  # None when the key was omitted; raises when it is present and wrong
```

## Separator

`module.path:name` and `module.path.name` are both accepted, everywhere.
Prefer `:` in new configuration — it says which half is the module without the
reader having to know the package layout — but `.` is accepted because existing
configuration uses it.

Exactly **one** attribute lookup is performed. `module:Outer.Inner` is not
supported: no caller needs it, and reading a chain would make the `.` form
ambiguous in the way the `:` form exists to prevent.

## `resolve_class` returns the class

Not an instance, and not as a convenience — it is the point.

```python
cls = resolve_class(path, Middleware)   # shape checked here
obj = cls(**params)                      # you construct, after
```

A resolver returning an *instance* has to construct the target before it can
check it. So a mistyped path — one naming a class that exists but is the wrong
kind — runs that class's `__init__` before being rejected: arbitrary code, with
whatever side effects it has (a network read, a file open, a log write).
Returning the class makes validate-before-instantiate the only order this
function can express, so no call site has to remember it.

The callers instantiate differently anyway; each passes its own parameters.

### `base` is typed loosely on purpose

It is annotated `ClassConstraint[T]` — a callable returning `T` — and not the
obvious `type[T]`:

```python
def resolve_class(ref: str, base: ClassConstraint[_T]) -> type[_T]: ...
```

mypy reads `type[T]` as *instantiable* and rejects an abstract class or a
protocol in that position. That is not an edge case to work around: a
constraint nobody can subclass constrains nothing, so abstract bases and
protocols are the entire population of useful arguments. The obvious spelling
would reject every correct call and accept only meaningless ones, and it would
push a `# type: ignore[type-abstract]` into every call site — including yours.

What the looser spelling gives up is narrow. A factory *function* also
satisfies it and mypy will not object, at which point `issubclass` raises an
unwrapped `TypeError` naming the problem. That is where a bad constraint
already landed, so nothing new is being tolerated.

## Errors

Two sibling types, both `ConfigurationError` subclasses:

| Type | Means | Optional? |
|---|---|---|
| `DottedPathError` | the path did not resolve | may be swallowed by an `optional: true` config key |
| `DottedPathTypeError` | it resolved, and the target is the wrong shape | **never** |

`DottedPathError` carries a machine-readable `reason` — `malformed`,
`module_not_found`, `attribute_not_found`, `not_callable` — so a caller can
catch one type and still branch on which fault it was.

**They are siblings, not parent and child, and that asymmetry is load-bearing.**
The obvious lenient handler is:

```python
try:
    cls = resolve_class(spec["class"], expected_base)
except DottedPathError:
    if optional:
        return None          # a missing module is a deployment condition
    raise
# DottedPathTypeError propagates — no clause needed, and none possible
```

Were the shape error a subclass, that handler would swallow it too, and
`optional: true` would silently grow to cover a spec filed under the wrong
config key. As siblings, the handler cannot match a shape mismatch at all, so
the never-optional contract holds by construction rather than by remembering to
order the `except` clauses.

### Messages are bounded

The message names the reference and the *type* of the underlying failure; the
underlying exception's text travels on `__cause__`.

That is deliberate. Importing a module executes it, so the caught exception can
carry anything — an absolute filesystem path, a credential read at module
scope, a stack from three libraries down — and these errors are rendered to
HTTP clients by surfaces that map `ConfigurationError` to a status. The
diagnostic is relocated to the log, not lost.

## Import is execution

!!! warning "Trust boundary"

    Resolving a dotted path **runs the target module's top level** — every
    import it performs, every decorator it applies, every line at module scope
    — before this module has looked at the attribute, let alone checked its
    shape. There is no allow-list and no sandbox.

    A dotted path must come from the same trust domain as the application's own
    code: a config file, a deployment's policy bundle, a declaration a platform
    team authored. **Never build one from end-user input, a request body, or a
    per-tenant blob supplied by the tenant.**

`resolve_class` returning the class is a **partial** mitigation, and partial is
the accurate word: it means a wrong-shape target never runs its constructor,
which closes the narrow case of a misfiled spec triggering ctor side effects. It
does nothing about the module import that already happened, and nothing at all
about a correctly shaped class from a hostile path.

## Failure policy stays with the caller

These functions raise. Whether that is fatal is the caller's decision, and the
two shapes in this workspace are:

**A single reference** — raise, or swallow under an `optional: true` key:

```python
try:
    return resolve_callable(spec["function"])
except DottedPathError:
    if spec.get("optional"):
        logger.warning(...)
        return None
    raise
```

**A whole config block** — collect every fault and raise once:

```python
faults = []
for key, entry in block.items():
    try:
        register(resolve_callable(entry))
    except DottedPathError as exc:
        faults.append(f"{key}: {exc}")

if faults:
    raise ConfigurationError(
        "Invalid hook configuration:\n"
        + "\n".join(f"  - {fault}" for fault in faults)
    )
```

An author with three typos should learn about three, not fix one and re-run to
discover the next. Note that the aggregate is a plain `ConfigurationError`, not
a `DottedPathError`: most of what a block loader rejects is not a dotted-path
failure at all — a rule missing its target, a hook naming an unknown event —
and `DottedPathError` would misdescribe those.

**Nothing should skip silently.** A config key that names something
unresolvable and produces a bot that starts cleanly and quietly does less than
its configuration says is the worst of the three outcomes: the deployment
believes it is running something it is not, and the only trace is a WARNING in
a process that logs plenty of them.

## Guards

Two, and neither substitutes for the other:

| Guard | Catches | Cannot catch |
|---|---|---|
| Agreement test | drift between the sites we know about | a new copy nobody adds to the table |
| `assert_no_ad_hoc_dotted_import` | a new copy, the day it is written | drift within an adopted site |

```python
from dataknobs_common.testing import assert_no_ad_hoc_dotted_import

def test_no_ad_hoc_dotted_path_resolution():
    assert_no_ad_hoc_dotted_import(
        *(ROOT / "packages").glob("*/src"),
        allow={
            # Reviewed and deliberately left; adoption tracked separately.
            "fsm/src/dataknobs_fsm/config/builder.py:357",
        },
    )
```

An `allow` entry matching nothing is an error. A suppression whose site moved
is a hole, and a silent one reads as a clean scan.
