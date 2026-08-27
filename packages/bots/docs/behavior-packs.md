# Behavior Packs

A **behavior pack** is a named, frozen bundle of bot-shaping declarations —
middleware to install, a reasoning strategy to require, stage primitives to
expect. A deployment selects packs and tunes them in one binding block, and
resolution folds the selection into a single composed declaration.

`dataknobs_bots.behavior_packs` supplies the *vocabulary*; the composition
machinery is `dataknobs_common.packs` (see the
[Pack Composition guide](https://kbs-labs.github.io/dataknobs/packages/common/packs/) for merge-rule
semantics, binding syntax, warnings, and error families).

> **DataKnobs ships zero packs.** This module gives you the field names and
> the rules by which they combine. The pack *content* is your deployment's
> own policy.

## Why

Without packs, a platform that runs many bots ends up with the same
cross-cutting concerns copy-pasted into every bot config — an audit
middleware here, a compliance middleware there, drifting as they are
edited independently. A pack names that bundle once. A binding block then
says which bundles a given deployment gets, and the composition rules —
not ad-hoc merge code — decide what happens when two bundles both speak to
the same field.

## The spec

```python
from dataknobs_bots import BehaviorPackSpec

audit = BehaviorPackSpec(
    name="audit",
    priority=10,
    middleware=({"class": "acme.mw.AuditMiddleware", "params": {"level": "info"}},),
    stage_synthesizers=("intent_confirm",),
)
```

Every field other than `name` is optional — a pack is a *partial*
contribution, and an unset field is one the pack does not speak to.

| Field | Type | Rule | Meaning |
|---|---|---|---|
| `name` | `str` | *(meta)* | Registration key. Required. |
| `priority` | `int` | *(meta)* | Fold order — **lower folds first**. Default `0`. |
| `required_strategy` | `str \| None` | `UNANIMOUS` | Reasoning strategy this pack requires (e.g. `"wizard"`). |
| `strategy_overrides` | `Mapping[str, Any]` | `MERGE` | Reasoning-block settings the pack contributes. |
| `middleware` | `tuple[Mapping, ...]` | `CONCAT` | Bot-turn middleware specs. |
| `conversation_middleware` | `tuple[Mapping, ...]` | `CONCAT` | LLM-call-wrap middleware specs. |
| `stage_synthesizers` | `tuple[str, ...]` | `CONCAT_UNIQUE` | Wizard stage primitives the pack expects to be registered. |

### Why each field composes the way it does

- **`required_strategy` is `UNANIMOUS`** — two packs demanding different
  strategies is *unsatisfiable*, not resolvable. Silently keeping one would
  ship a bot that violates the other pack's stated requirement, so the
  conflict raises.
- **`strategy_overrides` is `MERGE`** — independent knobs. The
  higher-priority pack wins a contested key, and the collision is reported
  as a `key_override` warning rather than applied silently.
- **`middleware` / `conversation_middleware` are `CONCAT`** — order is
  behavior, and a repeated spec is a deliberate second installation. Both
  are carried because a pack that can only populate half the install rail
  is half a pack.
- **`stage_synthesizers` is `CONCAT_UNIQUE`** — these are *names*, not
  instances. Registration is idempotent, so a duplicate is noise rather
  than intent.

### Middleware specs stay opaque

`middleware` and `conversation_middleware` hold the **raw spec mappings**
the bot config already accepts:

```python
{"class": "acme.mw.AuditMiddleware", "params": {...}, "optional": False}
```

They are kept as data — not instances — so this spec and `DynaBotConfig`
cannot drift, and so a pack stays serializable. Turn them into live
instances with
[`build_middleware` / `build_conversation_middleware`](middleware.md#building-middleware-from-specs).

> **Packs and bindings are trusted configuration.** A spec's `class` is a
> dotted path that gets imported and instantiated, so building one executes
> whatever that module and constructor do. `middleware` composes with
> `CONCAT`, which means a binding body can *append* a spec — so anyone who
> can write a binding can name any importable class.
>
> Author packs and bindings in the same trust domain as the application's
> own code. Never build either from end-user input or from a blob a tenant
> supplies. If per-tenant selection is needed, let the tenant choose among
> pack *names* you registered, and keep the pack contents yours.

## The registry

There is **no module-level registry**, and you should not create one: a
pack binding is a per-deployment decision, so a process-global registry is
a multi-tenant hazard. Construct one and own it.

```python
from dataknobs_common.packs import PackRegistry
from dataknobs_bots import BehaviorPackRegistry, BehaviorPackSpec

registry: BehaviorPackRegistry = PackRegistry("behavior_packs", BehaviorPackSpec)
registry.register_pack(audit)
registry.register_pack(compliance)
```

`BehaviorPackRegistry` is a type alias for
`PackRegistry[BehaviorPackSpec]`. It exists so consumer signatures can name
the concrete type; construction goes through `PackRegistry` as shown.

## Binding

A deployment's YAML selects and tunes packs:

```yaml
behavior_packs:
  audit:
    locked: true              # this deployment must not turn it off
  compliance:
    strategy_overrides:       # tune one field of the pack
      max_tool_iterations: 3
  experimental:
    enabled: false            # registered, not applied
```

```python
resolution = registry.resolve(config.get("behavior_packs", {}))

resolution.packs            # ('audit', 'compliance') — fold order
resolution.spec             # the composed BehaviorPackSpec
resolution.warnings         # structured PackWarning diagnostics
```

Every named pack must be registered — including one you are switching off,
since `enabled: false` is a statement about a pack the deployment knows
about. An empty binding mapping resolves to an all-default composed spec
with no applied packs — **packs are opt-in.**

When a platform baseline and a per-tenant overlay both have a say, combine
them with `merge_bindings` rather than by hand — later layers win per pack
and per key, and it is what lets a baseline's `locked: true` outrank a
tenant's `enabled: false`:

```python
from dataknobs_common.packs import merge_bindings

resolution = registry.resolve(merge_bindings(platform_bindings, tenant_bindings))
```

See [Pack Composition → Bindings](https://kbs-labs.github.io/dataknobs/packages/common/packs/) for
the full binding contract (`enabled` / `locked` / `priority`, field
overrides, layering, and why unknown keys are rejected rather than ignored).

## Installing a composed pack

The whole rail, end to end:

```python
from dataknobs_bots import (
    DynaBot,
    build_conversation_middleware,
    build_middleware,
    verify_stage_synthesizers,
)

resolution = registry.resolve(config.get("behavior_packs", {}))

for warning in resolution.warnings:
    logger.warning("behavior pack composition: %s", warning)

verify_stage_synthesizers(resolution.spec.stage_synthesizers)

bot = await DynaBot.from_config(
    bot_config,
    platform_middleware=build_middleware(resolution.spec.middleware),
    platform_conversation_middleware=build_conversation_middleware(
        resolution.spec.conversation_middleware
    ),
)
```

`platform_middleware` **appends** to whatever the bot's own config
declared, rather than replacing it — which is what a cross-cutting platform
concern wants. See
[Middleware → Platform (additive) middleware](middleware.md#platform-additive-middleware).

Because `middleware` is a `CONCAT` field folded in ascending priority, the
**lowest-priority pack's middleware runs first**.

### `required_strategy` and `strategy_overrides` are yours to apply

These two fields are deliberately **not** wired into any DK build path.
Read them as data and apply them to your own reasoning block:

```python
composed = resolution.spec

if composed.required_strategy:
    declared = bot_config.get("reasoning", {}).get("strategy")
    if declared and declared != composed.required_strategy:
        raise ConfigurationError(
            f"Behavior packs require the '{composed.required_strategy}' "
            f"strategy but this bot declares '{declared}'"
        )
    bot_config.setdefault("reasoning", {})["strategy"] = composed.required_strategy

if composed.strategy_overrides:
    bot_config.setdefault("reasoning", {}).update(composed.strategy_overrides)
```

No DK-owned assembly path is assumed. A deployment that builds its own bot
config keeps doing so.

## Verifying stage synthesizers

A pack declares synthesizer *names*; the synthesizers themselves are
registered at import time by whichever module defines them. Nothing
connects the two, so a typo'd or forgotten name would otherwise surface as
a wizard stage whose primitive silently never expands.

```python
from dataknobs_bots import verify_stage_synthesizers

verify_stage_synthesizers(resolution.spec.stage_synthesizers)
```

Raises `ConfigurationError` listing **every** missing name plus what is
registered, so one call reports the whole gap rather than the first of it.

It registers nothing and imports nothing on your behalf — call it *after*
importing the modules that register the synthesizers you expect. That
ordering is the point: the check is meaningful only once registration has
had its chance to happen.

The registration surface itself is `register_stage_synthesizer`, exported
from `dataknobs_bots.reasoning`.

## Serialization

`BehaviorPackSpec` extends `PackSpec`, which extends `StructuredConfig`, so
packs load from config mappings and round-trip:

```python
spec = BehaviorPackSpec.from_dict({
    "name": "audit",
    "priority": 10,
    "middleware": [                       # YAML list -> tuple
        {"class": "acme.mw.AuditMiddleware"},
    ],
    "stage_synthesizers": ["intent_confirm"],
})
```

Normalization is **shallow** — the middleware spec mappings inside the
tuple are shared by reference, so treat them as read-only.

## Worked example

```python
from dataknobs_common.packs import PackRegistry
from dataknobs_bots import BehaviorPackSpec

registry = PackRegistry("behavior_packs", BehaviorPackSpec)

registry.register_pack(BehaviorPackSpec(
    name="baseline",
    priority=0,
    middleware=({"class": "acme.mw.RequestLogger"},),
    strategy_overrides={"max_tool_iterations": 5},
))

registry.register_pack(BehaviorPackSpec(
    name="regulated",
    priority=10,
    required_strategy="wizard",
    middleware=({"class": "acme.mw.AuditTrail"},),
    strategy_overrides={"max_tool_iterations": 2},
    stage_synthesizers=("intent_confirm",),
))

resolution = registry.resolve({"baseline": {}, "regulated": {"locked": True}})

resolution.packs
# ('baseline', 'regulated')

resolution.spec.middleware
# ({'class': 'acme.mw.RequestLogger'}, {'class': 'acme.mw.AuditTrail'})
#  ^ baseline first — CONCAT in ascending priority

resolution.spec.strategy_overrides
# {'max_tool_iterations': 2}     ^ regulated wins the contested key

resolution.spec.required_strategy
# 'wizard'                       ^ only one pack set it

resolution.warnings
# one PackWarning, code 'key_override'       ^ the contested key, reported
```

## Failure modes

| Situation | Result |
|---|---|
| Two packs set different `required_strategy` | `PackResolutionError(reason="field_conflict")` |
| Binding names an unregistered pack | `PackResolutionError(reason="unknown_pack")` |
| `locked: true` with `enabled: false` | `PackResolutionError(reason="locked_pack_disabled")` |
| Typo'd binding key (`lockd: true`) | `PackResolutionError(reason="unknown_binding_key")` |
| Binding gives a field a value its rule cannot use (a string for `middleware`) | `PackResolutionError(reason="invalid_binding")` |
| Binding overrides a field whose rule then discards the value | Resolves, with a `binding_override_ignored` warning |
| Declared synthesizer not registered | `ConfigurationError` from `verify_stage_synthesizers` |
| Middleware spec class is the wrong flavor | `ConfigurationError` from `build_middleware` (always — never covered by `optional`) |

Note that two packs agreeing on `required_strategy` reconcile silently —
`UNANIMOUS` requires agreement, not uniqueness.

## See Also

- [Pack Composition](https://kbs-labs.github.io/dataknobs/packages/common/packs/) — merge rules,
  bindings, warnings, custom reducers
- [Middleware](middleware.md) — the spec shape, the two flavors, and the
  additive platform channel
- [Configuration](configuration.md) — full bot configuration reference
