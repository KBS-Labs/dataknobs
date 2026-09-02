# Changelog

All notable changes to the dataknobs-config package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## v0.7.1 - 2026-09-02

### Changed

- **`EnvironmentConfig` copies its hand-outs through
  `dataknobs_common.copy_structure`.** The private `_copy_structure` helper
  that did this is now a public utility in `dataknobs-common`, because a
  second package needed the same copy. Behaviour here is unchanged except
  that the shared version also keeps its memo's sources alive — a latent
  defect the call sites in this module never triggered, since the structures
  they copy stay referenced for the duration.

## v0.7.0 - 2026-08-26

### Fixed

- **A config that contains itself is now reported rather than followed round.**
  YAML anchors build one directly — `a: &x` with `b: *x` under it is a dict
  that contains itself, and `yaml.safe_load` accepts it without complaint. Both
  readers of the `$resource` format descended such a tree until the stack ran
  out. Either now raises `ConfigError` naming where the cycle closed and where
  that block was entered.

  The cycle guard that already existed did not reach this. It tracks resource
  *identities*, so it catches a resource whose config references it again —
  invisible in the authored tree, since those are two different objects naming
  one resource. A block reaching itself is a cycle in *object* identity with no
  `$resource` key involved anywhere. Neither detects the other, and a config
  can carry either.

  Both guards now live on one object that the walk threads through its
  recursion, rather than as two things to forward independently — a missed
  forward being silent is the hazard that shape exists to remove. It is a stack
  and not a visited set: an anchor reused for its ordinary purpose puts one
  object at two paths without either containing the other, which is legitimate,
  common, and still resolves. Only what a descent is currently inside can close
  a cycle.

### Added

- **`collect_marker_violations()` reports `$resource` marker breaches without
  resolving.** The rule that decides which `$`-prefixed keys a config may carry
  was reachable only through resolution, which needs an `EnvironmentConfig` and
  raises on the first breach. A caller that holds a config tree and reports a
  verdict on it — a validator, an editor, a config-authoring tool — has neither
  an environment nor permission to raise, and the only thing exported for it was
  `RESOURCE_MARKER_KEYS`: the vocabulary, not the rule. Offering the set alone
  is what left the caller that asked to write its own rule around it, and a
  transcribed rule drifts even where the set it consults cannot.

  The new function walks any config tree and returns a `MarkerViolation` per
  breach — dotted path, and the same sentence the resolver raises — applying
  both halves of the rule at every depth: the closed vocabulary of a block that
  *is* a reference, and the stranded `$required` / `$requires` that says a block
  was meant to be one. Resolution is unchanged: the two internal guards still
  raise on the first offender, with the same exception type and message, and now
  share their definition with the collector rather than restating it.

  One divergence is deliberate and documented on the function. It descends into
  a reference's inline defaults unconditionally, where resolution walks only
  those an environment does not override — so a malformed reference inside an
  overridden default, which no build ever reaches and which goes live the day
  that override is removed, is reported here. A validator's subject is the
  authored config, not one deployment of it.

### Documented

- **The configuration guides teach `StructuredConfigConsumer` rather than
  `ConfigurableBase`.** The deprecated base is soft-deprecated by design — no
  runtime warning is raised, so the transition stays quiet — which left
  documentation as the only channel through which a new adopter could learn it
  is going away, and the guides were the channel recommending it. The
  step-by-step guide, the system overview and the package README are now
  written against the typed-dispatch successor, with the config dataclass as
  the schema; each keeps `ConfigurableBase` visible as the predecessor it is,
  with a migration path rather than a rename.

## v0.6.0 - 2026-08-19

### Added

- **A `$resource` reference can now declare that its resource must exist.**
  A reference naming a resource the current environment does not define warns
  and resolves to the reference's inline defaults — or to `{}` when it
  declares none. That is rarely what a production deployment wants, because an
  empty config handed to a factory usually does not fail: it produces the
  factory's default. A degraded `conversation_storage` binding becomes an
  in-memory database, which holds state perfectly until the process restarts.

  The policy is now declarable at four levels, most specific first, each
  unset-means-defer so an explicit `false` and an unspecified value stay
  distinguishable:

  | Level | Spelling | Reachable by |
  |---|---|---|
  | The reference | `$required: true` | the config author |
  | The reference | a non-empty `$requires` | the config author |
  | Code | `resolve_for_build(strict_resources=True)`, `EnvironmentAwareConfig(..., strict_resources=True)` | the calling code, the embedding application |
  | The environment | `settings: {strict_resources: true}` | the operator |

  The default is unchanged: with nothing declared, a missing resource still
  warns and degrades. The environment level exists because it is the only one
  a deployment whose references are *generated at runtime* can reach — there
  is no authored reference to annotate, and every other level lives in code
  the operator does not deploy.

  Both `$required` and the environment setting accept a boolean, or the
  strings `"true"` / `"false"` in any case and with surrounding whitespace
  ignored, so either works through `${VAR}` expansion. Any other value raises
  rather than reading as lenient. The environment setting is checked when the
  environment is **constructed** — by `EnvironmentConfig(...)`, `.load()` and
  `.from_dict()` alike — rather than when a resource turns out to be missing:
  a malformed flag is malformed in every environment, and deferring the check
  would surface it first in whichever deployment happened to lack a resource,
  as an error about a setting. A value still spelled as a template
  (`${STRICT}`, under `substitute_vars=False`) is left alone.

- **`find_unresolved_resources()`** on `EnvironmentAwareConfig` — every
  unresolvable reference in the tree in one pass, as
  `UnresolvedResourceRef(path, resource_type, resource_name, required,
  has_inline_defaults)`. Raise-on-first is right for a build and wrong for a
  preflight: an operator auditing a config wants the whole list, not the first
  entry. It constructs nothing and raises nothing for a missing resource, and
  reports a variable-selected `$resource: ${VAR}` under its resolved name.
  `resolve_for_build(strict_resources=True)` remains the first-failure form,
  and is safe to run at boot purely to prove every binding exists.

  It runs the **same walk** as the build, differing only in what it does when
  a resource is absent: record it and continue down the lenient path, rather
  than raise or warn. A survey with a traversal of its own is a second opinion
  about the build rather than a prediction of it, and the two disagree in both
  directions — descending into inline defaults the build discards, and
  stopping short of ones it reaches. Every failure that is *not* about
  presence (a malformed reference, a reference cycle, a resource that does not
  declare a capability its reference `$requires`) raises here rather than
  being listed, so an empty list means a build reaches no unresolvable
  reference.

- **`EnvironmentConfig.get_resource(..., required=)`** separates data from
  policy. `defaults` has carried both meanings at once — the values to merge,
  and, by being non-`None`, the decision not to raise. That coupling made one
  combination unreachable: *use these values for keys the resource does not
  set, but still fail if the resource itself is absent*. `required=None`
  (default) preserves the historical behaviour exactly.

- **`resolve_resource_references(config, environment, ...)`** is exported from
  `dataknobs_config` — the shared primitive behind both resolvers, so a
  consumer holding a config tree and an environment does not have to become a
  third reader of the format. Reading it independently is what produced the
  divergences this release closes: a hand-written walk recognises `$resource`
  and `type`, and thereby discards every inline default, ignores `$required`
  and `$requires`, lets a misspelled marker through as data, and leaves a
  reference nested inside a resolved resource as a literal dict.

  `RESOURCE_MARKER_KEYS`, `STRICT_RESOURCES_SETTING` and
  `UnresolvedResourceRef` are exported alongside it, for the reader that
  genuinely cannot delegate — a schema validator, an editor — so neither
  copies the literals.

- **A reference cycle is reported rather than followed.** A resource that
  reaches itself (`a` → `b` → `a`) raises `ConfigError` naming the chain, in
  both the build and the survey.

- **Failure messages name the dotted config path** of the reference that
  failed, so three references to `default` stay distinguishable in a log.

### Changed

- **`$requires` on an absent resource now raises.** The severity was
  inverted: a resource that existed but lacked a declared capability aborted
  the build, while a resource that did not exist at all resolved to its inline
  defaults and was handed to a factory. A resource that is absent satisfies no
  capability, so the weaker claim can no longer be the only one enforced.

  Declare `$required: false` alongside `$requires` to keep the previous
  behaviour for a reference where it was intended — "if it is there it must do
  X; it may be absent" is coherent, and the capability check still runs
  against the degraded config in that case.

- **A `$`-prefixed key that is not a marker is now an error.** The marker set
  (`$resource`, `type`, `$requires`, `$required`) is closed, and the
  comprehension that builds a reference's inline defaults takes everything
  else — so a misspelled marker was not rejected, it was promoted to a default
  and passed to the factory as a keyword argument. `$requred: true` would
  therefore have meant *not required*, silently, at the exact site meant to
  close that class of failure.

  Two more spellings of the same mistake are closed with it. **`$requires`
  must be a list of names**: a bare `$requires: persistence` is truthy and
  iterates character by character, so it produced a check against letters.
  And **`$required` or `$requires` on a block with no `$resource` is
  rejected** — the guard above fires on a block that already *is* a
  reference, so a typo in the selector key itself (`$resorce:`) produced an
  ordinary dict that resolved to itself and reached the factory with its
  markers attached. A leftover policy marker is what gives that away.

- **`ConfigBindingResolver` now resolves references nested inside the resource
  it looks up.** It fetched a resource and handed it straight to the factory,
  so a resource carrying its own `$resource` block passed one on as a literal
  `{"$resource": ...}` keyword argument — the same silent degrade the marker
  guard closes, one layer down. Both resolvers now run the same resolution
  below the entry point: same marker validation, same precedence chain, same
  cycle guard, and a nested reference can declare `$required: false` for
  itself. Caller-supplied `**overrides` are resolved the same way.

  Its raise on a missing resource at the *top* is unchanged, and is now
  documented as the strict policy rather than left looking incidental. That
  API takes a `(type, name)` pair with no reference to read a policy off, so
  strictness is the only coherent answer there; the declarable behaviour
  belongs to the reference syntax. Both resolvers raise
  `ResourceNotFoundError` for a missing resource, `ConfigError` for a
  malformed reference, an under-capable one, or a cycle.

  ⚠️ **`ResourceNotFoundError` subclasses both `EnvironmentConfigError` and
  `KeyError`.** `resolve_for_build()` could not previously raise a `KeyError`;
  under a strict policy it can. Code wrapping resolution in `except KeyError`
  for unrelated reasons will swallow it. The hierarchy is unchanged — narrowing
  it is a breaking change to an exported type, and is not worth making here.
  Its `__str__` is restored, though: `KeyError.__str__` returns
  `repr(args[0])`, which wrapped the message in quotes and escaped every name
  inside it. The bases are what consumers depend on; the rendering is what a
  person reads.

- **`resolve_for_build(strict_resources=..., resolve_resources=False)` now
  raises `ValueError`.** The policy is read where references are resolved, so
  the pair validated nothing and returned a config anyway — from the method
  documented as *the* startup preflight. The instance-level policy is
  unaffected: it is a standing default rather than an assertion about one
  call. `find_unresolved_resources`'s `strict_resources` is keyword-only, for
  parity with `resolve_for_build`.

### Security

- **An `@`-reference could read any file on the volume.**
  `Config._load_referenced_file` composed a reference read out of a config
  *value* onto `config_root` and read the result with no containment. Any
  string in a config list beginning with `@` is a file reference, so a `..`
  segment climbed out of the config tree — and the absolute branch never
  consulted `config_root` at all, which is the wider hole of the two: it
  discards the root rather than climbing out of it. Both spellings loaded a
  file carrying an `api_key` from outside the tree.

  Both branches are now bounded, and the branch itself is gone — containment
  is judged on where the reference *lands*, so an absolute reference pointing
  back inside `config_root` is still fine and naming a subdirectory is still
  how a layout convention is expressed. Leaving the root raises
  `ConfigPathEscapeError` (a `ValueError`, and a `ConfigLoadError`). With no
  `config_root` set there is no tree, and nothing is bounded.

  **Breaking for a deployment that references across trees deliberately.** The
  migration is one argument: `Config(..., allow_reference_outside_config_root=True)`,
  also on `from_file` and `from_dict`. Each actual escape is then logged at
  WARNING and read anyway — the convention `find_config_file`'s `allow_outside`
  already follows, so the bypass is visible in a deployment's logs rather than
  silent. Only a real escape warns; a load that never leaves the root is quiet.

  The opt-out is a **caller argument and cannot be set from a config file**,
  which is the same reason the guard exists: a reference is bounded *because*
  it comes out of config content rather than from the calling code, so a switch
  readable from a `settings:` block would let that content turn off the check
  that bounds it. `SettingsManager.load_settings` refuses the key with a
  `ConfigError` naming the argument to use instead — refused rather than
  dropped, since a silent drop fails closed but leaves an operator who wrote it
  in YAML watching their references raise with nothing pointing at why. This
  matches every sibling opt-out in the package (`find_config_file`,
  `InheritableConfigLoader`, `EnvironmentConfig.load`), all caller parameters
  defaulting to `False`.

  `config_root` could widen the same boundary from content, and does not: a
  file load pins it to the entry file's own directory before that file's
  `settings:` block is read, so an entry file cannot name the root it is
  bounded to.

### Added

- **`allow_outside=True` lifts the name-containment bound, on all three
  loaders.** `InheritableConfigLoader(config_dir, allow_outside=True)`,
  `EnvironmentConfig.load(..., allow_outside=True)`, and
  `EnvironmentAwareConfig.load_app(..., allow_outside=True)` opt a deployment
  out for a layout that genuinely spans sibling trees — `configs/app.yaml`
  carrying `extends: ../shared/base`, which the containment bound below
  otherwise rejects with no remedy short of restructuring the tree.

  Off by default everywhere; the bound is the behaviour you get unless you
  ask for otherwise. A name that actually escapes is logged at WARNING when
  it does, so a widened boundary is auditable in a deployment's logs, while
  a contained name logs nothing and the signal stays meaningful.

  Two scoping notes worth reading before turning it on. On
  `InheritableConfigLoader` the flag covers every name the loader resolves —
  the requested config, each `extends:` target, and a resolver's output —
  because they all reach the same join. On the two environment-driven
  loaders it widens what a *process environment variable* can address:
  called without an explicit `environment`, they take the name from
  `DATAKNOBS_ENVIRONMENT` or `ENVIRONMENT`. `load_app` applies it to both of
  its lookups, the app name and the environment name, rather than silently
  covering only one.

### Security

- **A configuration name could address a file outside the directory it was
  loaded from.** `InheritableConfigLoader`, `EnvironmentConfig`, and
  `EnvironmentAwareConfig` all compose a name into their configured
  directory, and none of them bounded it: a name containing `..` walked out
  of the directory, and an absolute name discarded the directory entirely.
  All three now reject such a name — `InheritanceError`,
  `EnvironmentConfigError`, and `EnvironmentAwareConfigError` respectively,
  each naming the config rather than the resolved filesystem path.

  Three of the four names this affects are not a caller's own literal, which
  is what makes the bound worth having:

  - an `extends:` value is read out of a config file, and the loader
    recurses into parents entirely on its own, so `extends: "../../etc/x"`
    was followed with nothing between it and the read;
  - an environment name comes from `DATAKNOBS_ENVIRONMENT` or `ENVIRONMENT`
    whenever `EnvironmentConfig.load()` is called without one;
  - a resolved name comes from a consumer-supplied `ResourceResolver` or a
    `resolve_name` override.

  A name addressing a *subdirectory* is unaffected — `domains/child`, and
  every `extends:` target under a layout convention, resolves exactly as
  before. `load_from_file` is also unaffected in the direction that matters:
  the path the caller passes is the caller's own choice, so it still reads
  any file, and the `extends:` targets under it are bounded by the directory
  that file lives in.

### Changed

- **`EnvironmentConfig.load()` now fails at startup on an escaping
  environment name, where it previously started with an empty config.** An
  environment name that addressed nothing — including one that addressed
  outside `config_dir` — used to fall through to "no config file found for
  this environment" and return an empty `EnvironmentConfig`, so a
  deployment with a malformed `DATAKNOBS_ENVIRONMENT` booted and ran on
  defaults. An escaping name now raises `EnvironmentConfigError` instead.

  This is the intended fail-closed direction — a name that tried to leave
  the configuration directory is a misconfiguration worth halting on rather
  than silently running unconfigured — but it is a behaviour change for any
  deployment that was relying on the degraded path. A name that is merely
  *absent* from `config_dir` still returns an empty config exactly as
  before; only escaping names raise.

## v0.5.0 - 2026-08-11

### Fixed

- **`Config.get_registered_factories()` raised `AttributeError` on every
  call.** It asked its backing `Registry` for a `copy()` method that registry
  does not have, so a documented, exported accessor could not be called at all.
  It now returns a snapshot mapping the caller may mutate freely. The three
  methods of the runtime factory registry had no test between them, which is
  how this shipped; they have one now.

- **A configuration value whose own text contains `${...}` was expanded
  twice, replacing it with the value of an unrelated environment variable.**
  Substitution ran in two layers that did not know about each other — once
  when an `EnvironmentConfig` loaded, and again when a resolution layer read
  through it — and the second pass expanded the *output* of the first. A
  generated password of `p${x}ss` therefore arrived as `pINJECTEDss` if some
  variable `x` happened to be set. Nothing warned; the value was simply
  wrong, and wrong in a way that depends on the rest of the process
  environment.

  Two paths carried it. `EnvironmentAwareConfig.resolve_for_build` corrupted
  the resolved config dict a reviewer could at least diff.
  `ConfigBindingResolver` is the more serious one: it hands the value
  straight to a factory that builds a live resource, so a mangled DSN or API
  key produced a connection failure with no artifact anywhere showing the
  value that was actually used. Beyond corruption, the expansion reads the
  process environment using text taken from a secret — a value that happens
  to contain `${AWS_SECRET_ACCESS_KEY}` would have pulled that variable's
  contents into wherever the first value was headed.

  `EnvironmentConfig` now records whether its values have been substituted,
  and both layers substitute a source exactly once. Configs built directly
  via the dataclass constructor, or loaded with `substitute_vars=False`,
  still carry raw refs and are still expanded by the resolution layers — but
  they are not otherwise untouched: `resolve_for_build` expands them per
  resource as each is spliced rather than as a whole document, and `merge()`
  can now raise for them.

  Where each source is expanded follows from the same rule. A resource is
  still separable when it is spliced, so that is the latest point it can be
  expanded, and the latest point is the safest one — expanding earlier reads
  values the build then discards, so an unset required `${VAR}` in a resource
  no reference names cannot abort a build that never looked at it. A
  reference's inline defaults get the same treatment one level in: the splice
  discards every one the environment supplies, so each is expanded there,
  only once known to survive. A dev-time fallback such as
  `password: ${LOCAL_DB_PASSWORD}` therefore need not be set in production,
  where the environment overrides it; a default that *is* used still raises
  when its variable is unset. This holds for a default's value, not for the
  key naming it: what proves a default was discarded is its key, so a
  required `${VAR}` in key position must resolve either way. A `$resource`
  block nested inside a default —
  or inside a resource an unsubstituted environment supplies — reaches its
  own splice raw for the same reason. Nested inside a resource an
  already-substituted environment supplies, it was expanded at that
  environment's load, and `substituted` is what stops the splice expanding it
  again. Either way it is expanded exactly once.

  `substituted` describes the values a config was *built* with. Writing into
  `resources` or `settings` afterwards does not update it, and a layer
  reading a stale `True` will skip the pass those new values needed. Build
  the config you want rather than amending one; if you must amend, re-mark
  it with `dataclasses.replace(env, substituted=False)`. The dataclass is
  deliberately left mutable — freezing it would break consumers that
  assemble an environment field by field — so this is a stated contract
  rather than an enforced one.

- **A `$resource` reference naming a resource that does not exist now logs a
  warning.** The warning existed but was unreachable — it sat in an
  `except KeyError` branch, while the call it guarded returns the supplied
  defaults instead of raising whenever a defaults dict is passed, which that
  call site always does. A mistyped binding name therefore degraded in
  complete silence to the reference's inline defaults — an empty config when
  it declares none. The fallback behaviour itself is unchanged: a degraded
  config is still config, so it gets the same `$requires` check and the same
  resolution a found one does — which is what resolves a nested `$resource`
  inside the reference's inline defaults, and what keeps that nested
  reference's `$resource` / `type` marker keys from reaching a factory as
  keyword arguments.

- **`EnvironmentConfig.load()` reported the wrong provenance when the
  environment file was absent.** It short-circuits to an empty config before
  reaching the construction that records `substituted`, so it always reported
  `False` — while `from_dict({})` reported `True` for the same request. Two
  empty configs built the same way disagreed, and the documented truth table
  carved out no such case. Note the consequence: an absent-file config is now
  a *substituted* config, so merging one with a directly-constructed config
  is a mixed-provenance merge and can raise on an unset variable.

- **`InheritableConfigLoader` returned a different config depending on what
  had been loaded before it.** Resolving `extends:` loads the parent without
  substitution and expands the merged result at the end, but both forms were
  cached under the same name — so whichever was stored first was served to
  the other. Loading a child and then its parent returned the parent with
  `${VAR}` placeholders still in it, unexpanded. Loading a parent and then
  its child expanded the parent's values a *second* time, producing the
  corruption described in the first entry through a different route:
  `p${x}ss` again arriving as `pINJECTEDss`. Both symptoms came and went with
  load order. Configs whose values contain no `${` were unaffected in either
  direction. `clear_cache(name)` clears every variant stored under that name.

### Added

- **`EnvironmentConfig.substituted`** — whether `${VAR}` substitution has
  been applied to the values held. Set by `load()` / `from_dict()` from their
  `substitute_vars` argument; `False` for direct dataclass construction.
  Excluded from equality, so two configs holding the same values remain
  equal regardless of which layer expanded them.

- **`EnvironmentConfig.substituted_view()`** — an equivalent config with
  substitution applied, or `self` when it already is. Never mutates the
  receiver, so a caller holding raw refs on purpose keeps them even after a
  resolution layer reads through the config.

- **`InheritableConfigLoader.resolve_name()`, and a `resolver=` argument to
  go with it** — a public seam for how a configuration *name* maps to a
  location under `config_dir`. It applies to every `extends:` target as well
  as to the requested config, which is the point: parents are named inside
  config files and the recursion into them happens entirely inside the
  loader, so a deployment that could only intercept the entry point could not
  express a layout convention at all. A config tree under `domains/` whose
  children say `extends: parent` was unloadable without overriding a private
  method.

  Two ways to use it, and they are **alternatives, not layers.** An override
  replaces the default implementation, so a loader given both ignores the
  injected resolver — unless the override delegates to
  `super().resolve_name(...)`, in which case both mappings apply in sequence
  and a prefixing pair looks for `domains/domains/x.yaml`. Constructing that
  combination warns, because the first outcome is otherwise silent: the
  loader reads a file the caller did not configure and says nothing. It is a
  warning rather than an error, since overriding to normalize or log *and*
  delegating to `super()` is a legitimate use of both. Pick one:

  ```python
  from dataknobs_common import CallableResolver, MappingResolver

  # Inject a resolver — no consumer class needed for either common layout.
  InheritableConfigLoader(root, resolver=CallableResolver(lambda n: f"domains/{n}"))
  InheritableConfigLoader(root, resolver=MappingResolver({"tutor": "domains/bio-tutor"}))

  # Or override the public method, when the mapping needs loader state.
  class DomainAware(InheritableConfigLoader):
      def resolve_name(self, name: str) -> str:
          return f"domains/{name}"
  ```

  A resolver returning `None` means "no mapping" and falls back to identity,
  per the `ResourceResolver` contract. With no resolver and no override,
  `resolve_name` is identity and nothing about a flat layout changes.

  The resolved name is what the loader keys on throughout — the cache, the
  cycle-detection set, the `extends:` invalidation edges, and `clear_cache`
  — so two spellings of one config are one cache entry, one node, and one
  thing to clear, rather than a config that is two to one structure and one
  to another.

  `load_from_file` suppresses resolution for the file **and** its `extends:`
  subtree. A mapping is defined relative to `config_dir`, and that method
  rebinds `config_dir` to the file's own directory, so a convention applied
  there would look for its location beneath a directory the caller chose
  instead. The suppression is honored where resolution is invoked, so
  overriding `resolve_name` does not defeat it.

  `clear_cache` resolves the name it is given, the same way `load` does, so
  pass it the name you passed `load`. An already-resolved name is mapped a
  second time — harmless for a lookup table, which leaves a name it has no
  entry for alone, but a prefixing resolver double-prefixes and the call
  clears nothing. Nothing is raised; the debug log reports the names it
  targeted **and** how many cached entries that removed, so
  `Cleared 0 cache entries for: domains/domains/child` is the sign.

- **`InheritableConfigLoader.available_names()`** — the names `load` accepts,
  for this deployment's layout. `list_available()` now delegates to it, and
  it is the one to override.

  It exists because `resolve_name` is one-way. A resolver answers "where does
  this name live"; nothing runs it backwards to recover the names from the
  locations, so a deployment that governs the mapping has to govern
  enumeration too. The default — the stems of the files directly under
  `config_dir` — is the loadable set only while `resolve_name` is identity,
  and leaving it alone under a resolver does not raise, it reports the wrong
  thing quietly. Under a layout one directory down it returns `[]`, so the
  natural `for name in ...: load(name)` loop runs zero times; a layout mixing
  depths is worse than empty, because the stems it does find are *locations*,
  and mapping a location through `resolve_name` addresses something else
  again.

  ```python
  class DomainLoader(InheritableConfigLoader):
      def resolve_name(self, name: str) -> str:
          return f"domains/{name}"

      def available_names(self) -> list[str]:
          return self.stems_in(self.config_dir / "domains")
  ```

  A flat layout is unchanged: with no resolver and no override, the default
  is exactly what `list_available` returned before.

- **`InheritableConfigLoader.stems_in(directory)`** — the default
  `available_names` body, taking a directory, public so an override can point
  it somewhere else. It globs the extensions `load` itself probes, read from
  the one shared list, so enumeration cannot fall behind loading. Writing the
  glob by hand is the quiet way to get an override wrong: covering `*.yaml`
  alone omits every `.json` config from the listing while leaving each one
  perfectly loadable.

### Changed

- **A `class:` or `factory:` path may now be written `module.path:Name`, and a
  failure to resolve one says what actually went wrong.** Both keys resolve
  through `dataknobs_common.imports` rather than a local copy, so they accept
  the same two separators as every other dotted path in the workspace and
  report the same way.

  `ObjectBuilder._load_class` previously wrapped every failure in
  `Failed to load class {path} ({ExceptionType})`, which named the type of the
  underlying error but discarded its distinctions — a module that is not
  installed, a module that raised while importing, and a module missing the
  named attribute were one message. They are now separate `DottedPathError`
  reasons. The error type is unchanged in practice: `DottedPathError` is a
  `ConfigurationError`, which is what `ConfigError` already aliases, so
  `except ConfigError` catches exactly what it caught before.

- **A `$resource` name or `type` containing `${VAR}` now resolves.**
  `resolve_for_build` substitutes the app config before splicing in resource
  references, and a reference's marker keys — `$resource`, `type`,
  `$requires` — are expanded by that pass, so resource *selection* can be
  bound to an environment variable (`$resource: ${LLM_BINDING}`). Previously
  the literal text was looked up, matched nothing, and fell back to the
  reference's inline defaults.

  That fallback was silent (see `### Fixed`), so a deployment cannot find it
  in existing logs. To check whether you were relying on it, search your app
  configs for `$resource:` or `type:` values containing `${`; those are
  exactly the references whose behaviour changes. After upgrading, the
  now-reachable *"Resource '...' not found in environment ..., using
  defaults"* warning reports any that still fail to match.

- **`EnvironmentConfig.merge()` normalizes mixed substitution provenance.**
  Merging a substituted config with an unsubstituted one expands the
  unsubstituted side during the merge and returns a substituted result,
  rather than producing a config whose single flag is wrong for half its
  values. Merging two configs that agree is unchanged.

  That normalization runs a substitution pass, so `merge()` — previously pure
  data manipulation with no dependency on the process environment — can now
  raise `RequiredEnvVarError` when the two sides disagree and the
  unsubstituted one holds an unset required `${VAR}`. Merging two sides that
  agree still touches no environment variables and cannot raise.

- **`get_resource()`, `merge()` and `to_dict()` now copy nested structure.**
  All three documented or implied a copy but stopped at the resource-config
  level, so every container *inside* a resource was the environment's own
  object: `env.to_dict()["resources"][type][name]["pool"]` was the live
  config's own dict, and a consumer adjusting a nested section wrote through
  into an environment that outlives the resolution. This was masked wherever
  a substitution pass ran afterwards — `substitute_env_vars` rebuilds the
  structure through comprehensions, isolating the result incidentally — and
  surfaced once that pass was correctly skipped for an already-substituted
  environment.

  Containers are copied; every other value is passed through by identity,
  which is the same bound the substitution pass set while it was providing
  the isolation incidentally. Copying the leaves too would overshoot: a
  resource assembled in Python can hold a live object — a connection pool, a
  prebuilt provider, a lock — and duplicating one would hand a factory a
  second pool, or raise `TypeError` on a value that cannot be pickled. A
  container reached twice is copied once and the same copy used both times,
  so a structure that refers back to itself terminates rather than exhausting
  the stack, and one that shares a subtree between two keys keeps sharing it.

- **`InheritableConfigLoader.load(use_cache=False)` no longer writes to the
  cache.** Bypassing the cache now means not taking part in it in either
  direction. Two callers depend on this: `validate()` is a dry run, and
  `load_from_file()` reads with `config_dir` rebound to another directory —
  and since the cache key carries no directory, the entry it left behind
  answered later `load()` calls for the directory the loader was actually
  configured for. Inheritance edges are recorded on the same condition, so a
  bypassing load no longer files a dependent under a bare parent name from
  another directory either.

  Note for callers using `use_cache=False` as a reload: it is a bypass, not a
  refresh. It reads fresh but leaves the stored entry in place, so subsequent
  `load()` calls still get the cached one. To make a reload visible to other
  callers, `clear_cache(name)` and then `load(name)`.

- **`InheritableConfigLoader.clear_cache(name)` now clears dependents.**
  A cached child holds its parent's content merged in, so clearing only the
  parent left that copy answering — the staleness the call was made to
  resolve, surviving the call. Clearing a name now transitively clears every
  config that reached it through `extends:`. Invalidation runs down the
  inheritance edges, never up: clearing a child leaves its parent cached.

### Security

- **`ObjectBuilder._load_class` no longer puts a failed import's message into
  the `ConfigError` it raises.** `ConfigError` is a
  `dataknobs_common.exceptions.ConfigurationError`, which the `dataknobs-bots`
  API layer renders at the HTTP boundary, so whatever went into that string
  could reach a response body — and neither branch's text was this package's
  to publish. Resolving a dotted class path calls `importlib.import_module`,
  which *executes* the target module, so the non-`ImportError` branch carried
  text produced by module-level code the deployment supplied: a module that
  connects on import raises with its connection URL in the message. The
  `ImportError` branch looks narrow enough to be safe and is not — `cannot
  import name 'X' from 'pkg' (/abs/path/site-packages/pkg/__init__.py)` is an
  absolute filesystem path, which this package withholds from a not-found
  error for exactly that reason.

  Both messages now name the class path (which comes from the config, not from
  the exception) and the exception type; the original travels on `__cause__`.
  The `except TypeError` instantiation branches are unchanged, their text
  being a constructor signature mismatch.

- **A config file's resolved path and contents no longer reach the error
  message.** `Config` resolves a path to an absolute one before reporting it,
  so "configuration file not found" doubled as a map of the server's
  filesystem; and a parse failure relayed the parser's own text, which quotes
  the line it choked on — an unterminated quote on an `api_key` put the key
  itself in the message. Both types are rendered at the HTTP boundary by the
  `dataknobs-bots` API layer with their message shown (a 404 and a 422
  respectively), and bots are built lazily on the request path.

  A not-found error now names the file and carries the resolved path in
  `context`, which that type does not disclose and which a library caller
  reads directly. A parse failure names the file and the error class, with the
  parser's text on `__cause__`. The unsupported-extension case is rebuilt from
  the path rather than relayed, since the underlying message names the full
  path too.

  Both loaders route through one private reader, so the decision is made once;
  the duplicate existence check in `_load_file` went with it, which means an
  unreadable config no longer leaves `config_root` set — an object the raised
  exception already made unusable.

## v0.4.4 - 2026-07-29

## v0.4.3 - 2026-07-20

## v0.4.2 - 2026-07-15

## v0.4.1 - 2026-06-29

## v0.4.0 - 2026-05-26

### Added
- Async object-construction bridge for the `StructuredConfigConsumer`
  contract. `Config.build_object_async(ref, ...)` (and the underlying
  `ObjectBuilder.build_async`) prefer a target class's
  `from_config_async` — the async entry point from
  `dataknobs_common.structured_config` — awaiting it, then fall back to
  a factory's `create_async`, then to the synchronous `from_config` /
  direct-instantiation path, so the method is safe for any reference.
  `ConfigBindingResolver.resolve_async` likewise prefers a target's
  `from_config_async` over factory `create_async`/sync construction.
  The synchronous `build_object` / `resolve` paths are unchanged.
  Lets the object-graph layer build async consumers (databases that
  connect eagerly, LLM-backed bots, knowledge-base warmup) through the
  same uniform per-object contract the sync path already uses.

### Deprecated
- `ConfigurableBase` (`dataknobs_config.builders`) — soft-deprecated
  in favor of `StructuredConfigConsumer[ConfigT]` from
  `dataknobs_common.structured_config`. `ConfigurableBase` performs
  kwarg-splat construction (`cls(**config)`); the successor provides
  typed-dispatch construction with auto-derived `from_dict`, a
  `_normalize_dict` override hook, and a unified parity guard.
  Existing consumers continue to work; no runtime warning is raised
  so the transition stays quiet across the multi-cycle migration.
  Removal is scheduled for a future release once the in-tree
  migration is complete. See `packages/config/docs/configurable-base.md`
  for the migration sketch.

## v0.3.14 - 2026-05-18

## v0.3.13 - 2026-05-13

### Fixed
- Bumped minimum `pyyaml` requirement from `>=6.0` to `>=6.0.2` to
  exclude versions that lack cp312/cp313 wheels and fail to build from
  source against modern Cython (`'build_ext' object has no attribute
  'cython_sources'`). Surfaced by the floor resolve step in the
  `dependency-update` workflow.

## v0.3.12 - 2026-05-09

### Added
- `RequiredEnvVarError`, a `ValueError` subclass raised by
  `substitute_env_vars` (and every loader that calls it) when a required
  ``${VAR}`` or ``${VAR:?msg}`` is unset. Carries `var_name`,
  `bash_form`, and `explicit_message` so callers can branch on the
  failure shape without parsing message text. Existing `except
  ValueError` / `pytest.raises(ValueError)` callers keep working.

### Changed
- `EnvironmentConfig.load()` and `EnvironmentConfig.from_dict()` now apply
  `${VAR}` / `${VAR:default}` substitution by default, matching the
  behaviour of `InheritableConfigLoader.load()`. Pass the new keyword-only
  `substitute_vars=False` to opt out (e.g., to inspect raw refs). Required
  `${VAR}` refs without a default raise `ValueError` at load time.
- `substitute_env_vars` is now the canonical environment-variable
  substitution helper across the package. It accepts three keyword-only
  options: `type_coerce` (default `False`; coerce whole-value `${VAR}`
  placeholders to `int` / `float` / `bool`), `expand_user_paths` (default
  `True`; applies `os.path.expanduser` to substituted values), and
  `substitute_keys` (default `True`; substitutes `${VAR}` references in
  dict keys as well as values). The regex recognises bash-style
  `${VAR:-default}` and `${VAR:?error_msg}` in addition to the legacy
  `${VAR:default}` form. `Config._load_dict` calls `substitute_env_vars`
  with `type_coerce=True, expand_user_paths=False, substitute_keys=False`.
- `type_coerce=True` no longer treats `"0"` / `"1"` as booleans. Only the
  unambiguous bool words `true` / `false` / `yes` / `no` (case-insensitive)
  coerce to `bool`; numeric `"0"` and `"1"` coerce to `int`. This affects
  every caller that opts in to `type_coerce=True`, including
  `Config._load_dict`, the `VariableSubstitution` shim, and direct
  `substitute_env_vars(..., type_coerce=True)` callers.

### Deprecated
- `VariableSubstitution` is now a thin compatibility shim over
  `substitute_env_vars(data, type_coerce=True, expand_user_paths=False,
  substitute_keys=False)` and emits `DeprecationWarning` on construction.
  Use `substitute_env_vars` directly. The class will be removed in a
  future release.

### Changed
- `Config._load_file` raises `ValidationError` for malformed YAML or
  JSON config files. `yaml.YAMLError` and `json.JSONDecodeError` no
  longer escape; callers should catch `ValidationError` (or its base,
  `ConfigError`).
- Loader error messages for a non-dict root now read
  `"Expected a dict at the root of <path>, got <type>"` (previously
  `"must contain a dictionary"` / `"must be a dictionary"`).
  Exception types are unchanged.

### Internal
- `InheritableConfigLoader._load_file`, `EnvironmentConfig._load_file`,
  `EnvironmentAwareConfig._load_file`, `Config._load_file`, and
  `Config._load_referenced_file` share the
  `dataknobs_common.config_loading` helpers
  (`load_yaml_or_json`, `find_config_file`). Each loader wraps the
  helper's `ConfigLoadError` as its existing public error class
  (`InheritanceError`, `EnvironmentConfigError`,
  `EnvironmentAwareConfigError`, `ValidationError` /
  `ConfigFileNotFoundError`).
- `Config._load_referenced_file` coerces empty / falsy parsed
  payloads to `{}`.

## v0.3.11 - 2026-05-06

## v0.1.0 - 2025-01-12

### Added
- Initial release of dataknobs-config package
- Core `Config` class for managing modular configurations
- Support for YAML and JSON file formats
- Atomic configuration management with type/name-based access
- String reference system (`xref:`) for cross-referencing configurations
- Environment variable override system with bash-compatible naming
- Global settings and defaults management
- Path resolution for relative paths in configurations
- Object construction support with class instantiation and factory patterns
- Object caching for improved performance
- Comprehensive test suite with 91% code coverage
- Full type annotations with mypy support
- Detailed documentation and usage examples

### Features
- **Modular Design**: Organize configurations by type with atomic units
- **File Loading**: Load from YAML, JSON, or Python dictionaries
- **Cross-References**: Link configurations using `xref:type[name]` syntax
- **Environment Overrides**: Override any config value via environment variables
- **Path Resolution**: Automatic resolution of relative paths
- **Object Building**: Optional object construction from configurations
- **Settings Management**: Global and type-specific defaults
- **Extensible**: Clean interfaces for custom builders and factories

### Technical Details
- Python 3.8+ support
- Dependency: PyYAML >= 6.0
- Development dependencies include pytest, mypy, ruff, and types-PyYAML
- Follows PEP 8 style guidelines
- 100% type annotated codebase
