# Changelog

All notable changes to the dataknobs-config package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Fixed

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
  clears nothing. Nothing is raised; the log line names what was cleared.

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
          return sorted(
              path.stem
              for path in (self.config_dir / "domains").glob("*.yaml")
              if path.is_file()
          )
  ```

  A flat layout is unchanged: with no resolver and no override, the default
  is exactly what `list_available` returned before.

### Changed

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

- **`ObjectBuilder._load_class` no longer puts an arbitrary import failure's
  message into the `ConfigError` it raises.** Resolving a dotted class path
  calls `importlib.import_module`, which *executes* the target module, so the
  non-`ImportError` branch was interpolating text produced by module-level
  code the deployment supplied — a module that connects on import raises with
  its connection URL in the message. `ConfigError` is a
  `dataknobs_common.exceptions.ConfigurationError`, which the `dataknobs-bots`
  API layer renders at the HTTP boundary, so that string could reach a
  response body.

  The message now names the class path (which comes from the config, not from
  the exception) and the exception type; the original travels on `__cause__`.
  The `ImportError` branch is untouched — its text is module and attribute
  names — as are the `except TypeError` instantiation branches, whose text is
  a constructor signature mismatch.

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
