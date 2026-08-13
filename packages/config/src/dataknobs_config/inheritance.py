"""Configuration inheritance utilities.

This module provides simple configuration inheritance via an `extends` field,
complementing the existing Config system with single-file config loading.

The InheritableConfigLoader supports:
- Loading YAML/JSON configuration files
- Configuration inheritance via 'extends' field
- Deep merge with child overriding parent
- Environment variable substitution
- Caching for performance

Example:
    ```yaml
    # base.yaml
    llm:
      provider: openai
      model: gpt-4
      temperature: 0.7

    knowledge_base:
      chunk_size: 500
      overlap: 50

    # domain.yaml
    extends: base

    llm:
      model: gpt-4-turbo  # Override just this field

    domain_specific:
      feature_enabled: true
    ```

    ```python
    loader = InheritableConfigLoader("./configs")
    config = loader.load("domain")  # Merges base.yaml + domain.yaml
    ```
"""

import logging
import os
import re
import warnings
from pathlib import Path
from typing import Any

from dataknobs_common import ResourceResolver
from dataknobs_common.config_loading import (
    DEFAULT_CONFIG_EXTENSIONS,
    ConfigLoadError,
    ConfigPathEscapeError,
    find_config_file,
    load_yaml_or_json,
)

logger = logging.getLogger(__name__)


class InheritanceError(Exception):
    """Error during configuration inheritance resolution."""

    pass


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries.

    Recursively merges override into base, with override values taking
    precedence. Nested dictionaries are merged recursively; **every other type,
    lists included, is replaced** -- a list in ``override`` takes the place of
    the list in ``base`` rather than extending it. This is the merge semantic
    for the whole codebase; a caller wanting accumulation concatenates before
    merging.

    Neither argument is mutated. That guarantee is top-level: the copy at each
    level is shallow. A fresh dict is built only where *both* sides supply a
    dict; every other value, at every depth, is shared **by reference** with
    whichever input supplied it. That includes values under a key both inputs
    declare -- merging ``{"a": {"x": [1, 2]}}`` with ``{"a": {"y": 3}}``
    rebuilds ``a`` but hands back the very same ``x`` list.

    So the result is safe to rebind keys on and is **not** deeply isolated
    from its inputs -- a caller that mutates a nested value reached through
    the result mutates it in the input too. Callers needing isolation copy
    first; ``copy.deepcopy`` on the way in is the usual shape, and is why a
    caller merging into a module-level constant needs one.

    Args:
        base: Base dictionary (values used when not overridden)
        override: Override dictionary (takes precedence)

    Returns:
        New merged dictionary

    Example:
        >>> base = {"a": 1, "nested": {"x": 10, "y": 20}, "items": [1, 2]}
        >>> override = {"a": 2, "nested": {"y": 25, "z": 30}, "items": [3]}
        >>> deep_merge(base, override)
        {'a': 2, 'nested': {'x': 10, 'y': 25, 'z': 30}, 'items': [3]}
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            result[key] = deep_merge(result[key], value)
        else:
            # Override takes precedence
            result[key] = value

    return result


# Bash-superset pattern. Captures three groups:
#   1: variable name (no ":" or "}")
#   2: optional modifier — "" (legacy ${VAR:default}), "-" (bash ${VAR:-default}),
#                        or "?" (bash ${VAR:?error_msg})
#   3: optional default value or error message (everything until "}")
# When the ":..." section is absent, groups 2 and 3 are both None.
VAR_PATTERN: re.Pattern[str] = re.compile(r"\$\{([^}:]+)(?::([-?]?)([^}]*))?\}")


def substitute_env_vars(
    data: Any,
    *,
    type_coerce: bool = False,
    expand_user_paths: bool = True,
    substitute_keys: bool = True,
) -> Any:
    """Recursively substitute environment variables in configuration.

    Supported syntaxes (bash superset):
    - ``${VAR}`` — required; raises ``ValueError`` if unset
    - ``${VAR:default}`` — DataKnobs legacy form; uses default if unset
    - ``${VAR:-default}`` — bash-style alias for ``${VAR:default}``
    - ``${VAR:?error_msg}`` — bash-style; when unset, raises
      ``ValueError("Required environment variable not set: <error_msg>")``
      (the variable name is used in place of ``<error_msg>`` when
      ``error_msg`` is empty)

    Substitution applies to both dict keys and values, list items,
    and top-level strings. Non-string dict keys (integers, booleans, etc.)
    pass through unchanged.

    Args:
        data: Configuration data (dict, list, string, or primitive).
        type_coerce: When ``True``, a string that is *entirely* a single
            ``${VAR}`` placeholder (e.g., ``"${PORT}"``) returns the env var
            value coerced to ``int``/``float``/``bool`` when the literal
            looks like one. Mixed-content strings (``"port=${PORT}"``) always
            return strings. Default ``False``.
        expand_user_paths: When ``True``, applies ``os.path.expanduser()`` to
            substituted strings so ``"${PATH_VAR}"`` with value ``"~/foo"``
            yields ``"/home/.../foo"``. ``os.path.expanduser`` is a no-op for
            strings that do not start with ``~``, so URLs and connection
            strings pass through unchanged. Default ``True`` (preserves
            historical behavior). Set to ``False`` for strict no-touch
            substitution.
        substitute_keys: When ``True``, dict keys with ``${VAR}`` references
            are substituted. Dict keys are never type-coerced even when
            ``type_coerce=True``. Default ``True``.

    Returns:
        Data with environment variables substituted. When ``type_coerce`` is
        ``True`` the return type for whole-value placeholders may be
        ``int``/``float``/``bool``; otherwise always returns string.

    Raises:
        RequiredEnvVarError: When a required ``${VAR}`` is unset, or when
            ``${VAR:?msg}`` is unset (the message is included in the
            exception message). The class is a subclass of ``ValueError``
            so ``except ValueError`` continues to catch it; catch
            :class:`RequiredEnvVarError` directly to inspect the
            ``var_name`` / ``bash_form`` / ``explicit_message`` attributes.

    Example:
        >>> os.environ["MY_VAR"] = "hello"
        >>> substitute_env_vars({"key": "${MY_VAR}", "default": "${MISSING:world}"})
        {'key': 'hello', 'default': 'world'}
    """
    if isinstance(data, dict):
        if substitute_keys:
            return {
                (
                    _substitute_string(k, type_coerce=False, expand_user_paths=expand_user_paths)
                    if isinstance(k, str)
                    else k
                ): substitute_env_vars(
                    v,
                    type_coerce=type_coerce,
                    expand_user_paths=expand_user_paths,
                    substitute_keys=substitute_keys,
                )
                for k, v in data.items()
            }
        return {
            k: substitute_env_vars(
                v,
                type_coerce=type_coerce,
                expand_user_paths=expand_user_paths,
                substitute_keys=substitute_keys,
            )
            for k, v in data.items()
        }
    elif isinstance(data, list):
        return [
            substitute_env_vars(
                item,
                type_coerce=type_coerce,
                expand_user_paths=expand_user_paths,
                substitute_keys=substitute_keys,
            )
            for item in data
        ]
    elif isinstance(data, str):
        return _substitute_string(
            data, type_coerce=type_coerce, expand_user_paths=expand_user_paths
        )
    else:
        return data


def _substitute_string(
    value: str,
    *,
    type_coerce: bool,
    expand_user_paths: bool,
) -> str | int | float | bool:
    """Substitute environment variables in a string.

    Args:
        value: String potentially containing ``${VAR}`` references.
        type_coerce: When ``True`` and the entire string is a single
            ``${VAR}`` placeholder, coerce the resolved value to
            ``int``/``float``/``bool`` if it looks like one.
        expand_user_paths: When ``True`` apply ``os.path.expanduser`` to the
            final string result. Applied before ``type_coerce`` in the
            whole-value fast path so the two flags compose consistently.

    Returns:
        String with variables substituted, or coerced primitive when
        ``type_coerce`` matches a whole-value placeholder.

    Raises:
        RequiredEnvVarError: If a required ``${VAR}`` is unset, or if
            the ``${VAR:?msg}`` form fires. Subclass of ``ValueError``,
            so existing ``except ValueError`` callers continue to work.
    """
    if type_coerce:
        whole = VAR_PATTERN.fullmatch(value)
        if whole is not None:
            resolved = _resolve_match(whole)
            if expand_user_paths and resolved:
                resolved = os.path.expanduser(resolved)  # noqa: PTH111 — Path(x).expanduser() collapses "://" to ":/" in URLs
            return _convert_type(resolved)

    def replacer(match: re.Match[str]) -> str:
        return _resolve_match(match)

    result = VAR_PATTERN.sub(replacer, value)
    if expand_user_paths and result:
        return os.path.expanduser(result)  # noqa: PTH111 — Path(x).expanduser() collapses "://" to ":/" in URLs
    return result


class RequiredEnvVarError(ValueError):
    """Raised by :func:`substitute_env_vars` when a required env var is unset.

    Subclass of ``ValueError`` so callers using ``except ValueError`` or
    ``pytest.raises(ValueError)`` keep working. Catch this class directly
    when you need to distinguish missing-required-var failures from other
    ``ValueError`` causes, or to inspect:

    - :attr:`var_name`: the variable name that was unset.
    - :attr:`bash_form`: ``True`` when raised by the bash-style
      ``${VAR:?msg}`` form, ``False`` when raised by the bare ``${VAR}``
      form.
    - :attr:`explicit_message`: the user-supplied message from
      ``${VAR:?msg}`` (``None`` for the bare form or empty ``${VAR:?}``).

    Library code should not construct this exception directly; it is
    raised by the canonical helper.
    """

    def __init__(
        self,
        var_name: str,
        *,
        bash_form: bool,
        explicit_message: str | None,
    ) -> None:
        self.var_name = var_name
        self.bash_form = bash_form
        self.explicit_message = explicit_message
        message = explicit_message if explicit_message else var_name
        super().__init__(f"Required environment variable not set: {message}")


def _resolve_match(match: re.Match[str]) -> str:
    """Resolve a single ``${...}`` regex match to its environment value.

    Raises:
        RequiredEnvVarError: When the variable is required and unset, or
            when the ``${VAR:?msg}`` form fires with a missing variable.
    """
    var_name = match.group(1)
    modifier = match.group(2)  # None, "", "-", or "?"
    default_or_error = match.group(3)  # None or the captured trailing text

    env_value = os.environ.get(var_name)
    if env_value is not None:
        return env_value
    if modifier == "?":
        # Empty error message ("${VAR:?}") is treated as "no message" so
        # ``RequiredEnvVarError`` falls back to the variable name; only a
        # non-empty trailing string is preserved as the explicit message.
        explicit_message = default_or_error if default_or_error else None
        raise RequiredEnvVarError(
            var_name,
            bash_form=True,
            explicit_message=explicit_message,
        )
    if default_or_error is not None:
        return default_or_error
    raise RequiredEnvVarError(var_name, bash_form=False, explicit_message=None)


def _convert_type(value: str) -> str | int | float | bool:
    """Coerce a string to ``int``/``float``/``bool`` when it looks like one.

    Used by ``substitute_env_vars(..., type_coerce=True)`` for whole-value
    placeholders only. Preserves the original string when no coercion
    applies.

    Only the unambiguous bool words ``true`` / ``false`` / ``yes`` / ``no``
    (case-insensitive) coerce to ``bool``. Numeric strings such as ``"0"``
    and ``"1"`` coerce to ``int`` — bash conflates them with bool, but
    treating ``"0"`` as ``False`` surprises callers expecting an integer
    port / count / size.
    """
    lower = value.lower()
    if lower in ("true", "yes"):
        return True
    if lower in ("false", "no"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


class InheritableConfigLoader:
    """Configuration loader with inheritance support.

    Loads YAML/JSON configuration files with support for configuration
    inheritance via an `extends` field. Child configurations override
    parent values through deep merge.

    Configuration *names* are mapped to locations under ``config_dir`` by
    :meth:`resolve_name`, which a deployment can govern -- see that method
    and the ``resolver`` argument. That mapping is one-way, so a deployment
    that governs it also has to say which names exist: see
    :meth:`available_names`.

    A loader mutates instance state while :meth:`load_from_file` runs
    (``config_dir`` and resolution suppression are saved and restored around
    the call), so one loader is not safe to share across threads.

    Attributes:
        config_dir: Directory containing configuration files
        _cache: Resolved configurations, keyed by (resolved name,
            substitution mode)
        _resolver: Optional name->location mapping consulted by
            :meth:`resolve_name`
    """

    def __init__(
        self,
        config_dir: str | Path | None = None,
        *,
        resolver: ResourceResolver[str, str] | None = None,
        allow_outside: bool = False,
    ):
        """Initialize configuration loader.

        Args:
            config_dir: Directory containing configuration files.
                       If None, uses ./configs
            resolver: Optional name->location mapping consulted by
                     :meth:`resolve_name`. ``MappingResolver`` and
                     ``CallableResolver`` from ``dataknobs_common`` cover the
                     common layout conventions without a consumer class.
            allow_outside: Opt this loader out of the containment bound, for
                     a deployment whose layout genuinely spans sibling trees
                     (``configs/app.yaml`` with ``extends: ../shared/base``).
                     Off by default, and it applies to every name the loader
                     resolves -- the requested config, each ``extends:``
                     target, and a resolver's output alike, because they
                     reach the same join. A name that actually escapes is
                     logged at WARNING when it does, so the widened boundary
                     is auditable in a deployment's logs rather than silent;
                     a contained name logs nothing.
        """
        self.config_dir = Path(config_dir) if config_dir else Path("./configs")
        self._resolver = resolver
        self._allow_outside = allow_outside
        # An override *replaces* `resolve_name`, so a loader given both modes
        # ignores this resolver unless the override delegates to `super()`.
        # Silence is the whole problem -- the loader then reads a different
        # file than the caller configured, with no diagnostic -- so say it
        # once, here, where both inputs are visible. A warning rather than an
        # error: overriding to normalize or log *and* delegating is a
        # legitimate use of both, and raising would break it.
        if resolver is not None and type(self).resolve_name is not (
            InheritableConfigLoader.resolve_name
        ):
            warnings.warn(
                f"{type(self).__name__} overrides resolve_name(), which replaces "
                "the implementation that consults `resolver`. The injected "
                "resolver is ignored unless the override delegates to "
                "super().resolve_name(), in which case both mappings apply in "
                "sequence. These are alternatives, not layers -- pick one.",
                stacklevel=2,
            )
        # Suppresses name resolution for the duration of `load_from_file`,
        # which rebinds config_dir out from under any layout convention.
        self._bypass_resolution = False
        # Keyed by (resolved name, substitute_vars). Resolving `extends:`
        # recurses with substitute_vars=False, so one config can be produced in
        # two forms; the key records which. See `load` for why storing both
        # under one key makes a config's value depend on load order.
        #
        # The name half is the *resolved* name, so two spellings of one config
        # are one entry rather than two copies that can disagree.
        self._cache: dict[tuple[str, bool], dict[str, Any]] = {}
        # Resolved names, so two spellings of one config are one node in the
        # cycle graph rather than two.
        self._loading: set[str] = set()  # Track configs being loaded to detect cycles
        # resolved parent name -> resolved names that reached it through
        # `extends:`. A cached child holds its parent's content merged in, so
        # clearing the parent has to reach the children or the stale copy keeps
        # answering. Resolved on both sides to stay in the cache's namespace --
        # raw keys here would make the walk compute names `_cache` cannot match.
        self._dependents: dict[str, set[str]] = {}

    def resolve_name(self, name: str) -> str:
        """Map a configuration name to a name/path relative to ``config_dir``.

        Applied to the requested configuration AND to every ``extends:``
        target, so a layout convention governs inheritance as well as entry
        points -- a parent named bare inside a child still resolves.

        Default: consult the injected ``resolver``; fall back to identity when
        there is none or when it returns ``None`` (the ``ResourceResolver``
        contract for "no mapping").

        Subclasses may override this method instead of injecting a resolver.
        The two modes are **alternatives, not layers** -- an override replaces
        this implementation, so a loader given both ignores the injected
        resolver entirely unless the override delegates to
        ``super().resolve_name(...)``, in which case both mappings apply in
        sequence. Neither is likely to be what was meant; pick one mode. A
        loader constructed with both warns, since the first of those two
        outcomes is otherwise silent.

        Not applied under :meth:`load_from_file`, which rebinds ``config_dir``
        to the file's own directory; a ``config_dir``-relative convention
        cannot be correct against a ``config_dir`` the caller did not choose.

        A mapping decides *where inside* ``config_dir`` a name lives; it does
        not widen what a name may address. A resolved name that *lands*
        outside ``config_dir`` -- whether spelled with ``..`` or as an
        absolute path -- raises
        :class:`InheritanceError` when the load reaches the filesystem -- the
        bound applies to the resolver's output exactly as it does to a name a
        caller wrote.

        Args:
            name: Configuration name as written by the caller or in ``extends:``

        Returns:
            The name to look up under ``config_dir``, which must be inside it

        Example:
            ```python
            from dataknobs_common import CallableResolver

            loader = InheritableConfigLoader(
                root, resolver=CallableResolver(lambda n: f"domains/{n}")
            )
            ```
        """
        if self._resolver is None:
            return name
        resolved = self._resolver.resolve(name)
        return name if resolved is None else resolved

    def _resolved(self, name: str) -> str:
        """The name to operate on: resolved, or left alone under bypass.

        Every site that turns a caller's or an ``extends:`` name into the one
        this loader keys and reads by goes through here, so the suppression
        :meth:`load_from_file` sets is honored by construction rather than by
        two copies of the check agreeing. It is also why a subclass overriding
        :meth:`resolve_name` cannot defeat that suppression: under bypass the
        override is not called at all.
        """
        return name if self._bypass_resolution else self.resolve_name(name)

    def load(
        self,
        name: str,
        use_cache: bool = True,
        substitute_vars: bool = True,
    ) -> dict[str, Any]:
        """Load and resolve configuration with inheritance.

        ``name`` is mapped through :meth:`resolve_name` once, at the top, and
        the result is what keys the cache, the cycle-detection set and the
        inheritance edges, and what names the file to read.

        The cache is keyed by that resolved name **and** substitution mode.
        Resolving `extends:` recurses with ``substitute_vars=False``, so the
        same config can be produced in two forms; a shared key would let one
        serve a request for the other, in both directions -- returning raw
        ``${VAR}`` placeholders where expansion was asked for, or expanding an
        already-expanded value a second time.

        Args:
            name: Configuration name (without extension). May address a
                subdirectory of ``config_dir`` but not leave it -- see
                :meth:`resolve_name`
            use_cache: Whether to use cached configuration if available
            substitute_vars: Whether to substitute environment variables

        Returns:
            Resolved configuration dictionary

        Raises:
            InheritanceError: If the name -- or any ``extends:`` target it
                reaches -- addresses a file outside ``config_dir``, if the
                config is not found, if a cycle is detected, or on any other
                inheritance error

        Example:
            ```python
            loader = InheritableConfigLoader("./configs")
            config = loader.load("my-domain")
            ```
        """
        resolved = self._resolved(name)
        cache_key = (resolved, substitute_vars)

        # Check cache
        if use_cache and cache_key in self._cache:
            logger.debug("Using cached config: %s", resolved)
            return self._cache[cache_key]

        # Detect circular inheritance
        if resolved in self._loading:
            raise InheritanceError(f"Circular inheritance detected: {resolved}")

        self._loading.add(resolved)

        try:
            # Load raw configuration
            raw_config = self._load_file(resolved)

            # Handle inheritance
            if raw_config.get("extends"):
                parent_name = raw_config["extends"]
                logger.debug("Config '%s' extends '%s'", resolved, parent_name)

                # Load parent configuration (recursively handles inheritance)
                parent_config = self.load(parent_name, use_cache=use_cache, substitute_vars=False)

                # Record the edge so clearing the parent reaches this child --
                # but only when this load is taking part in the cache at all.
                # A bypassing load stores nothing for an edge to invalidate,
                # and `load_from_file` bypasses with `config_dir` rebound, so
                # recording there would file an edge under a bare name from
                # another directory and over-clear the configured one.
                #
                # Edges accumulate and are never pruned, so editing a config
                # to drop its `extends:` leaves the old edge in place until
                # the process restarts. That over-clears rather than
                # under-clears -- the cost is a cache miss, and the next load
                # is correct -- which is the safe direction for an
                # invalidation graph to be imprecise in.
                #
                # Keyed on the *resolved* parent, to stay in `_cache`'s
                # namespace. That means resolving `parent_name` here as well
                # as in the recursion above -- two invocations on the same
                # input, which a `ResourceResolver` is required to answer
                # identically, and not the same thing as applying the mapping
                # to an already-mapped name. A resolver that does real work
                # per call pays for both; `CachedResolver` from
                # `dataknobs_common` is the remedy.
                if use_cache:
                    self._dependents.setdefault(self._resolved(parent_name), set()).add(resolved)

                # Deep merge: child overrides parent
                raw_config = deep_merge(parent_config, raw_config)

                # Remove extends field from final config
                raw_config.pop("extends", None)

            # Substitute environment variables
            if substitute_vars:
                raw_config = substitute_env_vars(raw_config)

            # Cache the result. Gated on use_cache: bypassing the cache means
            # not taking part in it at all, in both directions. `validate` is
            # a dry run, and `load_from_file` reads with config_dir rebound to
            # another directory -- the key carries no directory, so a write
            # from there would answer later reads for the configured one.
            if use_cache:
                self._cache[cache_key] = raw_config
            logger.info("Loaded configuration: %s", resolved)

            return raw_config

        finally:
            self._loading.discard(resolved)

    def load_from_file(
        self,
        filepath: str | Path,
        substitute_vars: bool = True,
    ) -> dict[str, Any]:
        """Load configuration from a specific file path.

        This method bypasses the config_dir and loads directly from the path.
        Inheritance is resolved relative to the file's directory.

        :meth:`resolve_name` is **suppressed for the whole subtree** -- the
        entry file and every ``extends:`` target below it. A name mapping is
        defined relative to ``config_dir``, and this method rebinds
        ``config_dir`` to the file's own directory, so applying the mapping
        here would look for the convention's location underneath a directory
        the caller chose instead.

        Args:
            filepath: Path to configuration file
            substitute_vars: Whether to substitute environment variables

        Returns:
            Resolved configuration dictionary

        Raises:
            InheritanceError: If file not found or other error
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise InheritanceError(f"Configuration file not found: {filepath}")

        # Temporarily change config_dir to file's directory for inheritance,
        # and suppress name resolution for as long as it is rebound. The flag
        # is read in `_resolved`, which every resolution site goes through, so
        # a subclass overriding `resolve_name` cannot defeat the suppression --
        # the override is simply not called.
        #
        # Both are saved and restored rather than set and hard-cleared, so the
        # pair stays correct if this ever runs nested. Hard-clearing the flag
        # would be equivalent today and wrong the moment it is not, and the
        # failure is silent: the layout convention stops applying for every
        # later load, which surfaces as file-not-found on an unresolved path.
        old_config_dir = self.config_dir
        old_bypass = self._bypass_resolution
        self.config_dir = filepath.parent
        self._bypass_resolution = True

        try:
            return self.load(filepath.stem, use_cache=False, substitute_vars=substitute_vars)
        finally:
            self.config_dir = old_config_dir
            self._bypass_resolution = old_bypass

    def _load_file(self, name: str) -> dict[str, Any]:
        """Load raw configuration file.

        ``name`` is the final name :meth:`load` settled on -- resolved, or
        deliberately left alone under :meth:`load_from_file`. This method does
        not resolve it.

        The name is joined to ``config_dir`` and has to stay inside it. A name
        may address a subdirectory (``domains/child``), which is how a layout
        convention is expressed; one that *lands* outside -- whether spelled
        with ``..`` or as an absolute path --
        raises rather than reading the file it points at. The bound applies to
        every name that reaches here, and most of them are not the caller's
        literal: an ``extends:`` value comes out of a config file, and a
        resolved name comes from a consumer-supplied resolver.

        Args:
            name: Resolved configuration name (without extension)

        Returns:
            Parsed configuration dictionary

        Raises:
            InheritanceError: If the name addresses a file outside
                ``config_dir``, the file is not found, or it fails to parse
        """
        try:
            filepath = find_config_file(self.config_dir, name, allow_outside=self._allow_outside)
        except ConfigPathEscapeError as e:
            raise InheritanceError(str(e)) from e
        if filepath is None:
            raise InheritanceError(
                f"Configuration file not found: {name}.yaml, {name}.yml, "
                f"or {name}.json in {self.config_dir}"
            )

        try:
            return load_yaml_or_json(filepath)
        except ConfigLoadError as e:
            raise InheritanceError(str(e)) from e
        except OSError as e:
            raise InheritanceError(f"Failed to read configuration file {filepath}: {e}") from e

    def clear_cache(self, name: str | None = None) -> None:
        """Clear configuration cache.

        Pass the name you passed :meth:`load`. It is mapped through
        :meth:`resolve_name` the same way, so clearing a config clears what
        loading it stored, and two names the resolver maps together clear each
        other. Passing an already-resolved name instead maps it a second time:
        harmless for a lookup-table resolver, which leaves a name it has no
        entry for alone, but a prefixing one double-prefixes and the call
        clears nothing. Nothing is raised -- the debug log reports the names
        it targeted *and* how many cached entries that removed, and a clear
        that removed none is the sign.

        Clearing a name clears every substitution variant cached under it --
        the config, not one of the two forms it may have been stored in.
        Leaving a variant behind would re-create the load-order dependence the
        keying exists to prevent.

        It also clears, transitively, every config that reached this one
        through ``extends:``. A cached child holds its parent's content merged
        in, so clearing only the parent leaves that copy answering -- the
        staleness the call was made to resolve, surviving the call.
        Invalidation runs down the inheritance edges, never up: a child's
        parent is unaffected.

        Args:
            name: Specific config to clear, or None to clear all
        """
        if name:
            # Walk the recorded edges with a seen-set: the edges are data
            # recorded across loads, so a config edited to extend its own
            # descendant between two loads would otherwise loop here.
            stale: set[str] = set()
            # `resolve_name`, not `_resolved`: this walks `_cache`, and that
            # is always the resolved namespace. A bypassing load writes
            # nothing to it -- `load_from_file` passes `use_cache=False` --
            # so there is no bypassed entry here to find, and honoring the
            # flag would only mean failing to clear a resolved one.
            pending = [self.resolve_name(name)]
            while pending:
                current = pending.pop()
                if current in stale:
                    continue
                stale.add(current)
                pending.extend(self._dependents.get(current, ()))

            removed = [k for k in self._cache if k[0] in stale]
            for key in removed:
                del self._cache[key]
            for gone in stale:
                self._dependents.pop(gone, None)
            # The count, not just the names. The names are what was targeted,
            # and a call that targets a name nothing is cached under removes
            # nothing -- so naming them alone reads identically whether the
            # call worked, which is useless for the one failure this log is
            # here to expose: a doubly-resolved name clearing nothing.
            logger.debug(
                "Cleared %d cache entries for: %s",
                len(removed),
                ", ".join(sorted(stale)),
            )
        else:
            self._cache.clear()
            self._dependents.clear()
            logger.debug("Cleared all cached configurations")

    def available_names(self) -> list[str]:
        """The names :meth:`load` accepts, for this deployment's layout.

        Default: the stems of the files directly under ``config_dir``, which
        is the set of loadable names only while :meth:`resolve_name` is
        identity. A deployment that governs the name->location mapping has to
        govern this too, because the mapping is one-way: a resolver answers
        "where does this name live", and nothing can run it backwards to
        recover the names from the locations.

        Override this alongside :meth:`resolve_name`. Leaving it alone under a
        resolver does not raise -- it reports the wrong thing quietly. Under
        the layout the ``resolve_name`` example describes, every config is a
        directory down and the default returns ``[]``, so the natural
        ``for name in ...: load(name)`` loop runs zero times. A layout that
        mixes depths is worse than empty: the stems it does find are
        *locations*, and mapping a location through ``resolve_name`` addresses
        something else again.

        Returns:
            Configuration names, each valid to pass to :meth:`load`

        Example:
            ```python
            class DomainLoader(InheritableConfigLoader):
                def resolve_name(self, name: str) -> str:
                    return f"domains/{name}"

                def available_names(self) -> list[str]:
                    return self.stems_in(self.config_dir / "domains")
            ```
        """
        return self.stems_in(self.config_dir)

    @staticmethod
    def stems_in(directory: Path) -> list[str]:
        """Loadable names from one directory's files, deduplicated and sorted.

        The default :meth:`available_names` is this, applied to
        ``config_dir``; it is public because an override is the same thing
        applied to a different directory, and rewriting it by hand is quiet
        to get wrong. Globbing only ``*.yaml`` enumerates a subset of what
        :meth:`load` accepts -- nothing raises, the ``.json`` configs just
        never come up, and they are loadable the whole time. The extensions
        here are the ones ``load`` itself probes, read from the one shared
        list, so enumeration cannot fall behind loading.

        Args:
            directory: Where to look; a missing one yields no names

        Returns:
            File stems, sorted, with a name present in more than one
            supported extension reported once
        """
        if not directory.exists():
            return []

        return sorted(
            {
                path.stem
                for extension in DEFAULT_CONFIG_EXTENSIONS
                for path in directory.glob(f"*{extension}")
                if path.is_file()
            }
        )

    def list_available(self) -> list[str]:
        """List all available configuration names.

        Delegates to :meth:`available_names`, which is the method to override.

        Returns:
            List of configuration names (without extensions)
        """
        return self.available_names()

    def validate(self, name: str) -> tuple[bool, str | None]:
        """Validate a configuration file.

        Args:
            name: Configuration name

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            self.load(name, use_cache=False)
            return True, None
        except InheritanceError as e:
            return False, str(e)
        except ValueError as e:
            return False, str(e)


def load_config_with_inheritance(
    filepath: str | Path,
    substitute_vars: bool = True,
) -> dict[str, Any]:
    """Convenience function to load a config file with inheritance.

    Args:
        filepath: Path to configuration file
        substitute_vars: Whether to substitute environment variables

    Returns:
        Resolved configuration dictionary

    Example:
        ```python
        config = load_config_with_inheritance("configs/my-domain.yaml")
        ```
    """
    loader = InheritableConfigLoader()
    return loader.load_from_file(filepath, substitute_vars=substitute_vars)
