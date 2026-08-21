"""Environment-aware configuration with late-binding resource resolution.

This module provides the EnvironmentAwareConfig class that supports:
- Logical resource references resolved per-environment
- Late-binding of environment variables (at instantiation time, not load time)
- Separation of portable app config from infrastructure bindings

Substitution rule: **substitute each source exactly once, at the latest point
that source is still separable.** For the environment that is its load; for
the app config it is entry to :meth:`EnvironmentAwareConfig.resolve_for_build`,
which is still before resource refs are spliced in. Once spliced, the two are
merged beyond distinguishing, and a pass over the result would expand
environment values a second time — reinterpreting the *content* of a value as
a template. :attr:`EnvironmentConfig.substituted` is what makes the
environment's provenance knowable rather than guessed.

Two things are separable for longer than that, and so are expanded later: an
environment resource, which is expanded as it is spliced rather than with the
environment as a whole, and a reference's inline defaults, which the splice
discards wherever the environment supplies the key. Both follow the same rule
— expanding either earlier reads values the build then throws away, and an
unset required ``${VAR}`` among them aborts a build that never used it.

That makes a splice, not the entry point, the place expansion happens, and a
splice merges two sources whose provenances differ. So each source is
expanded and resolved on its own terms *before* the merge
(:func:`_resolve_source`) rather than the merged result being walked once
afterwards: one flag cannot be true of both halves, and walking the result
under it would either expand an environment's values a second time or leave a
default's nested refs raw.

A ``$resource`` block nested inside either source is held to one expansion
too, though not in the same place. Carried by an inline default — or by a
resource an *unsubstituted* environment supplies — it reaches its own splice
raw, because the pass over that source defers it, and is expanded there.
Carried by a resource an already substituted environment supplies, it was
expanded at that environment's load, and ``substituted`` is what stops the
splice expanding it a second time.

Example:
    ```python
    # Load with auto-detected environment
    config = EnvironmentAwareConfig.load_app(
        "my-bot",
        app_dir="config/apps",
        env_dir="config/environments"
    )

    # Get resolved config for object building (late binding happens here)
    resolved = config.resolve_for_build()

    # Get portable config for storage (no env vars resolved)
    portable = config.get_portable_config()
    ```

App config format (config/apps/my-bot.yaml):
    ```yaml
    name: my-bot
    version: "1.0.0"

    bot:
      llm:
        $resource: default
        type: llm_providers
        temperature: 0.7

      conversation_storage:
        $resource: conversations
        type: databases
    ```
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataknobs_common.config_loading import (
    ConfigLoadError,
    ConfigPathEscapeError,
    find_config_file,
    load_yaml_or_json,
)

from .environment_config import (
    STRICT_RESOURCES_SETTING,
    EnvironmentConfig,
    ResourceNotFoundError,
    parse_strictness_flag,
)
from .exceptions import ConfigError
from .inheritance import substitute_env_vars

logger = logging.getLogger(__name__)

#: Keys of a ``$resource`` block that select and constrain the resource
#: rather than supplying a default for it. Everything else in the block is an
#: inline default, kept raw by the entry pass and expanded at the splice.
#:
#: The set is **closed and enforced**: a ``$``-prefixed key outside it is a
#: malformed reference, not an inline default. It has to be, because the
#: comprehension that builds the defaults takes everything this set does not,
#: so a misspelled marker is otherwise promoted to a default and handed to a
#: factory as a keyword argument -- which is how ``$requred: true`` would
#: silently mean *not required*.
RESOURCE_MARKER_KEYS = frozenset({"$resource", "type", "$requires", "$required"})

#: The markers that say nothing on their own. ``type`` is an ordinary word and
#: appears all over a config; these two exist only to qualify a ``$resource``,
#: so finding one without it means the selector key is the misspelled one --
#: the single typo the closed set above cannot catch, because the guard it
#: feeds fires on a block that already contains ``$resource``.
_POLICY_MARKER_KEYS = frozenset({"$requires", "$required"})


def _render_path(path: str) -> str:
    """A dotted config path for a message, naming the root rather than "".

    ``UnresolvedResourceRef.path`` keeps the empty string -- it is a dotted
    path of zero segments, and a caller matching on prefixes needs it to stay
    one. A sentence needs a noun.
    """
    return path or "<root>"


def _validate_reference_markers(reference: Mapping[str, Any], *, path: str) -> None:
    """Reject a ``$``-prefixed key in a reference block that is not a marker.

    Runs on both the found and the missing path. A malformed reference is
    malformed in every environment, and catching it only where the resource
    happens to be absent would surface it first in whichever deployment is
    least equipped to read the message.
    """
    unknown = sorted(
        key
        for key in reference
        if isinstance(key, str) and key.startswith("$") and key not in RESOURCE_MARKER_KEYS
    )
    if not unknown:
        return

    markers = ", ".join(sorted(key for key in RESOURCE_MARKER_KEYS if key.startswith("$")))
    raise ConfigError(
        f"Unknown marker key(s) {unknown} in the $resource reference for "
        f"'{reference.get('$resource')}' at config path '{_render_path(path)}'. "
        f"A $-prefixed key must be one of: {markers}. Everything else in the "
        f"block is an inline default, so an unrecognised marker would otherwise "
        f"be passed to a factory as a keyword argument rather than rejected."
    )


def _validate_orphaned_markers(block: Mapping[str, Any], *, path: str) -> None:
    """Reject ``$required`` / ``$requires`` on a block with no ``$resource``.

    The closed marker set catches a typo in every marker but one. Its guard
    runs on a block that *is* a reference, and what makes a block a reference
    is the ``$resource`` key -- so a typo in that key produces an ordinary
    dict, which resolves to itself and reaches the factory with its markers
    still attached. ``$resorce: conversations`` is the shape, and it is the
    same silent degrade as the rest of the class.

    A leftover policy marker is what gives it away, and it is specific enough
    to act on: neither means anything except on a reference.
    """
    orphaned = sorted(key for key in block if key in _POLICY_MARKER_KEYS)
    if not orphaned:
        return

    raise ConfigError(
        f"Marker key(s) {orphaned} at config path '{_render_path(path)}' on a "
        f"block with no `$resource` key. They qualify a resource reference and "
        f"mean nothing without one, so this is a misspelled `$resource` -- the "
        f"block resolves to itself and reaches a factory with its markers "
        f"attached. Keys present: {sorted(str(key) for key in block)}."
    )


def _parse_requires(value: Any, *, where: str) -> list[str]:
    """Read ``$requires`` as a list of capability names.

    Its sibling ``$required`` takes a scalar, which makes ``$requires:
    persistence`` the natural slip. Unvalidated it is merely truthy, so it
    survived into two messages that read as nonsense -- an absent resource
    failed for ``['p', 'e', 'r', 's', ...]``, and a present one for eight
    missing single-character capabilities.
    """
    if value is None:
        return []
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ConfigError(
            f"{where} must be a list of capability names, got {value!r}. A bare "
            f"string is iterated character by character, so the check would run "
            f"against letters rather than capabilities."
        )
    non_strings = [item for item in value if not isinstance(item, str)]
    if non_strings:
        raise ConfigError(
            f"{where} must contain only capability names, got {non_strings!r} "
            f"among {list(value)!r}. Capabilities are matched against the "
            f"resource's `capabilities` list by value."
        )
    return list(value)


@dataclass(frozen=True)
class _ParsedReference:
    """A ``$resource`` block read once, for whoever walks it.

    The build and the survey used to parse a reference each -- the same five
    lines twice, one of them the marker guard -- and a shared format read in
    two places is a format with two definitions. Everything that decides what
    a reference *means* is settled here; what the callers differ on is what to
    do about a resource that is absent.
    """

    name: str
    resource_type: str
    defaults: dict[str, Any]
    requires: list[str]
    declared_required: bool | None


def _parse_reference(block: Mapping[str, Any], *, path: str) -> _ParsedReference:
    """Validate a reference block and split it into markers and defaults.

    ``$required`` is parsed here rather than on the branch that reads it, for
    the reason the marker guard runs here: a malformed value is malformed in
    every environment, and deferring the parse would surface it first in
    whichever deployment lacks the resource.
    """
    _validate_reference_markers(block, path=path)
    name = block["$resource"]
    return _ParsedReference(
        name=name,
        resource_type=block.get("type", "default"),
        # Everything the marker set does not claim is an inline default. That
        # is why the set has to be closed: this comprehension is what would
        # otherwise promote a misspelled marker to a factory keyword argument.
        defaults={k: v for k, v in block.items() if k not in RESOURCE_MARKER_KEYS},
        requires=_parse_requires(
            block.get("$requires"),
            where=f"`$requires` on the reference to '{name}' at config path '{_render_path(path)}'",
        ),
        declared_required=(
            parse_strictness_flag(
                block["$required"],
                where=f"`$required` on the reference to '{name}' at config path "
                f"'{_render_path(path)}'",
            )
            if "$required" in block
            else None
        ),
    )


def _reference_is_required(
    *,
    declared: bool | None,
    requires: Sequence[Any],
    resolver_default: bool | None,
    environment: EnvironmentConfig,
) -> tuple[bool, str]:
    """Decide whether a missing resource fails, and say which lever decided.

    The chain, most specific first. Each level is ``None``-means-defer, so
    "explicitly false" and "unspecified" stay distinguishable, and each level
    is the only one its owner can reach:

    1. the reference's ``$required``            -- the reference author
    2. its non-empty ``$requires``              -- the reference author
    3. ``strict_resources=`` on the resolution  -- the calling code, then the
       embedding application (collapsed by the caller, since both are code)
    4. the environment's ``strict_resources``   -- the operator
    5. ``False``                                -- unchanged default

    ``$requires`` sits *above* the code levels rather than below them because
    it is a claim about this reference in particular: a resolver-wide
    ``strict_resources=False`` is a statement about references that did not
    say anything, and this one did. Only the same author's ``$required:
    false`` overrides it -- "if it is there it must do X; it may be absent" is
    coherent, so it is expressible.

    Returns:
        ``(required, why)``. ``why`` is a phrase naming the level that made it
        strict, for the failure message; it is empty when ``required`` is
        False. An operator reading the failure needs to know which lever
        produced it in order to choose the right response.
    """
    if declared is not None:
        return (declared, "it declares `$required: true`" if declared else "")

    if requires:
        return (
            True,
            f"it declares `$requires: {list(requires)}`, which a resource that "
            f"is absent cannot satisfy",
        )

    if resolver_default is not None:
        if resolver_default:
            return (True, "this resolution ran with `strict_resources=True`")
        return (False, "")

    setting = environment.get_setting(STRICT_RESOURCES_SETTING)
    if setting is not None:
        # Normally already parsed once, at construction -- see
        # :meth:`EnvironmentConfig.__post_init__`. This still parses rather
        # than assuming, because the dataclass is public and mutable: a
        # setting written into ``settings`` after construction has been
        # through no guard at all.
        strict = parse_strictness_flag(
            setting,
            where=(f"Setting '{STRICT_RESOURCES_SETTING}' in environment '{environment.name}'"),
        )
        if strict:
            return (
                True,
                f"environment '{environment.name}' sets `{STRICT_RESOURCES_SETTING}: true`",
            )
        return (False, "")

    return (False, "")


@dataclass(frozen=True)
class UnresolvedResourceRef:
    """A ``$resource`` reference whose resource the environment does not define.

    Reported by :meth:`EnvironmentAwareConfig.find_unresolved_resources`, which
    surveys rather than resolves: an operator auditing a config tree wants
    every unresolvable reference in one pass, not one per run.

    Attributes:
        path: Dotted path to the reference within the config, list items
            spelled ``[0]`` -- e.g. ``bot.knowledge_base.vector_store``.
        resource_type: The reference's ``type`` (``"default"`` when absent).
        resource_name: The reference's ``$resource``, **after** ``${VAR}``
            expansion, so a reference that selects its resource by variable is
            reported under the name it would actually look up.
        required: The effective policy for this reference, per the precedence
            chain. ``True`` means a build would raise here rather than degrade.
        has_inline_defaults: Whether the reference declares any inline
            defaults. This distinguishes the two degradations: falling back to
            declared defaults is a config that still builds, while falling
            back to nothing is a factory about to be called with no arguments.
    """

    path: str
    resource_type: str
    resource_name: str
    required: bool
    has_inline_defaults: bool


def _substitute_deferring_defaults(config: Any, *, defer_defaults: bool = True) -> Any:
    """Expand ``${VAR}`` refs, holding ``$resource`` inline defaults back.

    Run over each source as it enters resolution — the app config on the way
    in, and every value a splice carries. Markers are always expanded, so a
    reference can select its resource by variable.

    Inline defaults are held back for two reasons that point the same way.
    The splice discards every one the environment supplies, so expanding one
    here reads a value the build then throws away — and an unset required
    ``${VAR}`` among them aborts a build that never used it. And the splice
    expands each default that survives it, so one expanded here would be
    expanded a second time there. Substitution is not idempotent: the second
    pass re-reads the first pass's output, so a value whose own text contains
    ``${...}`` is treated as a template.

    Holding a default back therefore keeps every value at exactly one
    expansion, performed at the splice that proves it survived.

    A default's *key* is expanded here regardless, because what proves the
    default survived is the key: the splice asks ``if key not in resolved``.
    Deferring it would only move the same expansion to the splice, which
    expands every default's key there to ask the question. So an unset
    required ``${VAR}`` in key position does abort the build even when the
    environment supplies that key — a value's survival is decided by
    something else, and a key's is decided by itself.

    ``defer_defaults=False`` expands everything, for the caller that will not
    splice at all. With no splice nothing discards a default and nothing else
    would expand it, so deferring there would strand a raw ``${VAR}``.
    """
    if isinstance(config, dict):
        deferring = defer_defaults and "$resource" in config
        substituted: dict[Any, Any] = {}
        for declared_key, value in config.items():
            # :func:`substitute_env_vars` expands keys as well as values, and
            # this is a wrapper around it. Re-walking the structure here and
            # calling it only at the leaves would keep everything it does to a
            # value and silently drop everything it does at a container --
            # which is how expanding keys got lost. Handing the key back to it
            # is also what passes a non-string key through untouched.
            key = substitute_env_vars(declared_key)
            if deferring:
                substituted[key] = (
                    substitute_env_vars(value) if key in RESOURCE_MARKER_KEYS else value
                )
            else:
                substituted[key] = _substitute_deferring_defaults(
                    value, defer_defaults=defer_defaults
                )
        return substituted
    if isinstance(config, list):
        return [
            _substitute_deferring_defaults(item, defer_defaults=defer_defaults) for item in config
        ]
    return substitute_env_vars(config)


def _child_path(path: str, key: Any) -> str:
    """Extend a dotted config path by one mapping key."""
    return f"{path}.{key}" if path else str(key)


def _walk(
    config: Any,
    environment: EnvironmentConfig,
    *,
    substitute: bool,
    strict_resources: bool | None,
    path: str,
    active: list[tuple[str, str]],
    survey: list[UnresolvedResourceRef] | None,
) -> Any:
    """Resolve every ``$resource`` reference in ``config``, recursively.

    **One traversal, two modes.** A preflight and a build have to reach the
    same references or the preflight is not one, and the way to guarantee that
    is not to write two walks carefully -- it is to write one. The two that
    existed drifted in both directions before either was released: the survey
    descended into inline defaults the build discards, and stopped short of
    ones the build reaches.

    ``survey`` is the whole of the difference. When it is ``None`` this
    resolves for a build: a missing resource raises under a strict policy and
    warns under a lenient one. When it is a list, a missing resource is
    appended to it and resolution continues down the lenient path -- so the
    walk reaches exactly what a build reaches, and reaches it the same way.
    Every *other* failure (a malformed reference, a cycle, a capability a
    present resource does not declare) raises in both modes, because a survey
    that reported a tree sound while the build raises on it would be worse
    than no survey at all.

    Args:
        config: The tree to walk
        environment: Environment resolved against
        substitute: Whether this source still needs ``${VAR}`` expansion, per
            :func:`_resolve_source`
        strict_resources: The collapsed code-level policy, forwarded through
            every recursion: a missed forward is silent, reverting a nested
            reference to leniency inside an otherwise strict resolution
        path: Dotted config path of ``config``, for messages and findings
        active: Resource identities currently being expanded, innermost last.
            The cycle guard; see :func:`_splice_reference`.
        survey: Collector for unresolvable references, or ``None`` to build

    Returns:
        The tree with every reference replaced by its resolved config
    """
    if isinstance(config, dict):
        if "$resource" in config:
            return _splice_reference(
                config,
                environment,
                substitute=substitute,
                strict_resources=strict_resources,
                path=path,
                active=active,
                survey=survey,
            )
        _validate_orphaned_markers(config, path=path)
        return {
            key: _walk(
                value,
                environment,
                substitute=substitute,
                strict_resources=strict_resources,
                path=_child_path(path, key),
                active=active,
                survey=survey,
            )
            for key, value in config.items()
        }
    if isinstance(config, list):
        return [
            _walk(
                item,
                environment,
                substitute=substitute,
                strict_resources=strict_resources,
                path=f"{path}[{index}]",
                active=active,
                survey=survey,
            )
            for index, item in enumerate(config)
        ]
    return config


def _splice_reference(
    block: Mapping[str, Any],
    environment: EnvironmentConfig,
    *,
    substitute: bool,
    strict_resources: bool | None,
    path: str,
    active: list[tuple[str, str]],
    survey: list[UnresolvedResourceRef] | None,
) -> Any:
    """Replace one reference block with the config it resolves to."""
    reference = _parse_reference(block, path=path)

    if not environment.has_resource(reference.resource_type, reference.name):
        resolved = _degrade_to_defaults(
            reference,
            environment,
            substitute=substitute,
            strict_resources=strict_resources,
            path=path,
            active=active,
            survey=survey,
        )
    else:
        resolved = _splice_found_resource(
            reference,
            environment,
            substitute=substitute,
            strict_resources=strict_resources,
            path=path,
            active=active,
            survey=survey,
        )

    # Validate $requires against capabilities metadata. On the degraded path
    # this is reached only when the reference declared `$required: false` --
    # absence otherwise fails above, since a resource that is not there
    # satisfies no capability at all. Where the author did opt out, the check
    # still runs against whatever capabilities the inline defaults declare:
    # "it may be absent" is not "and anything will do".
    if reference.requires and isinstance(resolved, dict):
        declared_capabilities = resolved.get("capabilities")
        if declared_capabilities is not None:
            missing = set(reference.requires) - set(declared_capabilities)
            if missing:
                raise ConfigError(
                    f"Resource '{reference.name}' at config path "
                    f"'{_render_path(path)}' is missing required capabilities: "
                    f"{sorted(missing)}. Declared: {declared_capabilities}"
                )

    return resolved


def _degrade_to_defaults(
    reference: _ParsedReference,
    environment: EnvironmentConfig,
    *,
    substitute: bool,
    strict_resources: bool | None,
    path: str,
    active: list[tuple[str, str]],
    survey: list[UnresolvedResourceRef] | None,
) -> Any:
    """Handle a reference whose resource this environment does not define.

    The one place the two modes differ, and the reason they are one walk
    everywhere else.
    """
    required, why = _reference_is_required(
        declared=reference.declared_required,
        requires=reference.requires,
        resolver_default=strict_resources,
        environment=environment,
    )

    if survey is not None:
        survey.append(
            UnresolvedResourceRef(
                path=path,
                resource_type=reference.resource_type,
                resource_name=reference.name,
                required=required,
                has_inline_defaults=bool(reference.defaults),
            )
        )
    elif required:
        raise ResourceNotFoundError(
            f"Resource '{reference.name}' of type '{reference.resource_type}' "
            f"not found in environment '{environment.name}' at config path "
            f"'{_render_path(path)}', and {why}"
        )
    else:
        # The only signal a lenient degrade gives, so it distinguishes the two
        # of them: falling back to declared defaults is a config that still
        # builds, while falling back to nothing is a factory about to be
        # called with no arguments at all.
        logger.warning(
            "Resource '%s' of type '%s' not found in environment '%s' at config path '%s'; %s",
            reference.name,
            reference.resource_type,
            environment.name,
            _render_path(path),
            (
                "falling back to its inline defaults"
                if reference.defaults
                else "it declares no inline defaults, so this resolves to an empty config"
            ),
        )

    # Nothing overrides them here, so every default survives and every one is
    # expanded. A degraded config is still config, so it gets the same walk as
    # a found one.
    return _resolve_source(
        reference.defaults,
        environment,
        substitute=substitute,
        strict_resources=strict_resources,
        path=path,
        active=active,
        survey=survey,
    )


def _splice_found_resource(
    reference: _ParsedReference,
    environment: EnvironmentConfig,
    *,
    substitute: bool,
    strict_resources: bool | None,
    path: str,
    active: list[tuple[str, str]],
    survey: list[UnresolvedResourceRef] | None,
) -> Any:
    """Merge the environment's resource with the defaults that survive it.

    The two sources are walked separately and merged after, because they do
    not share a provenance. An environment loaded with substitution arrives
    expanded; inline defaults always arrive raw. Walking the merged result
    under one flag would either re-expand the environment's values or leave
    the defaults' nested refs raw.
    """
    marker = (reference.resource_type, reference.name)
    if marker in active:
        chain = " -> ".join(f"{kind}/{name}" for kind, name in [*active, marker])
        raise ConfigError(
            f"Resource reference cycle at config path '{_render_path(path)}': "
            f"{chain}. A resource that reaches itself has no resolved form, so "
            f"this is reported rather than followed round -- unresolved, it "
            f"exhausts the stack instead."
        )

    # Held only for the resource's own expansion. A reference's inline
    # defaults are spliced after this is popped because they belong to the
    # call site, not to the resource: a default naming the same resource is an
    # ordinary second reference to it, and reporting that as a cycle would
    # reject a config that resolves perfectly well.
    active.append(marker)
    try:
        env_needs_pass = substitute and not environment.substituted
        resolved = _resolve_source(
            environment.get_resource(reference.resource_type, reference.name),
            environment,
            substitute=env_needs_pass,
            strict_resources=strict_resources,
            path=path,
            active=active,
            survey=survey,
        )
    finally:
        active.pop()

    # Inline defaults fill gaps *after* the source above, and each is expanded
    # only once it is known to survive -- so no value is handed to a second
    # expansion, and none is expanded that the environment overrode. The
    # survey follows the same rule: descending into a default the environment
    # supplies would report a reference the build never looks at.
    if isinstance(resolved, dict):
        for key, value in reference.defaults.items():
            if key not in resolved:
                resolved[key] = _resolve_source(
                    value,
                    environment,
                    substitute=substitute,
                    strict_resources=strict_resources,
                    path=_child_path(path, key),
                    active=active,
                    survey=survey,
                )

    return resolved


def _resolve_source(
    value: Any,
    environment: EnvironmentConfig,
    *,
    substitute: bool,
    strict_resources: bool | None,
    path: str,
    active: list[tuple[str, str]],
    survey: list[UnresolvedResourceRef] | None,
) -> Any:
    """Expand one source's ``${VAR}`` refs, then resolve its references.

    A splice merges two sources with different provenances, and the single
    ``substitute`` flag can only be true of one of them. So each is finished
    here, on its own terms, before the merge — rather than merged first and
    walked once under a flag that is wrong for half of the result.

    ``substitute`` says whether *this* source still needs expanding: an
    environment loaded with substitution does not, an inline default always
    does. The pass defers nested ``$resource`` defaults so that the walk below
    expands them at their own splice, exactly once.
    """
    if substitute:
        value = _substitute_deferring_defaults(value)
    return _walk(
        value,
        environment,
        substitute=substitute,
        strict_resources=strict_resources,
        path=path,
        active=active,
        survey=survey,
    )


def resolve_resource_references(
    config: Any,
    environment: EnvironmentConfig,
    *,
    substitute: bool = False,
    strict_resources: bool | None = None,
) -> Any:
    """Resolve every ``$resource`` reference in a config tree.

    The shared primitive behind :meth:`EnvironmentAwareConfig.resolve_for_build`
    and :meth:`ConfigBindingResolver.resolve`, exported so that a consumer with
    a config tree and an environment does not write a third reader of the
    format. Reading it independently is what produced the divergences this
    module now exists to prevent: markers unvalidated, inline defaults
    dropped, ``$required`` ignored, and a fallback branch for a missing
    resource that could not be reached.

    Args:
        config: Config tree, walked recursively. Not mutated.
        environment: Environment resolved against
        substitute: Whether ``config`` still needs ``${VAR}`` expansion.
            ``False`` (default) for a tree already expanded by its loader.
            When True the expansion happens here, holding inline defaults back
            for their own splices -- one pass per source, in the one place
            that knows which sources there are. Nested resources decide for
            themselves from :attr:`EnvironmentConfig.substituted`.
        strict_resources: Whether a reference naming a resource this
            environment does not define raises rather than degrading to its
            inline defaults. ``None`` defers to the environment's
            ``strict_resources`` setting, then to ``False``. A reference's own
            ``$required`` overrides either.

    Returns:
        The tree with every reference replaced by its resolved config

    Raises:
        ResourceNotFoundError: If a reference names a resource this
            environment does not define and the effective policy is strict
        ConfigError: If a reference is malformed, or a resource does not
            declare a capability its reference ``$requires``, or a resource
            reaches itself
    """
    return _resolve_source(
        config,
        environment,
        substitute=substitute,
        strict_resources=strict_resources,
        path="",
        active=[],
        survey=None,
    )


class EnvironmentAwareConfigError(Exception):
    """Error related to environment-aware configuration."""

    pass


class EnvironmentAwareConfig:
    """Configuration with environment-aware resource resolution.

    Manages application configuration with support for:
    - Logical resource references that resolve per-environment
    - Late-binding of environment variables
    - Portable config storage (unresolved)

    Attributes:
        environment: The EnvironmentConfig for resource resolution
        app_name: Name of the loaded application (if any)
    """

    def __init__(
        self,
        config: dict[str, Any],
        environment: EnvironmentConfig | None = None,
        app_name: str | None = None,
        *,
        strict_resources: bool | None = None,
    ):
        """Initialize environment-aware configuration.

        Args:
            config: Application configuration dictionary
            environment: Environment configuration for resource resolution.
                        If None, auto-detects and loads environment.
            app_name: Optional name for this application config
            strict_resources: Whether a ``$resource`` reference naming a
                resource this environment does not define should raise instead
                of degrading to the reference's inline defaults. ``None``
                (default) defers to the environment's ``strict_resources``
                setting, then to ``False`` -- so the default behaviour is
                unchanged. This is the level for an application that hands its
                config to a library which calls
                :meth:`resolve_for_build` itself; a reference's own
                ``$required`` still overrides it.
        """
        self._config = config
        self._environment = environment or EnvironmentConfig.load()
        self._app_name = app_name or config.get("name")
        self._strict_resources = strict_resources

    @property
    def environment(self) -> EnvironmentConfig:
        """Get the current environment configuration."""
        return self._environment

    @property
    def strict_resources(self) -> bool | None:
        """The instance-level missing-resource policy, or None to defer."""
        return self._strict_resources

    @property
    def environment_name(self) -> str:
        """Get the current environment name."""
        return self._environment.name

    @property
    def app_name(self) -> str | None:
        """Get the application name."""
        return self._app_name

    @classmethod
    def load_app(
        cls,
        app_name: str,
        app_dir: str | Path = "config/apps",
        env_dir: str | Path = "config/environments",
        environment: str | None = None,
        *,
        allow_outside: bool = False,
        strict_resources: bool | None = None,
    ) -> EnvironmentAwareConfig:
        """Load an application configuration with environment bindings.

        This is the primary entry point for loading configs in an
        environment-aware manner. Config files are loaded WITHOUT
        environment variable substitution (late binding).

        Args:
            app_name: Application/bot name (without .yaml extension)
            app_dir: Directory containing app configs
            env_dir: Directory containing environment configs
            environment: Environment name, or None to auto-detect
            allow_outside: Opt out of the containment bound, for a deployment
                whose layout genuinely spans sibling trees. Off by default.
                It applies to **both** lookups this method performs -- the
                app name against ``app_dir`` and the environment name against
                ``env_dir`` -- because a layout that spans trees on one side
                generally does on the other, and a flag that silently covered
                only one would be worse than none. Note what that widens: with
                ``environment=None`` the environment name comes from
                ``DATAKNOBS_ENVIRONMENT`` or ``ENVIRONMENT``. An escaping name
                is logged at WARNING when it escapes.
            strict_resources: Missing-resource policy for this config, per
                :meth:`__init__`. ``None`` leaves today's behaviour unchanged.

        Returns:
            EnvironmentAwareConfig with both app and environment loaded

        Raises:
            EnvironmentAwareConfigError: If app config not found or invalid
        """
        app_dir = Path(app_dir)
        env_config = EnvironmentConfig.load(environment, env_dir, allow_outside=allow_outside)

        # Find and load app config file
        config_path = cls._find_config_file(app_dir, app_name, allow_outside=allow_outside)
        if config_path is None:
            raise EnvironmentAwareConfigError(
                f"Application config not found: {app_name}.yaml in {app_dir}"
            )

        config = cls._load_file(config_path)

        logger.info(f"Loaded app config '{app_name}' for environment '{env_config.name}'")

        return cls(
            config=config,
            environment=env_config,
            app_name=app_name,
            strict_resources=strict_resources,
        )

    @classmethod
    def from_dict(
        cls,
        config: dict[str, Any],
        environment: str | None = None,
        env_dir: str | Path = "config/environments",
        *,
        strict_resources: bool | None = None,
    ) -> EnvironmentAwareConfig:
        """Create from a configuration dictionary.

        Args:
            config: Application configuration dictionary
            environment: Environment name, or None to auto-detect
            env_dir: Directory containing environment configs
            strict_resources: Missing-resource policy, per :meth:`__init__`

        Returns:
            EnvironmentAwareConfig instance
        """
        env_config = EnvironmentConfig.load(environment, env_dir)
        return cls(config=config, environment=env_config, strict_resources=strict_resources)

    @classmethod
    def _find_config_file(
        cls, config_dir: Path, name: str, *, allow_outside: bool = False
    ) -> Path | None:
        """Find a config file by name.

        The name is bounded by ``config_dir``: it may address a subdirectory,
        but one that *lands* outside -- whether spelled with ``..`` or as an
        absolute path -- raises rather than reading the file it points at.

        Args:
            config_dir: Directory to search
            name: Config name (without extension)

        Returns:
            Path to config file, or None if not found

        Raises:
            EnvironmentAwareConfigError: If ``name`` addresses a file outside
                ``config_dir`` and ``allow_outside`` is False
        """
        try:
            return find_config_file(config_dir, name, allow_outside=allow_outside)
        except ConfigPathEscapeError as e:
            raise EnvironmentAwareConfigError(str(e)) from e

    @classmethod
    def _load_file(cls, path: Path) -> dict[str, Any]:
        """Load and parse a config file WITHOUT env var substitution.

        Args:
            path: Path to config file

        Returns:
            Parsed configuration dictionary (with env var placeholders intact)
        """
        try:
            return load_yaml_or_json(path)
        except ConfigLoadError as e:
            raise EnvironmentAwareConfigError(f"Failed to load app config {path}: {e}") from e
        except OSError as e:
            raise EnvironmentAwareConfigError(f"Failed to read app config {path}: {e}") from e

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the config.

        Args:
            key: Configuration key (supports dot notation for nested access)
            default: Default value if not found

        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return copy.deepcopy(value)

    def resolve_for_build(
        self,
        config_key: str | None = None,
        resolve_resources: bool = True,
        resolve_env_vars: bool = True,
        *,
        strict_resources: bool | None = None,
    ) -> dict[str, Any]:
        """Resolve configuration for object building.

        This is the late-binding resolution point where:
        1. Logical resource names are resolved to concrete configs
        2. Environment variables are substituted
        3. Final merged configuration is returned

        Call this method immediately before instantiating objects.

        Args:
            config_key: Specific config key to resolve, or None for root
            resolve_resources: Whether to resolve logical resource refs
            resolve_env_vars: Whether to substitute environment variables
            strict_resources: Whether a reference naming a resource this
                environment does not define raises rather than degrading to
                the reference's inline defaults. ``None`` (default) defers to
                the instance default, then to the environment's
                ``strict_resources`` setting, then to ``False``. A reference's
                own ``$required`` overrides every level.

                ``resolve_for_build(strict_resources=True)`` **is** the
                startup preflight: it resolves without constructing anything,
                so it is safe to run at boot purely to prove every binding
                this config names exists in this environment. Use
                :meth:`find_unresolved_resources` instead to get every failure
                in one pass rather than the first one.

        Returns:
            Fully resolved configuration dictionary

        Raises:
            ValueError: If ``strict_resources`` is given explicitly while
                ``resolve_resources`` is False. The flag is only read where
                references are resolved, so the pair silently checks nothing --
                and this method documents itself as *the* startup preflight,
                which is exactly the caller that must not get a green result
                from a run that validated nothing. The *instance* policy is
                not refused here: it is a standing default rather than an
                assertion about this call.
            ResourceNotFoundError: If a reference names a resource this
                environment does not define and the effective policy is strict
            ConfigError: If a reference is malformed -- an unknown
                ``$``-prefixed marker key, an unparseable ``$required``, a
                ``$requires`` that is not a list of names -- or if a resource
                does not declare a capability its reference ``$requires``, or
                if a resource reaches itself
        """
        if strict_resources is not None and not resolve_resources:
            raise ValueError(
                "resolve_for_build(strict_resources=...) requires "
                "resolve_resources=True. The policy is read where references "
                "are resolved, so the two together would validate nothing and "
                "still return."
            )

        # Get the base configuration. Annotated because every step below —
        # ``get``, the substitution pass, the resource splice — walks an
        # arbitrary config tree and so is declared ``Any``; this method is
        # where the shape is actually promised.
        config: dict[str, Any] = self._config_subtree(config_key)

        # Late-bind app-authored ${VAR} refs BEFORE splicing in environment
        # values. Environment values were already substituted when the
        # environment was loaded (or are substituted exactly once, below,
        # when it was not) -- substituting after the splice would expand
        # them a second time, reinterpreting the *content* of a value as a
        # template. See the module docstring.
        #
        # Inline defaults are held back: the splice discards every one the
        # environment supplies, so expanding them here would read values the
        # build is about to throw away -- the same argument that keeps this
        # from expanding the whole environment, applied to the other source.
        #
        # Only when there is a splice to hold them for, though. Without one
        # nothing discards a default and nothing else expands it, so
        # deferring would hand a caller literal ${VAR} text -- which is the
        # branch below, and the only case this method expands anything itself.
        #
        # With a splice, the expansion belongs to the resolution: each source
        # is expanded once, at the point it is spliced, and the resolver is
        # what knows where those points are. Each resource is substituted as
        # it is spliced rather than the environment as a whole -- a resource
        # is still separable there, and expanding the whole environment would
        # read values no reference names, so an unset required ${VAR} in an
        # unrelated resource would abort a build that never looked at it.
        if resolve_resources:
            config = resolve_resource_references(
                config,
                self._environment,
                substitute=resolve_env_vars,
                strict_resources=self._effective_strict(strict_resources),
            )
        elif resolve_env_vars:
            config = _substitute_deferring_defaults(config, defer_defaults=False)

        return config

    def _config_subtree(self, config_key: str | None) -> Any:
        """The whole config, or the subtree one key names.

        Shared by the build and the survey so the two cannot disagree about
        what ``config_key`` addresses -- the smallest of the duplications that
        let them drift, and the one most likely to be copied again.
        """
        if not config_key:
            return copy.deepcopy(self._config)
        subtree = self.get(config_key)
        if subtree is None:
            raise EnvironmentAwareConfigError(f"Config key not found: {config_key}")
        return subtree

    def _effective_strict(self, strict_resources: bool | None) -> bool | None:
        """Collapse the two *code* levels of the precedence chain into one.

        Both are code -- one owned by the caller of a single resolution, the
        other by the application embedding this config -- so a failure message
        that distinguished them would name a distinction its reader cannot
        act on differently. They collapse here rather than deeper so the
        recursion threads a single value; the levels that remain distinguishable
        (the reference's own marker, and the operator's environment setting)
        are read where they live.
        """
        return strict_resources if strict_resources is not None else self._strict_resources

    def get_portable_config(self) -> dict[str, Any]:
        """Get the portable (unresolved) configuration.

        Returns the configuration with:
        - Logical resource references intact
        - Environment variables as placeholders

        This is the config that should be stored in databases
        for cross-environment portability.

        Returns:
            Unresolved configuration dictionary
        """
        return copy.deepcopy(self._config)

    def to_dict(self) -> dict[str, Any]:
        """Get the raw configuration dictionary.

        Alias for get_portable_config().

        Returns:
            Configuration dictionary
        """
        return self.get_portable_config()

    def with_environment(
        self,
        environment: str | EnvironmentConfig,
        env_dir: str | Path = "config/environments",
    ) -> EnvironmentAwareConfig:
        """Create a new instance with a different environment.

        Useful for testing or multi-environment scenarios.

        Args:
            environment: Environment name or EnvironmentConfig instance
            env_dir: Directory containing environment configs (if name provided)

        Returns:
            New EnvironmentAwareConfig with the specified environment
        """
        if isinstance(environment, str):
            env_config = EnvironmentConfig.load(environment, env_dir)
        else:
            env_config = environment

        return EnvironmentAwareConfig(
            config=copy.deepcopy(self._config),
            environment=env_config,
            app_name=self._app_name,
            # Carried, not re-defaulted. This method is on the common path for
            # a caller that supplies a config and an environment separately,
            # so dropping the policy here would silently revert strict mode to
            # lenient at precisely the point a second environment enters --
            # which is the point the policy is most likely to matter.
            strict_resources=self._strict_resources,
        )

    def find_unresolved_resources(
        self,
        config_key: str | None = None,
        *,
        strict_resources: bool | None = None,
    ) -> list[UnresolvedResourceRef]:
        """Every ``$resource`` reference whose resource this environment lacks.

        Raise-on-first is right for a build and wrong for a preflight: an
        operator auditing a config tree wants every unresolvable reference in
        one pass, not one per run. This constructs nothing -- resolution is
        dict manipulation, and no factory is reached -- and raises nothing for
        a missing resource.

        It runs the **same walk** as :meth:`resolve_for_build`, differing only
        in what it does when a resource is absent: record it and carry on down
        the lenient path, rather than raise or warn. That is what makes it a
        prediction of the build rather than a second opinion about it. A
        reference nested inside a resolved resource is surveyed, because a
        build reaches it; a reference nested inside an inline default the
        environment overrides is not, because a build discards it.

        **An empty list means a build reaches no unresolvable reference.**
        Every way a reference can fail *other* than by naming an absent
        resource raises here instead of being listed -- a malformed reference,
        a resource that reaches itself, or a present resource that does not
        declare a capability its reference ``$requires``. Listing is for the
        failure an operator fixes by adding bindings, and there is no useful
        sense in which the survey could report those others and still be a
        survey: a config that a build cannot walk has no complete list of
        unresolvable references to give.

        Args:
            config_key: Specific config key to survey, or None for the whole
                config
            strict_resources: Policy used to populate
                :attr:`UnresolvedResourceRef.required`, per
                :meth:`resolve_for_build`. It does not affect *which*
                references are reported -- every unresolvable one is, whatever
                the policy -- only whether each is reported as fatal.

        Returns:
            One entry per unresolvable reference, depth-first in config order

        Raises:
            ConfigError: If a reference is malformed, a resource reaches
                itself, or a present resource does not declare a capability
                its reference ``$requires``. A survey that reported a tree
                sound while the build raises on it would be worse than no
                survey.
        """
        # ``substitute=True`` expands as the build does, so a reference
        # selecting its resource by variable is reported under the name it
        # would actually look up -- the raw `${LLM_BINDING}` text would be a
        # finding nobody can act on -- and a reference nested in an inline
        # default is expanded at its own splice rather than early or not at
        # all.
        found: list[UnresolvedResourceRef] = []
        _resolve_source(
            self._config_subtree(config_key),
            self._environment,
            substitute=True,
            strict_resources=self._effective_strict(strict_resources),
            path="",
            active=[],
            survey=found,
        )
        return found

    def get_resource(
        self,
        resource_type: str,
        logical_name: str,
        defaults: dict[str, Any] | None = None,
        *,
        required: bool | None = None,
    ) -> dict[str, Any]:
        """Get a resolved resource configuration.

        Convenience method to directly access environment resources.

        Args:
            resource_type: Type of resource
            logical_name: Logical name of resource
            defaults: Default values if resource not found
            required: Whether an absent resource raises, independently of
                whether ``defaults`` were supplied. See
                :meth:`EnvironmentConfig.get_resource`. Note that this is the
                *direct-access* policy: a ``$resource`` reference resolved
                through :meth:`resolve_for_build` decides its own, and is not
                affected by this parameter.

        Returns:
            Resolved resource configuration
        """
        return self._environment.get_resource(
            resource_type, logical_name, defaults, required=required
        )

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get an environment setting.

        Args:
            key: Setting key
            default: Default value if not found

        Returns:
            Setting value
        """
        return self._environment.get_setting(key, default)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"EnvironmentAwareConfig(app={self._app_name!r}, "
            f"environment={self._environment.name!r})"
        )
