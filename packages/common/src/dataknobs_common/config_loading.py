"""Shared YAML/JSON configuration loading helpers.

Consolidates the file→dict and bytes→dict parse-and-validate chain
that was previously duplicated across ``dataknobs_config``,
``dataknobs_xization.ingestion``, ``dataknobs_fsm.config``, and
``dataknobs_bots.knowledge``. Each consumer keeps its own error class
by catching the helper's :class:`ConfigLoadError` hierarchy and
re-raising as its own type.

The helpers do **not** apply environment-variable substitution —
different consumers need different substitutors with different
options. Pair with :func:`dataknobs_config.substitute_env_vars` at the
consumer level when needed.

PyYAML is lazy-imported inside helper bodies so ``dataknobs-common``
keeps zero hard dependencies. Consumers that pass ``.yaml`` / ``.yml``
files must declare ``pyyaml`` in their own dependencies (matches the
existing pattern in ``dataknobs_xization.ingestion``). JSON support is
built-in (stdlib).

Example:
    >>> from pathlib import Path
    >>> from dataknobs_common.config_loading import (
    ...     find_config_file, load_yaml_or_json,
    ... )
    >>> path = find_config_file(Path("configs/environments"), "production")
    >>> data = load_yaml_or_json(path) if path else {}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal, overload

from dataknobs_common.paths import safe_join

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_EXTENSIONS: tuple[str, ...] = (".yaml", ".yml", ".json")


class ConfigLoadError(Exception):
    """Base exception for config-loading helper failures.

    Subclasses cover specific failure modes; consumers that need a
    custom error class should catch this base type and re-raise.
    """


class ConfigParseError(ConfigLoadError):
    """Raised when YAML or JSON parsing fails."""


class ConfigShapeError(ConfigLoadError):
    """Raised when ``require_dict=True`` and the parsed root is not a dict."""


class ConfigUnsupportedFormatError(ConfigLoadError):
    """Raised when the file extension or format hint is not recognized."""


class ConfigYAMLNotInstalledError(ConfigLoadError):
    """Raised when a YAML payload is requested but PyYAML is not installed."""


class ConfigPathEscapeError(ConfigLoadError):
    """Raised when a config name addresses a file outside the search directory."""


def find_config_file(
    config_dir: str | Path,
    name: str,
    *,
    extensions: tuple[str, ...] = DEFAULT_CONFIG_EXTENSIONS,
    allow_outside: bool = False,
) -> Path | None:
    """Find a config file by basename, probing each extension in order.

    ``name`` addresses a file **inside** ``config_dir``. It may name a
    subdirectory (``"domains/child"``), which is how a layout convention
    is expressed; it may not walk out of the directory with ``..`` and it
    may not be absolute, because joining an absolute name discards
    ``config_dir`` altogether. Either is rejected before any file is
    probed, so an escaping name fails the same way whether or not the
    file it names exists.

    The bound matters because the name is not always the caller's own
    literal — it can be an ``extends:`` value read out of a config file,
    an environment name read from the process environment, or the output
    of a consumer-supplied name resolver.

    Containment is judged lexically, by
    :func:`dataknobs_common.paths.safe_join`; a symlink inside
    ``config_dir`` is followed as the deployment intended it. The path
    returned is composed exactly as before — the guard reads the name, it
    does not rewrite it.

    Args:
        config_dir: Directory to search.
        name: Basename without extension. May address a subdirectory of
            ``config_dir``.
        extensions: Extensions to try, in priority order. A leading
            dot is added if missing (``"yaml"`` → ``".yaml"``); the
            comparison itself is filesystem-cased (the extension
            string is used verbatim after normalization).
        allow_outside: Opt out of the containment check for a caller that
            deliberately addresses a tree outside ``config_dir``. An
            escaping name is then logged at WARNING and probed anyway, so
            the bypass is visible in a deployment's logs rather than
            silent. No in-tree caller passes it.

    Returns:
        Path to the first existing match, or ``None`` if no candidate
        exists.

    Raises:
        ConfigPathEscapeError: ``name`` addresses a file outside
            ``config_dir`` and ``allow_outside`` is False.
    """
    directory = Path(config_dir)
    if safe_join(directory, name) is None:
        if not allow_outside:
            raise ConfigPathEscapeError(
                f"Configuration name addresses a file outside the configuration directory: {name!r}"
            )
        logger.warning(
            "Configuration name %r addresses a file outside %s; allowed by allow_outside=True",
            name,
            directory,
        )
    normalized = tuple(ext if ext.startswith(".") else f".{ext}" for ext in extensions)
    for ext in normalized:
        candidate = directory / f"{name}{ext}"
        if candidate.exists():
            return candidate
    return None


#: ``require_dict`` decides the return type, so it is spelled as overloads
#: rather than left to the docstring. Every caller that passes ``True`` (or
#: takes the default) declares a ``dict[str, Any]`` return of its own, and a
#: helper returning bare ``Any`` made each of those a *Returning Any from
#: function declared to return* finding — a whole class of them, one per
#: caller, all traceable to one signature.
@overload
def load_yaml_or_json(
    path: str | Path,
    *,
    require_dict: Literal[True] = ...,
    encoding: str = ...,
) -> dict[str, Any]: ...


@overload
def load_yaml_or_json(
    path: str | Path,
    *,
    require_dict: bool,
    encoding: str = ...,
) -> Any: ...


def load_yaml_or_json(
    path: str | Path,
    *,
    require_dict: bool = True,
    encoding: str = "utf-8",
) -> Any:
    """Load and parse a YAML or JSON file based on its extension.

    Args:
        path: File path. Extension determines parser
            (``.yaml`` / ``.yml`` → YAML, ``.json`` → JSON;
            comparison is case-insensitive).
        require_dict: When ``True`` (default), raise
            :class:`ConfigShapeError` if the parsed root is not a dict.
            When ``False``, any parsed value (list, scalar, ``None``)
            is returned as-is.
        encoding: File encoding (default ``utf-8``).

    Returns:
        Parsed data. Type is ``dict[str, Any]`` when
        ``require_dict=True``, otherwise ``Any``.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ConfigUnsupportedFormatError: Extension is not recognized.
        ConfigYAMLNotInstalledError: ``.yaml`` / ``.yml`` requested
            but PyYAML is not installed.
        ConfigParseError: YAML or JSON parsing failed.
        ConfigShapeError: ``require_dict=True`` and the root is not a
            dict.
        OSError: I/O failure reading the file (caller may wrap).
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in (".yaml", ".yml"):
        format_: Literal["yaml", "json"] = "yaml"
    elif suffix == ".json":
        format_ = "json"
    else:
        raise ConfigUnsupportedFormatError(
            f"Unsupported config file extension: {suffix!r} (path: {p})"
        )

    with open(p, encoding=encoding) as f:
        text = f.read()

    return parse_yaml_or_json(
        text,
        format=format_,
        source_name=str(p),
        require_dict=require_dict,
    )


@overload
def parse_yaml_or_json(
    data: bytes | str,
    *,
    format: Literal["yaml", "json"],
    source_name: str | None = ...,
    require_dict: Literal[True] = ...,
) -> dict[str, Any]: ...


@overload
def parse_yaml_or_json(
    data: bytes | str,
    *,
    format: Literal["yaml", "json"],
    source_name: str | None = ...,
    require_dict: bool,
) -> Any: ...


def parse_yaml_or_json(
    data: bytes | str,
    *,
    format: Literal["yaml", "json"],
    source_name: str | None = None,
    require_dict: bool = True,
) -> Any:
    """Parse YAML or JSON content already in memory.

    Used by consumers that read content from a non-filesystem source
    (e.g. ``RAGKnowledgeBase`` reading bytes from a
    ``KnowledgeResourceBackend``).

    Args:
        data: Raw bytes or text. Bytes are decoded as UTF-8.
        format: ``"yaml"`` or ``"json"``.
        source_name: Optional name used in error messages (e.g. a URI
            or backend filename). Defaults to ``"<format>"`` when
            unset.
        require_dict: When ``True`` (default), raise
            :class:`ConfigShapeError` if the root is not a dict.

    Returns:
        Parsed data.

    Raises:
        ConfigUnsupportedFormatError: ``format`` is not ``"yaml"`` or
            ``"json"``.
        ConfigYAMLNotInstalledError: ``format="yaml"`` but PyYAML is
            not installed.
        ConfigParseError: Parsing failed, or input bytes were not
            valid UTF-8.
        ConfigShapeError: ``require_dict=True`` and the root is not a
            dict.
    """
    name = source_name or f"<{format}>"
    if isinstance(data, bytes):
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ConfigParseError(f"Failed to decode bytes as UTF-8 ({name}): {e}") from e
    else:
        text = data

    if format == "json":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            raise ConfigParseError(f"Failed to parse JSON ({name}): {e}") from e
    elif format == "yaml":
        try:
            import yaml
        except ImportError as e:
            raise ConfigYAMLNotInstalledError(
                f"PyYAML is required to parse YAML ({name}). Install with: pip install pyyaml"
            ) from e
        try:
            parsed = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise ConfigParseError(f"Failed to parse YAML ({name}): {e}") from e
    else:
        raise ConfigUnsupportedFormatError(
            f"Unsupported format: {format!r} (must be 'yaml' or 'json')"
        )

    if require_dict and not isinstance(parsed, dict):
        raise ConfigShapeError(
            f"Expected a dict at the root of {name}, got {type(parsed).__name__}"
        )

    return parsed
