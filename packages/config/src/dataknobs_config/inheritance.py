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
from pathlib import Path
from typing import Any

from dataknobs_common.config_loading import (
    ConfigLoadError,
    find_config_file,
    load_yaml_or_json,
)

logger = logging.getLogger(__name__)


class InheritanceError(Exception):
    """Error during configuration inheritance resolution."""
    pass


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries.

    Recursively merges override into base, with override values taking precedence.
    Nested dictionaries are merged recursively; all other types are replaced.

    Args:
        base: Base dictionary (values used when not overridden)
        override: Override dictionary (takes precedence)

    Returns:
        New merged dictionary

    Example:
        >>> base = {"a": 1, "nested": {"x": 10, "y": 20}}
        >>> override = {"a": 2, "nested": {"y": 25, "z": 30}}
        >>> deep_merge(base, override)
        {'a': 2, 'nested': {'x': 10, 'y': 25, 'z': 30}}
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
                    _substitute_string(
                        k, type_coerce=False, expand_user_paths=expand_user_paths
                    )
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
        ValueError: If a required variable is unset.
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

    Attributes:
        config_dir: Directory containing configuration files
        cache: Configuration cache for performance
    """

    def __init__(self, config_dir: str | Path | None = None):
        """Initialize configuration loader.

        Args:
            config_dir: Directory containing configuration files.
                       If None, uses ./configs
        """
        self.config_dir = Path(config_dir) if config_dir else Path("./configs")
        self._cache: dict[str, dict[str, Any]] = {}
        self._loading: set[str] = set()  # Track configs being loaded to detect cycles

    def load(
        self,
        name: str,
        use_cache: bool = True,
        substitute_vars: bool = True,
    ) -> dict[str, Any]:
        """Load and resolve configuration with inheritance.

        Args:
            name: Configuration name (without extension)
            use_cache: Whether to use cached configuration if available
            substitute_vars: Whether to substitute environment variables

        Returns:
            Resolved configuration dictionary

        Raises:
            InheritanceError: If config not found, cycle detected, or other error

        Example:
            ```python
            loader = InheritableConfigLoader("./configs")
            config = loader.load("my-domain")
            ```
        """
        # Check cache
        if use_cache and name in self._cache:
            logger.debug("Using cached config: %s", name)
            return self._cache[name]

        # Detect circular inheritance
        if name in self._loading:
            raise InheritanceError(f"Circular inheritance detected: {name}")

        self._loading.add(name)

        try:
            # Load raw configuration
            raw_config = self._load_file(name)

            # Handle inheritance
            if raw_config.get("extends"):
                parent_name = raw_config["extends"]
                logger.debug("Config '%s' extends '%s'", name, parent_name)

                # Load parent configuration (recursively handles inheritance)
                parent_config = self.load(parent_name, use_cache=use_cache, substitute_vars=False)

                # Deep merge: child overrides parent
                raw_config = deep_merge(parent_config, raw_config)

                # Remove extends field from final config
                raw_config.pop("extends", None)

            # Substitute environment variables
            if substitute_vars:
                raw_config = substitute_env_vars(raw_config)

            # Cache the result
            self._cache[name] = raw_config
            logger.info("Loaded configuration: %s", name)

            return raw_config

        finally:
            self._loading.discard(name)

    def load_from_file(
        self,
        filepath: str | Path,
        substitute_vars: bool = True,
    ) -> dict[str, Any]:
        """Load configuration from a specific file path.

        This method bypasses the config_dir and loads directly from the path.
        Inheritance is resolved relative to the file's directory.

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

        # Temporarily change config_dir to file's directory for inheritance
        old_config_dir = self.config_dir
        self.config_dir = filepath.parent

        try:
            return self.load(filepath.stem, use_cache=False, substitute_vars=substitute_vars)
        finally:
            self.config_dir = old_config_dir

    def _load_file(self, name: str) -> dict[str, Any]:
        """Load raw configuration file.

        Args:
            name: Configuration name (without extension)

        Returns:
            Parsed configuration dictionary

        Raises:
            InheritanceError: If file not found or parse error
        """
        filepath = find_config_file(self.config_dir, name)
        if filepath is None:
            raise InheritanceError(
                f"Configuration file not found: {name}.yaml or {name}.json "
                f"in {self.config_dir}"
            )

        try:
            return load_yaml_or_json(filepath)
        except ConfigLoadError as e:
            raise InheritanceError(str(e)) from e
        except OSError as e:
            raise InheritanceError(
                f"Failed to read configuration file {filepath}: {e}"
            ) from e

    def clear_cache(self, name: str | None = None) -> None:
        """Clear configuration cache.

        Args:
            name: Specific config to clear, or None to clear all
        """
        if name:
            self._cache.pop(name, None)
            logger.debug("Cleared cache for: %s", name)
        else:
            self._cache.clear()
            logger.debug("Cleared all cached configurations")

    def list_available(self) -> list[str]:
        """List all available configuration files.

        Returns:
            List of configuration names (without extensions)
        """
        if not self.config_dir.exists():
            return []

        configs = set()
        for pattern in ["*.yaml", "*.yml", "*.json"]:
            for file in self.config_dir.glob(pattern):
                if file.is_file():
                    configs.add(file.stem)

        return sorted(configs)

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
