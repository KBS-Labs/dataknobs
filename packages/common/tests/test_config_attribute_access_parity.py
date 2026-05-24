"""Tests for ``assert_config_attribute_access_matches_dataclass``.

The helper AST-walks a consumer's MRO for ``self.<config_attr>.<name>``
reads and asserts each ``<name>`` is a field or attribute of the typed
config dataclass. It is the body-access counterpart to
``assert_dataclass_config_matches_ctor`` (the ctor-surface guard).

These tests exercise each behaviour against synthetic consumer/config
pairs (real classes, not mocks) so the diagnostic output stays
deterministic:

* a consumer reading only fields (and config *methods*) passes;
* a config method read does not false-positive (the valid surface is
  the union of fields and dataclass attributes/methods);
* a read of a non-existent attribute fires;
* inherited base-class reads are covered (MRO walk), in both the
  clean and the drifted direction;
* ``ignore_attrs`` whitelists an intentional off-config read;
* ``config_attr`` retargets the audited attribute name;
* a non-dataclass config type is rejected with a clear error.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from dataknobs_common.testing import (
    assert_config_attribute_access_matches_dataclass,
)


@dataclass
class _Cfg:
    alpha: int = 0
    beta: str = "b"

    def derived(self) -> str:
        """A config helper method — a valid (non-field) read target."""
        return f"{self.alpha}-{self.beta}"


class _GoodConsumer:
    def __init__(self, config: _Cfg) -> None:
        self.config = config

    def use(self) -> str:
        # Reads two fields and a method — all on _Cfg.
        return f"{self.config.alpha}{self.config.beta}{self.config.derived()}"


class _BadConsumer:
    def __init__(self, config: _Cfg) -> None:
        self.config = config

    def use(self) -> object:
        return self.config.does_not_exist  # not a field or attribute of _Cfg


class _MethodOnlyConsumer:
    def __init__(self, config: _Cfg) -> None:
        self.config = config

    def use(self) -> str:
        # Only a method read — must not be flagged just because it isn't
        # a dataclass *field*.
        return self.config.derived()


def test_good_consumer_passes() -> None:
    """A consumer reading only fields and config methods passes."""
    assert_config_attribute_access_matches_dataclass(_GoodConsumer, _Cfg)


def test_method_read_is_not_a_gap() -> None:
    """Reading a config *method* is valid — fields and attrs both count."""
    assert_config_attribute_access_matches_dataclass(_MethodOnlyConsumer, _Cfg)


def test_unknown_attribute_fires() -> None:
    """Reading an attribute that isn't on the config type raises."""
    with pytest.raises(AssertionError, match="does_not_exist"):
        assert_config_attribute_access_matches_dataclass(_BadConsumer, _Cfg)


# --- MRO coverage -----------------------------------------------------------


class _CleanBase:
    def __init__(self, config: _Cfg) -> None:
        self.config = config

    def base_use(self) -> int:
        return self.config.alpha  # inherited field read


class _CleanDerived(_CleanBase):
    """Defines no reads of its own; the field read lives on the base."""


class _DriftedBase:
    def __init__(self, config: _Cfg) -> None:
        self.config = config

    def base_use(self) -> object:
        return self.config.ghost  # inherited non-field read


class _DerivedOverClean(_DriftedBase):
    """Clean leaf, but inherits a drifted base read."""


def test_inherited_clean_read_passes() -> None:
    """A field read inherited from a base class is found and passes."""
    assert_config_attribute_access_matches_dataclass(_CleanDerived, _Cfg)


def test_inherited_drifted_read_fires() -> None:
    """A non-field read inherited from a base class is caught (MRO walk)."""
    with pytest.raises(AssertionError, match="ghost"):
        assert_config_attribute_access_matches_dataclass(_DerivedOverClean, _Cfg)


# --- ignore_attrs / config_attr / non-dataclass -----------------------------


def test_ignore_attrs_whitelists_read() -> None:
    """An off-config read named in ``ignore_attrs`` passes."""
    assert_config_attribute_access_matches_dataclass(
        _BadConsumer, _Cfg, ignore_attrs=frozenset({"does_not_exist"})
    )


class _CustomAttrConsumer:
    def __init__(self, config: _Cfg) -> None:
        self.cfg = config

    def use(self) -> int:
        return self.cfg.alpha


class _CustomAttrBad:
    def __init__(self, config: _Cfg) -> None:
        self.cfg = config

    def use(self) -> object:
        return self.cfg.nope


def test_config_attr_override_passes() -> None:
    """``config_attr`` retargets which instance attribute is audited."""
    assert_config_attribute_access_matches_dataclass(
        _CustomAttrConsumer, _Cfg, config_attr="cfg"
    )


def test_config_attr_override_fires() -> None:
    """With a retargeted attr, a drifted read off it is still caught."""
    with pytest.raises(AssertionError, match="nope"):
        assert_config_attribute_access_matches_dataclass(
            _CustomAttrBad, _Cfg, config_attr="cfg"
        )


def test_default_config_attr_ignores_other_attrs() -> None:
    """Reads off a differently-named attr are out of scope by default.

    ``_CustomAttrBad`` reads ``self.cfg.nope`` — with the default
    ``config_attr="config"`` that read is not audited, so the helper
    passes. This pins the D4 scoping: only ``self.config.<attr>`` is
    walked, so reads off other objects (params, a dataknobs ``Config``,
    a dict) are never false-flagged.
    """
    assert_config_attribute_access_matches_dataclass(_CustomAttrBad, _Cfg)


class _NotADataclass:
    alpha = 0


def test_non_dataclass_config_rejected() -> None:
    """Passing a non-dataclass config type raises a clear error."""
    with pytest.raises(AssertionError, match="not a dataclass"):
        assert_config_attribute_access_matches_dataclass(
            _GoodConsumer, _NotADataclass
        )
