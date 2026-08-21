"""The ``_UNKNOWN_KEYS`` policy on :class:`StructuredConfig`.

``from_dict`` projects a dict onto the declared fields and drops the rest.
That is right for a config travelling with the routing key that selected it,
and wrong wherever every field has a working default: there, a misspelled key
does not fail, it succeeds against the wrong thing. The policy lets a class
say which of the two it is.

The default stays ``"ignore"`` deliberately. Flipping it for every
``StructuredConfig`` subclass at once is a separate decision from giving one
family the ability to opt in, and these tests pin the default so that
decision has to be taken rather than arrived at.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

import pytest

from dataknobs_common.structured_config import StructuredConfig


@dataclass(frozen=True)
class LenientConfig(StructuredConfig):
    """Default policy: an unmatched key is discarded."""

    host: str = "localhost"


@dataclass(frozen=True)
class StrictConfig(StructuredConfig):
    """Opted in: an unmatched key is a ``ValueError``."""

    host: str = "localhost"
    connection_string: str = ""

    _UNKNOWN_KEYS: ClassVar[Literal["ignore", "raise"]] = "raise"


@dataclass(frozen=True)
class StrictWithAlias(StrictConfig):
    """Declares an input spelling its normalizer consumes."""

    port: int = 5432

    _INPUT_KEYS: ClassVar[frozenset[str]] = frozenset({"hostname"})

    @classmethod
    def _normalize_dict(cls, raw: dict[str, str]) -> dict[str, str]:
        if "hostname" in raw:
            raw.setdefault("host", raw.pop("hostname"))
        return raw


@dataclass(frozen=True)
class StrictWithMoreAliases(StrictWithAlias):
    """Adds one alias; must keep its base's."""

    _INPUT_KEYS: ClassVar[frozenset[str]] = frozenset({"portnum"})

    @classmethod
    def _normalize_dict(cls, raw: dict[str, str]) -> dict[str, str]:
        if "portnum" in raw:
            raw.setdefault("port", raw.pop("portnum"))
        return super()._normalize_dict(raw)


class TestTheDefaultIsUnchanged:
    def test_an_unmatched_key_is_still_discarded(self) -> None:
        cfg = LenientConfig.from_dict({"host": "h", "hosst": "typo"})
        assert cfg.host == "h"

    def test_the_class_reports_the_default_policy(self) -> None:
        assert LenientConfig._UNKNOWN_KEYS == "ignore"
        assert StructuredConfig._UNKNOWN_KEYS == "ignore"


class TestOptingIn:
    def test_an_unmatched_key_raises(self) -> None:
        with pytest.raises(ValueError, match="hosst"):
            StrictConfig.from_dict({"hosst": "typo"})

    def test_the_error_suggests_the_near_miss(self) -> None:
        with pytest.raises(ValueError, match="did you mean 'host'"):
            StrictConfig.from_dict({"hosst": "typo"})

    def test_the_error_suggests_a_longer_key_the_input_prefixes(self) -> None:
        """``connection`` scores too low for difflib but prefixes the real key."""
        with pytest.raises(ValueError, match="did you mean 'connection_string'"):
            StrictConfig.from_dict({"connection": "postgres://..."})

    def test_the_error_lists_the_accepted_keys(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            StrictConfig.from_dict({"wildly_unrelated": 1})
        message = str(excinfo.value)
        assert "Accepted keys: connection_string, host." in message
        assert "did you mean" not in message, "no near miss should be invented"

    def test_every_unmatched_key_is_named_at_once(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            StrictConfig.from_dict({"aaa": 1, "zzz": 2})
        message = str(excinfo.value)
        assert "'aaa'" in message and "'zzz'" in message

    def test_a_matching_config_still_builds(self) -> None:
        assert StrictConfig.from_dict({"host": "h"}).host == "h"


class TestRoutingKeys:
    """The object-graph layer's vocabulary is tolerated, not advertised."""

    @pytest.mark.parametrize("key", ["backend", "factory", "name", "type"])
    def test_a_routing_key_passes_through(self, key: str) -> None:
        cfg = StrictConfig.from_dict({key: "whatever", "host": "h"})
        assert cfg.host == "h"

    def test_a_routing_key_is_not_offered_as_an_answer(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            StrictConfig.from_dict({"wildly_unrelated": 1})
        assert "backend" not in str(excinfo.value)


class TestInputKeys:
    def test_a_declared_alias_is_accepted(self) -> None:
        assert StrictWithAlias.from_dict({"hostname": "h"}).host == "h"

    def test_an_undeclared_neighbour_of_it_is_not(self) -> None:
        with pytest.raises(ValueError, match="hostnam"):
            StrictWithAlias.from_dict({"hostnam": "h"})

    def test_aliases_union_across_the_mro(self) -> None:
        """A subclass declares only what it adds."""
        cfg = StrictWithMoreAliases.from_dict({"hostname": "h", "portnum": 6000})
        assert (cfg.host, cfg.port) == ("h", 6000)

    def test_the_error_lists_inherited_aliases_too(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            StrictWithMoreAliases.from_dict({"wildly_unrelated": 1})
        message = str(excinfo.value)
        assert "hostname" in message and "portnum" in message


class TestAccepts:
    def test_it_reports_fields(self) -> None:
        assert StrictConfig.accepts("host") is True

    def test_it_reports_declared_aliases(self) -> None:
        assert StrictWithAlias.accepts("hostname") is True

    def test_it_reports_inherited_aliases(self) -> None:
        assert StrictWithMoreAliases.accepts("hostname") is True

    def test_it_rejects_an_unknown_key(self) -> None:
        assert StrictConfig.accepts("hosst") is False

    def test_it_excludes_routing_keys(self) -> None:
        """Tolerated by ``from_dict``, but not part of this config's surface."""
        assert StrictConfig.accepts("backend") is False

    def test_it_answers_for_a_lenient_class_too(self) -> None:
        """The question is about the surface, not about the policy."""
        assert LenientConfig.accepts("host") is True
        assert LenientConfig.accepts("hosst") is False
