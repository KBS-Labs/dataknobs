"""``StructuredConfig.fingerprint()``: a stable key for a config that is not hashable.

A config compares field by field and is not hashable, so it cannot key a dict,
a set or an ``lru_cache`` itself. ``fingerprint()`` is what keys one instead: a
digest of the config's class and fields, equal for equal configs, different
for configs that differ, and stable across processes. It is a digest rather
than the fields themselves, so a credential in the config does not appear in
the key, and it refuses a value it cannot identify rather than guess.
"""

from __future__ import annotations

import datetime
import decimal
import enum
import fractions
import pathlib
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

import pytest

from dataknobs_common.retry import RetryConfig
from dataknobs_common.structured_config import StructuredConfig


class Mode(enum.Enum):
    FAST = "fast"
    SLOW = "slow"


@dataclass(frozen=True)
class Inner(StructuredConfig):
    name: str = "inner"


@dataclass
class Plain:
    """A plain dataclass held by a config, as ``DatabaseConfig`` holds a schema."""

    columns: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Service(StructuredConfig):
    host: str = "localhost"
    port: int = 5432
    password: str | None = None
    tags: list[str] = field(default_factory=list)
    labels: Mapping[Any, Any] = field(default_factory=dict)
    roles: frozenset[str] = frozenset()
    mode: Mode = Mode.FAST
    inner: Inner = field(default_factory=Inner)
    plain: Plain = field(default_factory=Plain)
    root: pathlib.Path | None = None
    hook: Any = None


@dataclass(frozen=True)
class Twin(StructuredConfig):
    """The same fields as ``Inner``, under another name."""

    name: str = "inner"


@dataclass(frozen=True)
class Streaming(StructuredConfig):
    model: str = "m"
    on_token: Any = None

    _FINGERPRINT_EXCLUDE: ClassVar[frozenset[str]] = frozenset({"on_token"})


@dataclass(frozen=True)
class Misspelt(StructuredConfig):
    model: str = "m"

    _FINGERPRINT_EXCLUDE: ClassVar[frozenset[str]] = frozenset({"modle"})


class Endpoint:
    """A value type a config holds and knows how to identify."""

    def __init__(self, url: str) -> None:
        self.url = url


@dataclass(frozen=True)
class Client(StructuredConfig):
    endpoint: Any = None

    def _fingerprint_value(self, value: Any, path: str) -> Any:
        if isinstance(value, Endpoint):
            return value.url
        return super()._fingerprint_value(value, path)


@dataclass(frozen=True)
class Echoing(StructuredConfig):
    """Hands back what it was given: the encoding must refuse, not loop."""

    endpoint: Any = None

    def _fingerprint_value(self, value: Any, path: str) -> Any:
        return value


@dataclass(frozen=True)
class Wrapping(StructuredConfig):
    """Wraps what it was given, which re-entering the hook would wrap forever."""

    endpoint: Any = None

    def _fingerprint_value(self, value: Any, path: str) -> Any:
        return [type(value).__name__, value]


@dataclass(frozen=True)
class Timed(StructuredConfig):
    model: str = "m"
    note: str = field(default="", compare=False)


def a_factory_class() -> type[StructuredConfig]:
    """A config class built inside a function: no import reaches it."""

    @dataclass(frozen=True)
    class Made(StructuredConfig):
        n: int = 1

    return Made


def module_level_hook() -> None:
    """A function a config can name, as a callback or a factory."""


class TestEqualConfigsShareAFingerprint:
    def test_a_fingerprint_is_a_hex_digest(self) -> None:
        digest = Service().fingerprint()
        assert len(digest) == 64
        int(digest, 16)

    def test_two_equal_configs_share_one(self) -> None:
        assert Service(tags=["a"]).fingerprint() == Service(tags=["a"]).fingerprint()

    def test_the_roundtrip_keeps_it(self) -> None:
        config = RetryConfig(max_attempts=5, retry_on_exceptions=[ValueError])
        restored = RetryConfig.from_dict(config.to_dict())
        assert restored == config
        assert restored.fingerprint() == config.fingerprint()

    def test_set_order_does_not_change_it(self) -> None:
        a = Service(roles=frozenset(["x", "y", "z"]))
        b = Service(roles=frozenset(["z", "y", "x"]))
        assert a.fingerprint() == b.fingerprint()

    def test_mapping_order_does_not_change_it(self) -> None:
        a = Service(labels={"a": 1, "b": 2})
        b = Service(labels={"b": 2, "a": 1})
        assert a.fingerprint() == b.fingerprint()

    def test_it_is_the_same_in_another_process(self) -> None:
        """Stable across runs, so it can key something that outlives the process."""
        code = (
            "from dataknobs_common.retry import RetryConfig;"
            "print(RetryConfig(max_attempts=4).fingerprint())"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout.strip()
        assert out == RetryConfig(max_attempts=4).fingerprint()


class TestConfigsThatDifferDoNotShareOne:
    def test_a_different_value(self) -> None:
        assert Service(port=1).fingerprint() != Service(port=2).fingerprint()

    def test_a_different_class_with_the_same_fields(self) -> None:
        assert Inner().fingerprint() != Twin().fingerprint()

    def test_a_key_and_its_spelling_as_a_string(self) -> None:
        assert Service(labels={1: "a"}).fingerprint() != Service(labels={"1": "a"}).fingerprint()

    def test_mixed_key_types_are_encoded_rather_than_refused(self) -> None:
        Service(labels={1: "a", "b": 2, None: 3}).fingerprint()

    def test_a_nested_config(self) -> None:
        a = Service(inner=Inner(name="a"))
        b = Service(inner=Inner(name="b"))
        assert a.fingerprint() != b.fingerprint()

    def test_a_plain_dataclass_held_by_the_config(self) -> None:
        a = Service(plain=Plain(columns=["x"]))
        b = Service(plain=Plain(columns=["y"]))
        assert a.fingerprint() != b.fingerprint()

    def test_a_path(self) -> None:
        a = Service(root=pathlib.Path("/a"))
        b = Service(root=pathlib.Path("/b"))
        assert a.fingerprint() != b.fingerprint()

    def test_an_enum_member(self) -> None:
        assert Service(mode=Mode.FAST).fingerprint() != Service(mode=Mode.SLOW).fingerprint()


class TestCredentials:
    def test_a_password_differs_the_fingerprint_without_appearing_in_it(self) -> None:
        """Two configs differing only by credential must not share a cached client."""
        a = Service(password="correct-horse")
        b = Service(password="battery-staple")
        assert a.fingerprint() != b.fingerprint()
        assert "correct-horse" not in a.fingerprint()


class TestTypesAndFunctionsAreNamedByImportPath:
    def test_exception_types_in_a_retry_policy(self) -> None:
        a = RetryConfig(retry_on_exceptions=[ValueError])
        b = RetryConfig(retry_on_exceptions=[KeyError])
        assert a.fingerprint() != b.fingerprint()
        assert a.fingerprint() == RetryConfig(retry_on_exceptions=[ValueError]).fingerprint()

    def test_a_module_level_function(self) -> None:
        assert Service(hook=module_level_hook).fingerprint() != Service().fingerprint()

    def test_a_lambda_is_refused_naming_the_field(self) -> None:
        """A lambda has no name to identify it by, so two would share a key."""
        with pytest.raises(TypeError, match=r"Service\.hook"):
            Service(hook=lambda: None).fingerprint()

    def test_a_bound_method_is_refused(self) -> None:
        """Two instances' methods share a qualified name and differ in behaviour."""
        with pytest.raises(TypeError, match=r"Service\.hook.*_FINGERPRINT_EXCLUDE"):
            Service(hook=Plain().__eq__).fingerprint()

    def test_a_type_defined_inside_a_function_is_refused(self) -> None:
        class Local:
            pass

        with pytest.raises(TypeError, match=r"Service\.hook"):
            Service(hook=Local).fingerprint()

    def test_the_refusal_names_the_path_through_a_container(self) -> None:
        with pytest.raises(TypeError, match=r"Service\.labels\['k'\]"):
            Service(labels={"k": object()}).fingerprint()


class TestExtendingIt:
    def test_excluded_fields_are_left_out(self) -> None:
        assert (
            Streaming(on_token=lambda t: None).fingerprint()
            == Streaming(on_token=print).fingerprint()
        )

    def test_an_excluded_name_that_is_no_field_is_refused(self) -> None:
        with pytest.raises(ValueError, match=r"modle"):
            Misspelt().fingerprint()

    def test_the_exclusion_is_a_declared_policy(self) -> None:
        with pytest.raises(ValueError, match=r"_FINGERPRINT_EXCLUDE"):

            @dataclass(frozen=True)
            class Bare(StructuredConfig):
                model: str = "m"

                _FINGERPRINT_EXCLUDE: ClassVar[Any] = "model"

    def test_a_config_can_encode_a_value_type_of_its_own(self) -> None:
        a = Client(endpoint=Endpoint("https://a"))
        assert a.fingerprint() == Client(endpoint=Endpoint("https://a")).fingerprint()
        assert a.fingerprint() != Client(endpoint=Endpoint("https://b")).fingerprint()

    def test_what_the_extension_returns_is_kept_apart_from_a_plain_value(self) -> None:
        """An endpoint encoded as its URL is not the URL held as a string."""
        assert Client(endpoint=Endpoint("u")) != Client(endpoint="u")
        assert Client(endpoint=Endpoint("u")).fingerprint() != Client(endpoint="u").fingerprint()

    def test_what_the_extension_returns_must_be_encodable(self) -> None:
        with pytest.raises(TypeError, match=r"Echoing\.endpoint"):
            Echoing(endpoint=object()).fingerprint()

    def test_the_extension_is_consulted_once_per_value(self) -> None:
        """What it returns is encoded without it, however deep the value sits."""
        with pytest.raises(TypeError, match=r"Wrapping\.endpoint"):
            Wrapping(endpoint=object()).fingerprint()

    def test_a_field_equality_ignores_is_ignored_too(self) -> None:
        """Equal configs share a fingerprint, so ``compare=False`` reaches it."""
        assert Timed(note="a") == Timed(note="b")
        assert Timed(note="a").fingerprint() == Timed(note="b").fingerprint()


class TestATypeNoImportReachesIsRefused:
    """A class defined inside a function shares its name with every other one.

    Two calls of one factory make two classes of one name, which compare
    unequal; a fingerprint naming them by that name would make them one key.
    """

    def test_a_config_class_built_in_a_function(self) -> None:
        a, b = a_factory_class()(), a_factory_class()()
        assert a != b
        with pytest.raises(TypeError, match=r"Made.*module level"):
            a.fingerprint()

    def test_a_nested_config_of_such_a_class(self) -> None:
        with pytest.raises(TypeError, match=r"Service\.hook"):
            Service(hook=a_factory_class()()).fingerprint()

    def test_a_dataclass_of_such_a_class(self) -> None:
        @dataclass(frozen=True)
        class Local:
            n: int = 1

        with pytest.raises(TypeError, match=r"Service\.hook"):
            Service(hook=Local()).fingerprint()

    def test_an_enum_of_such_a_class(self) -> None:
        class Local(enum.Enum):
            A = 1

        with pytest.raises(TypeError, match=r"Service\.hook"):
            Service(hook=Local.A).fingerprint()


class TestValuesPythonCallsEqualShareOne:
    """Equal field values encode alike, whatever type spelled them."""

    def test_an_int_from_yaml_where_the_default_is_a_float(self) -> None:
        restored = RetryConfig.from_dict({"initial_delay": 1})
        assert restored == RetryConfig()
        assert restored.fingerprint() == RetryConfig().fingerprint()

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            (True, 1),
            (1, 1.0),
            (0.0, -0.0),
            (decimal.Decimal("1.0"), decimal.Decimal("1.00")),
            (decimal.Decimal("0.5"), 0.5),
            (fractions.Fraction(1, 2), 0.5),
            (float("inf"), decimal.Decimal("Infinity")),
            (bytearray(b"ab"), b"ab"),
            (pathlib.PurePosixPath("/a"), pathlib.PosixPath("/a")),
            (
                datetime.datetime(2026, 1, 1, 12, tzinfo=datetime.UTC),
                datetime.datetime(
                    2026, 1, 1, 13, tzinfo=datetime.timezone(datetime.timedelta(hours=1))
                ),
            ),
        ],
    )
    def test_equal_values(self, a: Any, b: Any) -> None:
        assert Service(hook=a) == Service(hook=b)
        assert Service(hook=a).fingerprint() == Service(hook=b).fingerprint()

    @pytest.mark.parametrize(
        ("a", "b"),
        [
            (decimal.Decimal("0.1"), 0.1),
            (0.5, "0.5"),
            (1, "1"),
            (
                datetime.datetime(2026, 1, 1, 12),
                datetime.datetime(2026, 1, 1, 12, tzinfo=datetime.UTC),
            ),
            (datetime.date(2026, 1, 1), datetime.datetime(2026, 1, 1)),
            (pathlib.PurePosixPath("/a"), pathlib.PureWindowsPath("/a")),
        ],
    )
    def test_unequal_values(self, a: Any, b: Any) -> None:
        assert Service(hook=a) != Service(hook=b)
        assert Service(hook=a).fingerprint() != Service(hook=b).fingerprint()

    @pytest.mark.parametrize("nan", [float("nan"), decimal.Decimal("NaN")])
    def test_nan_is_refused(self, nan: Any) -> None:
        """NaN is unequal to itself, so no key can stand for it."""
        with pytest.raises(ValueError, match=r"Service\.hook.*NaN"):
            Service(hook=nan).fingerprint()


def test_a_value_that_holds_itself_is_refused() -> None:
    looped: list[Any] = []
    looped.append(looped)
    with pytest.raises(ValueError, match=r"Service\.tags\[0\].*holds itself"):
        Service(tags=looped).fingerprint()


def test_the_encoding_is_pinned() -> None:
    """A change to the encoding changes every stored key, so it is a decision.

    Update this digest only on purpose, and say so in the changelog.
    """
    assert RetryConfig(max_attempts=4, retry_on_exceptions=[ValueError]).fingerprint() == _PINNED


#: Written when the encoding was; see the test above.
_PINNED = "48160c98eb794a0899430f7e0d5a17bf85e1d97e71d5ec6422efe37426ac87f2"
