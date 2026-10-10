"""A :class:`StructuredConfig` is compared field by field and is not hashable.

The base is ``@dataclass(frozen=True)`` with equality on, and every subclass is
the same. Left alone, ``dataclasses`` gives each one a ``__hash__`` over its
field tuple, so the type answers :class:`collections.abc.Hashable` True whatever
its fields hold, and ``hash()`` raises the moment one of them is a list or a
mapping. A config holding containers is the normal case, so the check would be
wrong for most of the family.

Equality cannot be the part that gives: ``from_dict(to_dict()) == cfg`` is the
base's own contract. So every subclass is declared unhashable, by
``__init_subclass__``, which runs before the subclass's ``@dataclass`` and
leaves ``__hash__`` in its dict where the decorator will not regenerate it.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field

import pytest

from dataknobs_common.structured_config import StructuredConfig
from dataknobs_common.testing import assert_structured_config_roundtrip


@dataclass(frozen=True)
class Listed(StructuredConfig):
    """The ordinary shape: a scalar and a container."""

    name: str = "x"
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Scalars(StructuredConfig):
    """Every field hashable, which is what made the generated hash look safe."""

    name: str = "x"
    size: int = 1


class TestASubclassIsNotHashable:
    """Every subclass answers the check honestly, whatever its fields hold."""

    def test_the_type_does_not_claim_to_be_hashable(self) -> None:
        assert not issubclass(Listed, Hashable)
        assert not isinstance(Listed(tags=["a"]), Hashable)

    def test_hashing_one_raises_the_ordinary_error(self) -> None:
        with pytest.raises(TypeError, match="unhashable"):
            hash(Listed(tags=["a"]))

    def test_a_config_of_scalars_is_not_hashable_either(self) -> None:
        """Hashability is a property of the type, not of what one value holds.

        A rule that let a config hash while its fields happened to allow it
        would change answer the day a list field was added, which is a
        breaking change nobody would see in the diff that made it.
        """
        assert not isinstance(Scalars(), Hashable)

    def test_the_base_answers_as_its_subclasses_do(self) -> None:
        assert not issubclass(StructuredConfig, Hashable)

    def test_equality_still_compares_the_fields(self) -> None:
        assert Listed.from_dict({"tags": ["a"]}) == Listed.from_dict({"tags": ["a"]})
        assert Listed(tags=["a"]) != Listed(tags=["b"])

    def test_the_roundtrip_still_holds(self) -> None:
        assert_structured_config_roundtrip(Listed(name="y", tags=["a", "b"]))

    def test_a_subclass_of_a_subclass_is_not_hashable(self) -> None:
        """Each subclass's own ``@dataclass`` would regenerate the hash.

        Which is why a declaration on the base body alone is not enough, and
        why the hook runs for every level.
        """

        @dataclass(frozen=True)
        class Deeper(Listed):
            extra: int = 0

        assert not isinstance(Deeper(), Hashable)


class TestASubclassThatWritesItsOwnHashKeepsIt:
    """The opt-out: a hash written in the class body is the subclass's answer."""

    def test_a_hand_written_hash_is_kept(self) -> None:
        @dataclass(frozen=True, eq=False)
        class Versioned(StructuredConfig):
            version: int = 1
            notes: list[str] = field(default_factory=list)

            def __eq__(self, other: object) -> bool:
                return isinstance(other, Versioned) and other.version == self.version

            def __hash__(self) -> int:
                return hash(self.version)

        a, b = Versioned(version=2, notes=["a"]), Versioned(version=2, notes=["b"])
        assert isinstance(a, Hashable)
        assert hash(a) == hash(b)
        assert a == b


class TestEqualityWrittenWithoutAHashIsJudgedByTheResult:
    """``__eq__`` without a hash of its own is refused only where it goes wrong.

    Python records a class body that defines ``__eq__`` without ``__hash__`` by
    setting ``__hash__ = None`` in it, and an explicit ``__hash__ = None`` looks
    the same. Under ``@dataclass`` with ``eq=True``, ``dataclasses`` reads that
    pair as *implicit* and generates a hash over the fields, which the new
    equality does not agree with. Under ``eq=False``, or with no decorator, the
    ``None`` stays and the class is honest: its own equality, and unhashable.

    Nothing at class definition can see which decorator follows, so the class
    is judged at its first construction, by what the decorator actually did.
    """

    def test_equality_alone_under_the_default_decorator_is_refused(self) -> None:
        # The missing `__hash__` is the subject: this is the class refused.
        @dataclass(frozen=True)
        class ByName(StructuredConfig):  # noqa: PLW1641
            name: str = "x"

            def __eq__(self, other: object) -> bool:
                return isinstance(other, ByName) and other.name == self.name

        with pytest.raises(TypeError, match=r"ByName defines __eq__.*eq=False"):
            ByName()

    def test_an_explicit_none_beside_equality_is_refused_the_same_way(self) -> None:
        @dataclass(frozen=True)
        class ByName(StructuredConfig):
            name: str = "x"

            def __eq__(self, other: object) -> bool:
                return isinstance(other, ByName) and other.name == self.name

            __hash__ = None  # type: ignore[assignment]

        with pytest.raises(TypeError, match=r"ByName defines __eq__"):
            ByName()

    def test_a_subclass_of_a_refused_class_is_refused_too(self) -> None:
        @dataclass(frozen=True)
        class ByName(StructuredConfig):  # noqa: PLW1641
            name: str = "x"

            def __eq__(self, other: object) -> bool:
                return isinstance(other, ByName) and other.name == self.name

        @dataclass(frozen=True)
        class Below(ByName):
            extra: int = 0

        with pytest.raises(TypeError, match=r"ByName defines __eq__"):
            Below()

    def test_equality_alone_under_eq_false_keeps_it_and_is_unhashable(self) -> None:
        @dataclass(frozen=True, eq=False)
        class ByName(StructuredConfig):  # noqa: PLW1641
            name: str = "x"
            notes: list[str] = field(default_factory=list)

            def __eq__(self, other: object) -> bool:
                return isinstance(other, ByName) and other.name == self.name

        assert ByName(notes=["a"]) == ByName(notes=["b"])
        assert not isinstance(ByName(), Hashable)

    def test_equality_alone_on_an_undecorated_subclass_is_accepted(self) -> None:
        class Loose(Listed):  # noqa: PLW1641
            def __eq__(self, other: object) -> bool:
                return isinstance(other, Loose) and other.name == self.name

        assert Loose(tags=["a"]) == Loose(tags=["b"])
        assert not isinstance(Loose(), Hashable)

    def test_equality_with_a_hash_is_accepted(self) -> None:
        @dataclass(frozen=True)
        class ByName(StructuredConfig):
            name: str = "x"

            def __eq__(self, other: object) -> bool:
                return isinstance(other, ByName) and other.name == self.name

            def __hash__(self) -> int:
                return hash(self.name)

        assert hash(ByName()) == hash(ByName())

    def test_a_slotted_class_with_equality_alone_is_refused(self) -> None:
        @dataclass(frozen=True, slots=True)
        class ByName(StructuredConfig):  # noqa: PLW1641
            name: str = "x"

            def __eq__(self, other: object) -> bool:
                return isinstance(other, ByName) and other.name == self.name

        with pytest.raises(TypeError, match=r"ByName defines __eq__"):
            ByName()


class TestSlotsAreSupported:
    """``slots=True`` rebuilds the class, which runs the hook a second time.

    The rebuilt class's dict is the decorated one: it holds the ``__eq__`` the
    decorator generated beside the ``None`` the hook wrote on the first pass.
    Read as a body, that pair is the refused shape, so a rebuild has to be
    recognised as one rather than judged again.
    """

    def test_a_slotted_config_is_defined_and_unhashable(self) -> None:
        @dataclass(frozen=True, slots=True)
        class Slotted(StructuredConfig):
            n: int = 0
            tags: list[str] = field(default_factory=list)

        assert "__slots__" in Slotted.__dict__
        assert not isinstance(Slotted(), Hashable)
        assert Slotted(n=1) == Slotted(n=1)
        assert_structured_config_roundtrip(Slotted(n=1, tags=["a"]))

    def test_a_slotted_config_keeps_a_hash_it_writes(self) -> None:
        @dataclass(frozen=True, slots=True)
        class Slotted(StructuredConfig):
            n: int = 0

            def __hash__(self) -> int:
                return hash(self.n)

        assert hash(Slotted(n=3)) == hash(3)


@dataclass(frozen=True, eq=False)
class ByVersion(StructuredConfig):
    """Identified by its version number, as a stored config version is."""

    version: int = 1
    notes: list[str] = field(default_factory=list)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ByVersion) and other.version == self.version

    def __hash__(self) -> int:
        return hash(self.version)


class TestAHashWrittenByHandIsInherited:
    """A subclass of a config that writes its own hash inherits that hash.

    Every subclass's dict holds either ``None`` or a hash its body wrote, so an
    inherited hash that is not ``None`` was written by hand. Handing it on keeps
    an undecorated subclass's equality and hash in step, since it inherits the
    equality too. A decorated subclass compares every field, which is finer than
    the parent's equality, so two equal instances also agree on the parent's
    hash.
    """

    def test_an_undecorated_subclass_keeps_the_hash_and_the_equality(self) -> None:
        class Undecorated(ByVersion):
            pass

        a, b = Undecorated(version=2, notes=["a"]), Undecorated(version=2, notes=["b"])
        assert a == b
        assert hash(a) == hash(b)

    def test_a_decorated_subclass_hashes_consistently_with_its_field_equality(self) -> None:
        @dataclass(frozen=True)
        class Decorated(ByVersion):
            label: str = "x"

        assert Decorated(version=2, label="y") == Decorated(version=2, label="y")
        assert hash(Decorated(version=2, label="y")) == hash(Decorated(version=2, label="z"))

    def test_a_hash_from_a_mixin_that_is_not_a_config_is_not_inherited(self) -> None:
        """Only a hash a config wrote is the family's opt-out."""

        class Keyed:
            def __hash__(self) -> int:
                return 0

        @dataclass(frozen=True)
        class Mixed(Keyed, StructuredConfig):
            n: int = 0

        assert not isinstance(Mixed(), Hashable)
