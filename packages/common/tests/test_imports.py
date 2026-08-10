"""Tests for :mod:`dataknobs_common.imports`.

The workspace-level guard in ``tests/test_dotted_path_agreement.py`` checks
that every *consumer* of this family behaves identically. This file checks the
family itself: each ``reason``, both separators, the shape checks, and the two
properties that are invisible to an assertion on the return value —
``resolve_class`` not constructing anything, and the error text staying bounded.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pytest

from dataknobs_common.exceptions import (
    ConfigurationError,
    DottedPathError,
    DottedPathReason,
    DottedPathTypeError,
)
from dataknobs_common.imports import (
    dotted_path,
    resolve_callable,
    resolve_class,
    resolve_dotted,
    resolve_optional_callable,
)

HERE = __name__


# ── Targets ───────────────────────────────────────────────────────────

constructions = 0


def a_function(*args: Any, **kwargs: Any) -> str:
    return "resolved"


not_callable_at_all = 42


@runtime_checkable
class Shaped(Protocol):
    def shaped_method(self) -> None: ...


class Conforming:
    def __init__(self) -> None:
        global constructions
        constructions += 1

    def shaped_method(self) -> None: ...


class NotConforming:
    def __init__(self) -> None:
        global constructions
        constructions += 1


class ExplodesOnConstruction:
    def __init__(self) -> None:
        raise AssertionError("constructed, and it should not have been")

    def shaped_method(self) -> None: ...


@pytest.fixture(autouse=True)
def _reset_counter():
    global constructions
    constructions = 0
    yield


# ── resolve_dotted ────────────────────────────────────────────────────


@pytest.mark.parametrize("sep", [":", "."], ids=["colon", "dot"])
def test_resolve_dotted_accepts_both_separators(sep: str) -> None:
    assert resolve_dotted(f"{HERE}{sep}a_function") is a_function


def test_resolve_dotted_returns_a_non_callable_without_complaint() -> None:
    """The base of the family checks nothing — that is what it is for."""
    assert resolve_dotted(f"{HERE}:not_callable_at_all") == 42


@pytest.mark.parametrize(
    "ref",
    ["", "   ", "nodots", ":name", "module:", ".name", "module."],
    ids=["empty", "blank", "no-separator", "no-module", "no-attr", "dot-no-module", "dot-no-attr"],
)
def test_malformed_references_are_rejected(ref: str) -> None:
    with pytest.raises(DottedPathError) as excinfo:
        resolve_dotted(ref)
    assert excinfo.value.reason is DottedPathReason.MALFORMED


def test_a_non_string_reference_is_malformed_not_a_crash() -> None:
    """A YAML author writing ``function: 42`` gets a config error, not a TypeError."""
    with pytest.raises(DottedPathError) as excinfo:
        resolve_dotted(42)  # type: ignore[arg-type]
    assert excinfo.value.reason is DottedPathReason.MALFORMED


def test_a_missing_module_reports_module_not_found() -> None:
    with pytest.raises(DottedPathError) as excinfo:
        resolve_dotted("no_such_module_anywhere:name")
    assert excinfo.value.reason is DottedPathReason.MODULE_NOT_FOUND
    assert excinfo.value.ref == "no_such_module_anywhere:name"


def test_a_missing_attribute_reports_attribute_not_found_and_suggests() -> None:
    with pytest.raises(DottedPathError) as excinfo:
        resolve_dotted(f"{HERE}:no_such_attribute")
    assert excinfo.value.reason is DottedPathReason.ATTRIBUTE_NOT_FOUND
    # The enumeration is the one thing worth keeping from the resolver this
    # replaced: a typo'd name is usually close to a real one.
    assert "a_function" in str(excinfo.value)


def test_the_suggestions_are_the_module_s_own_symbols_not_its_imports() -> None:
    """Otherwise the list is mostly imports and truncation cuts the useful half.

    This module imports ``Any``, ``Protocol``, ``pytest`` and four exception
    types — all public and all callable. Sorted alphabetically and cut at ten,
    an unfiltered list reaches none of the functions defined here, which are
    the only candidates a typo could have meant.
    """
    with pytest.raises(DottedPathError) as excinfo:
        resolve_dotted(f"{HERE}:a_functoin")

    message = str(excinfo.value)
    assert "a_function" in message
    assert "Protocol" not in message


def test_a_re_export_module_still_suggests_something() -> None:
    """The filter must not empty the list where it is most needed.

    A package ``__init__`` defines nothing of its own — every public name in
    it is an import — so own-module filtering would leave nothing to suggest.
    """
    with pytest.raises(DottedPathError) as excinfo:
        resolve_dotted("dataknobs_common:no_such_export")

    assert "(none)" not in str(excinfo.value)


def test_only_one_attribute_is_looked_up() -> None:
    """``module:Outer.Inner`` is not supported, and says so rather than half-working.

    With a single ``rpartition``, ``a.b:C.D`` splits on the colon and then
    fails to find an attribute literally named ``C.D`` — the right answer, but
    the reason matters: it must read as an unresolvable attribute, not as a
    successful traversal of something else.
    """
    with pytest.raises(DottedPathError) as excinfo:
        resolve_dotted(f"{HERE}:Conforming.shaped_method")
    assert excinfo.value.reason is DottedPathReason.ATTRIBUTE_NOT_FOUND


# ── resolve_callable ──────────────────────────────────────────────────


@pytest.mark.parametrize("sep", [":", "."], ids=["colon", "dot"])
def test_resolve_callable_accepts_both_separators(sep: str) -> None:
    assert resolve_callable(f"{HERE}{sep}a_function") is a_function


def test_resolve_callable_rejects_a_non_callable() -> None:
    with pytest.raises(DottedPathError) as excinfo:
        resolve_callable(f"{HERE}:not_callable_at_all")
    assert excinfo.value.reason is DottedPathReason.NOT_CALLABLE


def test_a_class_is_callable_so_resolve_callable_accepts_it() -> None:
    """Documented, not incidental — and the reason ``resolve_class`` exists.

    Two call sites resolved a *class* through the callable resolver and got
    away with it because a class passes ``callable()``. It is a real gap in
    what that check can promise, not a bug in this function.
    """
    assert resolve_callable(f"{HERE}:Conforming") is Conforming
    assert constructions == 0


# ── resolve_class ─────────────────────────────────────────────────────


@pytest.mark.parametrize("sep", [":", "."], ids=["colon", "dot"])
def test_resolve_class_accepts_both_separators(sep: str) -> None:
    assert resolve_class(f"{HERE}{sep}Conforming", Shaped) is Conforming


def test_resolve_class_returns_the_class_and_constructs_nothing() -> None:
    """The property the signature exists to guarantee."""
    resolved = resolve_class(f"{HERE}:Conforming", Shaped)

    assert resolved is Conforming
    assert constructions == 0, "resolve_class constructed the target"


def test_a_wrong_shape_class_is_rejected_without_being_constructed() -> None:
    with pytest.raises(DottedPathTypeError) as excinfo:
        resolve_class(f"{HERE}:NotConforming", Shaped)

    assert excinfo.value.expected is Shaped
    assert constructions == 0, "the wrong-shape class ran its __init__"


def test_the_constructor_of_a_conforming_class_is_still_not_run() -> None:
    """A conforming target whose ``__init__`` would fail still resolves.

    Stronger than the counter, and immune to someone "simplifying" the
    counter away: this class raises if constructed at all, so the test fails
    loudly rather than by an assertion on a number.
    """
    assert resolve_class(f"{HERE}:ExplodesOnConstruction", Shaped) is ExplodesOnConstruction


def test_a_module_level_function_is_not_a_class() -> None:
    with pytest.raises(DottedPathTypeError):
        resolve_class(f"{HERE}:a_function", Shaped)


def test_a_non_runtime_checkable_base_raises_typeerror_unwrapped() -> None:
    """A programmer error in the *caller*, deliberately not dressed as config.

    Wrapping this in ``ConfigurationError`` would point the reader at a config
    file that is perfectly fine.
    """

    class NotRuntimeCheckable(Protocol):
        def whatever(self) -> None: ...

    with pytest.raises(TypeError):
        resolve_class(f"{HERE}:Conforming", NotRuntimeCheckable)


# ── resolve_optional_callable ─────────────────────────────────────────


def test_none_resolves_to_none() -> None:
    assert resolve_optional_callable(None, field_name="hook", owner="thing") is None


def test_a_present_but_broken_reference_still_raises() -> None:
    """ "Omitted" and "wrong" are different states; only the first is optional."""
    with pytest.raises(DottedPathError):
        resolve_optional_callable(f"{HERE}:no_such_attribute", field_name="hook", owner="thing")


def test_the_error_names_the_config_site() -> None:
    with pytest.raises(DottedPathError) as excinfo:
        resolve_optional_callable(
            f"{HERE}:no_such_attribute", field_name="dedup_key", owner="my_source"
        )

    message = str(excinfo.value)
    assert "dedup_key" in message
    assert "my_source" in message
    # The underlying reason survives the re-wrap — a caller branching on
    # `reason` must not be defeated by the lift that added the config site.
    assert excinfo.value.reason is DottedPathReason.ATTRIBUTE_NOT_FOUND


# ── Error shape ───────────────────────────────────────────────────────


def test_both_types_are_configuration_errors() -> None:
    """So existing ``except ConfigurationError`` clauses keep working."""
    assert issubclass(DottedPathError, ConfigurationError)
    assert issubclass(DottedPathTypeError, ConfigurationError)


def test_the_two_error_types_are_siblings_not_parent_and_child() -> None:
    """The asymmetry is the contract, so it gets an assertion of its own.

    A caller writing the obvious lenient handler — ``except DottedPathError:
    if optional: return None`` — must not swallow a shape mismatch. Making
    ``DottedPathTypeError`` a subclass would break that silently, and every
    other test here would still pass.
    """
    assert not issubclass(DottedPathTypeError, DottedPathError)
    assert not issubclass(DottedPathError, DottedPathTypeError)


def test_a_lenient_handler_cannot_swallow_a_shape_mismatch() -> None:
    """The same property, stated as the caller experiences it."""
    with pytest.raises(DottedPathTypeError):
        try:
            resolve_class(f"{HERE}:NotConforming", Shaped)
        except DottedPathError:  # the "optional: true" handler
            pytest.fail("the lenient handler caught a shape mismatch")


def test_the_message_does_not_carry_the_underlying_exception_text() -> None:
    """Bounded messages: the detail belongs on ``__cause__``.

    ``import_module`` executes the target, so the caught exception's text is
    arbitrary consumer output — and these errors are rendered to HTTP clients
    by surfaces that map ``ConfigurationError``.
    """
    with pytest.raises(DottedPathError) as excinfo:
        resolve_dotted("no_such_module_anywhere:name")

    message = str(excinfo.value)
    assert "ModuleNotFoundError" in message, "the failure type should be named"
    assert excinfo.value.__cause__ is not None
    assert str(excinfo.value.__cause__) not in message


def test_an_exploding_module_reports_import_failed_not_module_not_found(
    tmp_path, monkeypatch
) -> None:
    """The import runs the target, so the failure is not always an ImportError.

    Catching only ``ImportError`` here would let a ``RuntimeError`` raised at
    the target's module scope escape untyped, past every caller's ``except``.

    And the reason must not be ``module_not_found``: the module was found. A
    caller skipping absent optional dependencies on that reason would
    otherwise silently skip a module that is installed and raising — the two
    states want opposite responses, and only one is safe to swallow.
    """
    import sys

    module = tmp_path / "dk_exploding_fixture.py"
    module.write_text("raise RuntimeError('secret: hunter2')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("dk_exploding_fixture", None)

    with pytest.raises(DottedPathError) as excinfo:
        resolve_dotted("dk_exploding_fixture:anything")

    assert excinfo.value.reason is DottedPathReason.IMPORT_FAILED
    assert "RuntimeError" in str(excinfo.value)
    assert "hunter2" not in str(excinfo.value)


def test_a_module_missing_a_dependency_reports_module_not_found(tmp_path, monkeypatch) -> None:
    """The target exists; something it imports does not.

    This is the optional-dependency case — a tool whose module imports an SDK
    the deployment did not install — so it belongs with "not installed"
    rather than with "present and broken", even though the target module
    itself was found. Telling it apart from a mistyped path would take the
    deployment's intent, which this layer does not have.
    """
    import sys

    module = tmp_path / "dk_missing_dep_fixture.py"
    module.write_text("import dk_no_such_sdk_anywhere\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("dk_missing_dep_fixture", None)

    with pytest.raises(DottedPathError) as excinfo:
        resolve_dotted("dk_missing_dep_fixture:anything")

    assert excinfo.value.reason is DottedPathReason.MODULE_NOT_FOUND


def test_a_broken_from_import_reports_import_failed(tmp_path, monkeypatch) -> None:
    """A plain ``ImportError`` means the module began executing and failed.

    ``ModuleNotFoundError`` is an ``ImportError`` subclass, so the ordering of
    the two ``except`` clauses is what places this case: the module was found
    and its own ``from x import y`` did not resolve. That is a defect (or a
    version skew), not an absent dependency.
    """
    import sys

    (tmp_path / "dk_partial_fixture.py").write_text("VALUE = 1\n")
    module = tmp_path / "dk_broken_from_fixture.py"
    module.write_text("from dk_partial_fixture import MISSING_NAME\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("dk_broken_from_fixture", None)
    sys.modules.pop("dk_partial_fixture", None)

    with pytest.raises(DottedPathError) as excinfo:
        resolve_dotted("dk_broken_from_fixture:anything")

    assert excinfo.value.reason is DottedPathReason.IMPORT_FAILED


def test_the_reason_vocabulary_rejects_a_typo() -> None:
    """``reason=`` normalizes, following ``PackResolutionError``."""
    with pytest.raises(ValueError):
        DottedPathError("x", ref="a:b", reason="modul_not_found")


def test_the_error_carries_ref_and_reason_in_context() -> None:
    error = DottedPathError("x", ref="a:b", reason=DottedPathReason.MALFORMED)
    assert error.ref == "a:b"
    assert error.context["ref"] == "a:b"
    assert error.context["reason"] is DottedPathReason.MALFORMED


def _write_lazy_export_module(tmp_path, monkeypatch, name: str, raises: str):
    """A PEP 562 module whose ``__getattr__`` raises *raises*.

    The shape ``dataknobs_common.events`` uses for ``SqsEventBus``: the
    attribute is not in the namespace, and touching it runs an import.
    """
    import sys

    (tmp_path / f"{name}.py").write_text(f"def __getattr__(attr):\n    raise {raises}\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(name, None)


def test_a_lazy_export_that_raises_is_still_a_dotted_path_error(tmp_path, monkeypatch) -> None:
    """The attribute lookup is a *second* execution point, not just a read.

    A PEP 562 module-level ``__getattr__`` runs arbitrary code, and the
    standard use of it is a lazy export that imports an optional dependency
    on first access — which is what ``dataknobs_common.events`` does for
    ``SqsEventBus``.

    Catching only ``AttributeError`` there let the resulting ``ImportError``
    escape raw, so a caller's ``except DottedPathError`` did not match and
    ``optional: true`` did not cover it. The type is the property under test:
    every failure of this function must arrive as one exception type.
    """
    _write_lazy_export_module(
        tmp_path,
        monkeypatch,
        "dk_lazy_raises_fixture",
        "ImportError('lazy export is broken')",
    )

    with pytest.raises(DottedPathError) as excinfo:
        resolve_dotted("dk_lazy_raises_fixture:SomeExport")

    # Not `attribute_not_found`: the module did not say "no such attribute",
    # it ran code and that code failed. Reporting the miss would send the
    # reader to look for a typo in a name that is spelled correctly.
    assert excinfo.value.reason is DottedPathReason.IMPORT_FAILED
    assert isinstance(excinfo.value.__cause__, ImportError)


def test_a_lazy_export_missing_its_dependency_reports_module_not_found(
    tmp_path, monkeypatch
) -> None:
    """The optional-dependency case, at the attribute site.

    Classified by the same rule as the import site — a shared helper, so the
    two execution points cannot report the same failure differently. This is
    the reason a caller's ``optional: true`` is entitled to swallow.
    """
    _write_lazy_export_module(
        tmp_path,
        monkeypatch,
        "dk_lazy_missing_dep_fixture",
        "ModuleNotFoundError(\"No module named 'dk_no_such_sdk'\")",
    )

    with pytest.raises(DottedPathError) as excinfo:
        resolve_dotted("dk_lazy_missing_dep_fixture:SomeExport")

    assert excinfo.value.reason is DottedPathReason.MODULE_NOT_FOUND


def test_a_hostile_dir_cannot_replace_the_error(tmp_path, monkeypatch) -> None:
    """The suggestion builder runs inside an ``except`` — it must not raise.

    ``_suggestions`` walks ``dir(module)`` calling ``getattr`` on each name,
    and ``getattr(..., None)`` swallows ``AttributeError`` only. A module
    advertising a name whose access raises something else would take the
    builder down *while it was building the message*, replacing the
    ``DottedPathError`` the caller is owed with an unrelated one chained
    behind "during handling of the above exception".
    """
    import sys

    (tmp_path / "dk_hostile_dir_fixture.py").write_text(
        "def __dir__():\n"
        "    return ['landmine', 'real_name']\n"
        "\n"
        "def real_name():\n"
        "    return 1\n"
        "\n"
        "def __getattr__(attr):\n"
        "    if attr == 'landmine':\n"
        "        raise ImportError('boom')\n"
        "    raise AttributeError(attr)\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("dk_hostile_dir_fixture", None)

    with pytest.raises(DottedPathError) as excinfo:
        resolve_dotted("dk_hostile_dir_fixture:no_such_attribute")

    assert excinfo.value.reason is DottedPathReason.ATTRIBUTE_NOT_FOUND


# ---------------------------------------------------------------------------
# dotted_path — the inverse
# ---------------------------------------------------------------------------


def _a_module_level_function() -> int:
    return 7


class _AModuleLevelClass:
    class Nested:
        pass


def test_a_dotted_path_round_trips_through_the_resolver() -> None:
    """The whole point: what this spells, ``resolve_dotted`` reads back."""
    assert resolve_dotted(dotted_path(_AModuleLevelClass)) is _AModuleLevelClass
    assert resolve_dotted(dotted_path(_a_module_level_function)) is _a_module_level_function


def test_a_dotted_path_uses_the_canonical_separator() -> None:
    """``module:name``, the form the module docstring tells callers to prefer."""
    assert dotted_path(_AModuleLevelClass) == f"{HERE}:_AModuleLevelClass"


def test_a_nested_qualname_is_refused_rather_than_spelled() -> None:
    """``resolve_dotted`` performs exactly one attribute lookup.

    Spelling ``module:Outer.Inner`` would produce a string this family cannot
    read back, so the failure belongs here — where the object is in hand and
    can be named — rather than at resolution time, where only the string is.
    """
    with pytest.raises(ValueError, match="nested"):
        dotted_path(_AModuleLevelClass.Nested)


def test_a_closure_is_refused() -> None:
    """A local function's qualname carries ``<locals>`` — also unresolvable."""

    def _inner() -> None:  # pragma: no cover - never called
        pass

    with pytest.raises(ValueError, match="nested"):
        dotted_path(_inner)


#: A genuine module-level lambda, held in a tuple so that assigning one to a
#: bare name is not what is under test. Its ``__qualname__`` is ``<lambda>`` —
#: no dot, so the nested-qualname check above does not see it.
_LAMBDAS = (lambda: 7,)


def test_a_module_level_lambda_is_refused() -> None:
    """``<lambda>`` is not a name, so nothing can look it up.

    It reaches here as the one unresolvable qualname with no dot in it: a
    closure's ``f.<locals>.g`` is caught as nested, but a lambda defined at
    module scope is a plain ``<lambda>`` and would be spelled as
    ``module:<lambda>`` — a string ``resolve_dotted`` raises on, at the far end
    of whatever generated it.
    """
    with pytest.raises(ValueError, match="not a name"):
        dotted_path(_LAMBDAS[0])


def test_an_object_with_no_module_metadata_is_refused() -> None:
    """An instance carries neither attribute; naming it beats a confusing path."""
    with pytest.raises(ValueError, match="__qualname__"):
        dotted_path(_AModuleLevelClass())


def test_a_main_module_target_is_refused() -> None:
    """``__main__`` names a different object in every process that imports it."""

    class _Fake:
        __module__ = "__main__"
        __qualname__ = "Whatever"

    with pytest.raises(ValueError, match="__main__"):
        dotted_path(_Fake)
