"""``_create_resource`` resolves a dotted class path like everything else does.

A ``custom`` resource names its provider with a dotted ``class:`` string, which
is the same operation the rest of the workspace routes through
``dataknobs_common.imports``. This file pins the three properties that
distinguishes doing it once from doing it again locally:

* an unimportable path is a **configuration error**, not a bare
  ``ModuleNotFoundError`` escaping from ``importlib``;
* a wrong-shape target is rejected **before** it is constructed;
* a resource type the builder cannot build says so as a ``ValueError``, the
  error its own docstring documents.

The last is not hypothetical. ``llm`` and ``vector_store`` stayed in the
built-in type map after the modules behind them were migrated out of this
package, so a configuration naming either got a ``ModuleNotFoundError`` naming
an internal module path — from a type the config schema still offers.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from dataknobs_common.exceptions import ConfigurationError
from dataknobs_fsm.config.builder import FSMBuilder
from dataknobs_fsm.config.schema import ResourceConfig
from dataknobs_fsm.resources.base import IResourceProvider

from tests import _resource_fixtures as fixtures

FIXTURES = "tests._resource_fixtures"


@pytest.fixture(autouse=True)
def _reset_instantiation_counter() -> Iterator[None]:
    fixtures.reset_instantiations()
    yield


def _custom(class_path: str, **extra: object) -> ResourceConfig:
    """A ``custom`` resource config naming *class_path* as its provider."""
    return ResourceConfig(
        name="fixture", type="custom", config={"class": class_path, **extra}
    )


@pytest.mark.parametrize("separator", [":", "."], ids=["colon", "dot"])
def test_a_custom_resource_class_resolves_by_either_separator(
    separator: str,
) -> None:
    """``module:Name`` and ``module.Name`` mean the same thing here too.

    This site accepted only ``.``, so the same provider was nameable under one
    spelling and not the other depending on which config key it was written
    under. Nothing about a dotted path depends on the subsystem reading it.
    """
    resource = FSMBuilder()._create_resource(
        _custom(f"{FIXTURES}{separator}ConformingResource")
    )

    assert isinstance(resource, fixtures.ConformingResource)
    assert isinstance(resource, IResourceProvider)


def test_an_unimportable_custom_class_raises_a_configuration_error() -> None:
    """A typo in ``class:`` is a config fault, reportable as one.

    It used to leave ``importlib``'s own ``ModuleNotFoundError`` to escape
    through ``FSMBuilder.build``, which no caller writing
    ``except ConfigurationError`` around FSM construction would catch.
    """
    with pytest.raises(ConfigurationError):
        FSMBuilder()._create_resource(_custom("no_such_module:Resource"))


def test_a_present_module_with_a_missing_attribute_is_a_configuration_error() -> None:
    """The other half of a typo: right module, wrong name."""
    with pytest.raises(ConfigurationError):
        FSMBuilder()._create_resource(_custom(f"{FIXTURES}:NoSuchResource"))


def test_a_wrong_shape_custom_class_is_rejected_before_it_is_constructed() -> None:
    """A class that is not a provider must not run its ``__init__``.

    The site resolved a class and instantiated it with no shape check at all,
    so a mistyped ``class:`` ran an unrelated constructor — arbitrary code,
    with whatever side effects it has — and was noticed only if the resulting
    object happened to fail later.

    The counter is what gives this teeth: asserting only that the call raised
    would pass against an implementation that constructed the object, checked
    it, threw it away and raised.
    """
    with pytest.raises(ConfigurationError):
        FSMBuilder()._create_resource(_custom(f"{FIXTURES}:NotAResource"))

    assert fixtures.instantiations.count == 0, (
        "the wrong-shape class was constructed before being rejected "
        f"({fixtures.instantiations.count} construction(s))"
    )


def test_a_custom_provider_is_not_handed_the_path_that_named_it() -> None:
    """``class`` selects the provider; it is not one of its arguments.

    The builder read ``class`` out of the resource config and then passed that
    same config through as keyword arguments, so every custom provider was
    constructed with a stray ``class="pkg.mod:Name"``. It can never be a
    declared parameter — ``class`` is a reserved word — so the only provider
    that survived it was one absorbing ``**kwargs``, which is why the fixtures
    written alongside the resolver adoption did not see it.
    """
    resource = FSMBuilder()._create_resource(
        _custom(f"{FIXTURES}:StrictSignatureResource", param1="value1")
    )

    assert isinstance(resource, fixtures.StrictSignatureResource)
    assert resource.param1 == "value1", "declared parameters must still arrive"


def test_a_provider_that_defines_no_init_is_constructed() -> None:
    """A conforming provider need not declare a constructor at all.

    ``IResourceProvider`` is a method-only Protocol, so a class satisfying it
    may inherit ``object.__init__`` — a slot wrapper. Reading parameter names
    off ``__init__.__code__`` raised ``AttributeError`` on exactly that shape;
    ``inspect.signature`` reports it as taking nothing, which is true.
    """
    resource = FSMBuilder()._create_resource(_custom(f"{FIXTURES}:NoInitResource"))

    assert isinstance(resource, fixtures.NoInitResource)


def test_a_custom_resource_without_a_class_is_still_a_value_error() -> None:
    """The pre-resolution guard is unchanged by adopting the resolver."""
    with pytest.raises(ValueError, match="requires 'class'"):
        FSMBuilder()._create_resource(
            ResourceConfig(name="fixture", type="custom", config={})
        )


@pytest.mark.parametrize("resource_type", ["llm", "vector_store"])
def test_a_type_the_builder_cannot_build_reports_it_as_unsupported(
    resource_type: str,
) -> None:
    """A built-in type whose module is gone must fail like an unknown one.

    ``ResourceType`` still offers ``llm`` and ``vector_store``, so a config may
    legitimately name them; the modules behind them moved to other packages.
    The honest answer is the ``ValueError`` this method's docstring already
    promises for a type it cannot build — not a ``ModuleNotFoundError`` naming
    a module path that no longer exists.
    """
    with pytest.raises(ValueError, match="Unsupported resource type"):
        FSMBuilder()._create_resource(
            ResourceConfig(name="fixture", type=resource_type, config={})
        )


@pytest.mark.parametrize(
    ("resource_type", "config"),
    [
        ("database", {}),
        ("async_database", {}),
        ("filesystem", {}),
        ("http", {"base_url": "https://service.invalid"}),
    ],
)
def test_the_builtin_types_that_do_resolve_still_do(
    resource_type: str, config: dict[str, object]
) -> None:
    """Regression guard for the entries that are not going anywhere.

    The shape check this adoption adds is only correct if *every* surviving
    built-in passes it — ``IResourceProvider`` is what ``register_provider``
    declares, and these are what it is handed. So ``http`` is here despite
    requiring an argument the other three do not; carrying a config per type
    costs one column and keeps the claim and the coverage the same size.
    """
    resource = FSMBuilder()._create_resource(
        ResourceConfig(name="fixture", type=resource_type, config=config)
    )

    assert isinstance(resource, IResourceProvider)
