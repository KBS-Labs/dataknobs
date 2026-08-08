"""Every public dict-merge entry point in the workspace agrees.

There used to be five independent implementations of dict-deep-merge, and one
of them -- reachable through ``ConfigLoader.merge_configs`` -- *extended* lists
where the other four replaced them. It disagreed with a sibling function
twenty modules away in its own package, and stayed green for as long as it did
because the test pointed at it asserted only the scalar fields all five agreed
about.

They are one implementation now (``dataknobs_config.deep_merge``). This guard
is what makes a sixth copy fail loudly instead of drifting quietly: it drives
each surviving entry point through its **public** API, never through the shared
helper, so a caller that stops delegating is caught by the semantics changing
rather than by anyone noticing a new private function.

Filed here rather than in any one package because the subject is agreement
*between* packages -- under ``bots`` it would read as a bots test and be moved
or deleted during a bots refactor.

The parametrized ``(name, merge)`` table is the shape, not an accident: the
same guard is wanted for other consolidated primitives, and six separate test
functions would not be copyable.

Two entry points reach the merge through machinery that is not the merge, and
a future change to either would red-light this file for reasons having nothing
to do with merging:

* ``_via_inheritable_loader`` passes the result through ``substitute_env_vars``
  (``~`` expansion and ``${}`` substitution). Benign for this fixture; a value
  containing either would make that one entry point disagree spuriously.
* ``_via_bot_config_builder`` depends on ``DynaBotConfigBuilder.build()``
  tolerating unknown top-level keys, which is how the fixture survives a
  validating path. If bots ever rejects unknown keys, a bots-owned change
  fails a workspace guard with a message about list merging.
"""

from typing import Any, Callable

import pytest

# One fixture exercising all three value kinds a merge has to decide about.
FIXTURE_BASE: dict[str, Any] = {
    "scalar": 1,
    "nested": {"keep": "a", "beat": "b"},
    "listy": [1, 2],
    # `deep` exists because the divergence lived in the *recursive* branch. A
    # top-level list alone would not have caught it.
    "deep": {"inner": {"listy": [1, 2]}},
}
FIXTURE_OVERRIDE: dict[str, Any] = {
    "scalar": 2,
    "nested": {"beat": "B", "add": "c"},
    "listy": [3],
    "deep": {"inner": {"listy": [3]}},
}
EXPECTED: dict[str, Any] = {
    "scalar": 2,
    "nested": {"keep": "a", "beat": "B", "add": "c"},
    "listy": [3],
    "deep": {"inner": {"listy": [3]}},
}


def _via_deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """The canonical implementation, called directly."""
    from dataknobs_config import deep_merge

    return deep_merge(base, override)


def _via_inheritable_loader(
    base: dict[str, Any], override: dict[str, Any]
) -> dict[str, Any]:
    """A child config resolving its ``extends:`` parent."""
    import json
    import tempfile
    from pathlib import Path

    from dataknobs_config.inheritance import InheritableConfigLoader

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        (directory / "parent.json").write_text(json.dumps(base))
        (directory / "child.json").write_text(json.dumps({"extends": "parent", **override}))

        merged = InheritableConfigLoader(directory).load("child")

    # `extends` is consumed by the loader and does not survive into the result.
    return merged


# The sections `DynaBotConfigBuilder.build()` requires. The fixture is not a
# bot config, and `build()` validates, so the merge is driven through the real
# public path with these present and they are projected back off the result.
# Reading `_config` directly instead would skip the validation and the assembly
# that a caller actually goes through.
_MINIMAL_BOT: dict[str, Any] = {
    "llm": {"provider": "echo", "model": "test"},
    "conversation_storage": {"backend": "memory"},
}


def _via_bot_config_builder(
    base: dict[str, Any], override: dict[str, Any]
) -> dict[str, Any]:
    """``DynaBotConfigBuilder.merge_overrides``, via `from_config` / `build`."""
    from dataknobs_bots.config.builder import DynaBotConfigBuilder

    built = (
        DynaBotConfigBuilder.from_config({**base, **_MINIMAL_BOT})
        .merge_overrides(override)
        .build()
    )
    return {key: value for key, value in built.items() if key not in _MINIMAL_BOT}


def _via_apply_template(
    base: dict[str, Any], override: dict[str, Any]
) -> dict[str, Any]:
    """``apply_template`` with a template registered for the test."""
    from dataknobs_fsm.config.schema import TEMPLATES, UseCaseTemplate, apply_template

    template = UseCaseTemplate.DATABASE_ETL
    original = TEMPLATES[template]
    TEMPLATES[template] = base
    try:
        return apply_template(template, overrides=override)
    finally:
        TEMPLATES[template] = original


def _via_conversion_options(
    base: dict[str, Any], override: dict[str, Any]
) -> dict[str, Any]:
    """``ConversionOptions.merge_metadata``, merging two records' metadata."""
    from dataknobs_data.pandas import ConversionOptions

    return ConversionOptions().merge_metadata(base, override)


MERGERS: list[tuple[str, Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]]] = [
    ("dataknobs_config.deep_merge", _via_deep_merge),
    ("InheritableConfigLoader extends:", _via_inheritable_loader),
    ("DynaBotConfigBuilder.merge_overrides", _via_bot_config_builder),
    ("dataknobs_fsm apply_template", _via_apply_template),
    ("ConversionOptions.merge_metadata", _via_conversion_options),
]


@pytest.mark.parametrize(("name", "merge"), MERGERS, ids=[m[0] for m in MERGERS])
def test_entry_points_agree(
    name: str, merge: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
) -> None:
    """Scalars override, nested dicts merge, lists replace -- at every depth."""
    assert merge(FIXTURE_BASE, FIXTURE_OVERRIDE) == EXPECTED


def test_the_fixture_would_catch_an_extending_merge() -> None:
    """The fixture distinguishes replace from extend -- at both depths.

    Without this, a fixture whose lists happened to compare equal under either
    policy would let the whole guard above pass vacuously. Asserts the
    expectation is *not* what an extending merge produces, which is the exact
    defect this file exists to prevent from recurring.
    """
    assert EXPECTED["listy"] != FIXTURE_BASE["listy"] + FIXTURE_OVERRIDE["listy"]
    assert (
        EXPECTED["deep"]["inner"]["listy"]
        != FIXTURE_BASE["deep"]["inner"]["listy"] + FIXTURE_OVERRIDE["deep"]["inner"]["listy"]
    )


def test_merge_configs_agrees_on_a_schema_shaped_fixture() -> None:
    """``ConfigLoader.merge_configs`` -- the entry point the defect reached.

    It round-trips through ``FSMConfig`` validation and so cannot take the
    shared fixture, but it asserts the same three properties against real FSM
    fields: scalar override (``name``), nested-dict merge (``data_mode``), and
    **list replace** (``networks``) -- which is where two configurations each
    declaring one network named "main" used to produce two.

    The nested-dict property needs a key only *one* side declares to have any
    teeth. Asserting on a key both sides set proves nothing: merge and replace
    produce the same value for it, so such an assertion passes with the
    recursive branch deleted entirely -- the same blindness that let the
    original defect survive, one level down.
    """
    from dataknobs_fsm.config.loader import ConfigLoader
    from dataknobs_fsm.config.schema import (
        FSMConfig,
        NetworkConfig,
        StateConfig,
    )
    from dataknobs_fsm.core.data_modes import DataHandlingMode

    def one_network(
        name: str, state: str, data_mode: dict[str, Any]
    ) -> FSMConfig:
        return FSMConfig(
            name=name,
            networks=[
                NetworkConfig(
                    name="main",
                    states=[StateConfig(name=state, is_start=True, arcs=[])],
                ),
            ],
            main_network="main",
            data_mode=data_mode,
        )

    merged = ConfigLoader().merge_configs(
        one_network(
            "first",
            "from_first",
            {"default": DataHandlingMode.COPY, "copy_config": {"only_on_first": 1}},
        ),
        one_network("second", "from_second", {"default": DataHandlingMode.REFERENCE}),
    )

    assert merged.name == "second"
    assert merged.data_mode.default == DataHandlingMode.REFERENCE
    # Declared only by the first config. Under a whole-value replace of
    # `data_mode` this is {} -- which is what gives the assertion its teeth.
    assert merged.data_mode.copy_config == {"only_on_first": 1}
    assert len(merged.networks) == 1
    assert [s.name for s in merged.networks[0].states] == ["from_second"]
