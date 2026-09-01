"""A configuration key the schema does not know must not be discarded in silence.

No model in :mod:`dataknobs_fsm.config.schema` declared ``extra``, so pydantic's
``ignore`` default applied to all ten. A key at the wrong nesting level, or a
misspelled one, validated and was dropped --- the FSM then ran on the defaults
the author believed they had overridden, with no exception, no log line and no
field left to inspect.

The asymmetry is what makes this a defect rather than a matter of taste: the
same schema is already strict about *values*, so a ``FunctionReference`` whose
``type`` is unknown raises immediately. One author error was loud in a value and
silent in a key.

One key is deliberately exempt. A removed ``transaction:`` block warns and
loads, because a config that was correct against an earlier version needs a
migration signal rather than a failure --- the distinction being that its key
*was* valid once, where a misspelling never was. Both halves are pinned below;
the exemption is the reason this fix is not a one-line change.

Real configs and the real loader throughout: the subject is what the loader does
with a configuration, so a stand-in for either would be testing the stand-in.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from dataknobs_fsm.config import schema as schema_module
from dataknobs_fsm.config.builder import FSMBuilder
from dataknobs_fsm.config.loader import ConfigLoader


def _config(**overrides: Any) -> dict[str, Any]:
    """A minimal FSM configuration that loads, before any override."""
    config: dict[str, Any] = {
        "name": "t",
        "main_network": "main",
        "networks": [
            {
                "name": "main",
                "states": [
                    {"name": "start", "is_start": True},
                    {"name": "finish", "is_end": True},
                ],
            }
        ],
    }
    config.update(overrides)
    return config


def _extra_keys(error: ValidationError) -> list[str]:
    return sorted(
        ".".join(str(part) for part in err["loc"])
        for err in error.errors()
        if err["type"] == "extra_forbidden"
    )


# --------------------------------------------------------------------------- #
# Keys the schema does not know
# --------------------------------------------------------------------------- #


def test_a_data_mode_block_under_a_network_is_refused() -> None:
    """The reported shape, stated as the author would hit it.

    ``data_mode`` is a top-level block, and a network is where a reader looks
    for something that governs the states inside it. Written there it was
    accepted, discarded, and every state ran ``copy`` --- so the author read
    their own configuration back as evidence of a mode that was never in
    effect.
    """
    config = _config()
    config["networks"][0]["data_mode"] = {"default": "reference"}

    with pytest.raises(ValidationError) as excinfo:
        ConfigLoader().load_from_dict(config)

    assert "networks.0.data_mode" in _extra_keys(excinfo.value)


def test_a_resource_key_outside_the_config_block_is_refused() -> None:
    """The shape that cost the most, and the one that was already in the tree.

    ``ResourceConfig`` carries its backend settings in ``config``. Spelled flat
    --- which is how a reader would guess --- the resource was built with an
    empty ``config``: a sqlite database with no path, from a configuration that
    named one.
    """
    config = _config(
        resources=[{"name": "db", "type": "database", "provider": "sqlite", "path": ":memory:"}]
    )

    with pytest.raises(ValidationError) as excinfo:
        ConfigLoader().load_from_dict(config)

    assert _extra_keys(excinfo.value) == ["resources.0.path", "resources.0.provider"]


def test_a_misspelled_state_flag_is_refused() -> None:
    """A typo is the same defect as a misplacement, and the commoner one."""
    config = _config()
    config["networks"][0]["states"][1] = {"name": "finish", "is_endd": True}

    with pytest.raises(ValidationError) as excinfo:
        ConfigLoader().load_from_dict(config)

    assert "networks.0.states.1.is_endd" in _extra_keys(excinfo.value)


def test_every_config_model_refuses_unknown_keys() -> None:
    """The guard, so that the next model added inherits the property.

    Naming the ten models here would pass while a new eleventh silently
    reverted to ``ignore``, which is the failure this whole file is about. The
    check walks the module instead.
    """
    models = [
        value
        for value in vars(schema_module).values()
        if isinstance(value, type) and issubclass(value, BaseModel) and value is not BaseModel
    ]
    assert models, "no configuration models found --- the walk is looking in the wrong place"

    permissive = [m.__name__ for m in models if m.model_config.get("extra") != "forbid"]
    assert not permissive, f"these models still discard unknown keys silently: {permissive}"


# --------------------------------------------------------------------------- #
# What must keep working
# --------------------------------------------------------------------------- #


def test_the_removed_transaction_block_still_warns_and_loads(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The one key that is exempt, and the reason the fix is not one line.

    A ``transaction:`` block configured a coordinator that has been removed. It
    warns and loads on purpose: unlike a misspelling, it *was* valid, so its
    author needs a migration signal rather than a failed load. A naive
    ``extra="forbid"`` turns that warning into an error and breaks the
    compatibility the removal deliberately provided.
    """
    config = _config(transaction={"strategy": "single"})

    with caplog.at_level(logging.WARNING, logger="dataknobs_fsm.config.schema"):
        loaded = ConfigLoader().load_from_dict(config)

    assert loaded.name == "t"
    assert any("transaction" in record.message for record in caplog.records), (
        "the removed key loaded without telling anyone it was ignored"
    )


def test_both_spellings_of_the_state_schema_block_still_validate() -> None:
    """``populate_by_name`` and the extras check must not collide.

    A state's schema block may be written ``schema`` or ``data_schema``. The
    first is an alias, which is exactly the kind of key an extras check can
    mistake for an unknown one.
    """
    for spelling in ("schema", "data_schema"):
        config = _config()
        config["networks"][0]["states"][0][spelling] = {"type": "object"}

        loaded = ConfigLoader().load_from_dict(config)

        assert loaded.networks[0].states[0].data_schema == {"type": "object"}, spelling


def test_arbitrary_data_still_travels_in_metadata() -> None:
    """Refusing unknown keys does not refuse unknown data.

    ``metadata`` is where a configuration carries what the schema has no field
    for, and it is what makes the refusal a routing decision rather than a
    restriction.
    """
    config = _config()
    config["networks"][0]["states"][0]["metadata"] = {"owner": "team", "timeout": 30}

    fsm = FSMBuilder().build(ConfigLoader().load_from_dict(config))

    assert fsm.networks["main"].states["start"].metadata["owner"] == "team"
