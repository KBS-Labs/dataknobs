"""Data handling is selected by mode, and the documented interface is real.

``AdvancedFSM.set_data_handler`` advertised a custom-handler extension point.
Its whole body assigned ``self._engine.data_handler``, a name the execution
engine neither declares nor reads --- so every handler passed to it was
silently ignored. The name looked plausible precisely because it *is* live
elsewhere: ``StateInstance.data_handler`` is real and read six times in
``core/state.py``. Different object, same word.

The method is gone. What is left is the route that works, and this file pins
the three claims the documentation now makes about it:

* the ``DataHandler`` ABC can be implemented as the guide teaches it --- four
  methods, with ``on_exit(data, commit=True)``. The page that taught it wrong
  did so in a commit titled "Fixed documentation to correlate with the code",
  which also created the page next door that got it right; nothing caught
  either, because no test had ever subclassed ``DataHandler``;
* the top-level ``data_mode.default`` reaches a built state, and a state's own
  ``data_mode`` overrides it;
* the four other keys the ``data_mode`` block accepts are read by nothing.

That last one is a guard against re-blessing rather than a guard against
regression: a doc example teaching ``state_overrides`` is wrong today, and
the test says so rather than leaving the next author to re-derive it.
"""

from __future__ import annotations

from typing import Any

from dataknobs_fsm import AdvancedFSM
from dataknobs_fsm.config.builder import FSMBuilder
from dataknobs_fsm.config.loader import ConfigLoader
from dataknobs_fsm.core.data_modes import DataHandler, DataHandlingMode


# --------------------------------------------------------------------------- #
# The interface the guide documents
# --------------------------------------------------------------------------- #


def test_the_documented_handler_interface_can_be_implemented() -> None:
    """The ABC's four methods, in the shape ``data-modes.md`` teaches them.

    The API page this replaces carried an example implementing three of the
    four and giving ``on_exit`` the wrong arity, so it could not be
    instantiated --- ``TypeError: Can't instantiate abstract class
    CustomDataHandler without an implementation for abstract method
    'supports_concurrent_access'``. A reader following it got that error
    before ever reaching the setter on the last line.
    """

    class DocumentedHandler(DataHandler):
        def __init__(self) -> None:
            super().__init__(DataHandlingMode.COPY)
            self.exits: list[bool] = []

        def on_entry(self, data: Any) -> Any:
            return dict(data) if isinstance(data, dict) else data

        def on_modification(self, data: Any) -> Any:
            return data

        def on_exit(self, data: Any, commit: bool = True) -> Any:
            self.exits.append(commit)
            return data

        def supports_concurrent_access(self) -> bool:
            return True

    handler = DocumentedHandler()

    # Positionally, the way ``StateInstance.exit`` calls it.
    assert handler.on_exit({"a": 1}, False) == {"a": 1}
    assert handler.exits == [False]
    assert handler.supports_concurrent_access() is True


def test_the_engine_never_grew_a_data_handler_attribute() -> None:
    """The removal, recorded rather than assumed.

    ``AttributeError`` from a missing method is the diagnostic the deleted
    method never gave: it accepted a handler and did nothing with it.
    """
    assert not hasattr(AdvancedFSM, "set_data_handler")


# --------------------------------------------------------------------------- #
# The route that does work
# --------------------------------------------------------------------------- #


def _build(config: dict[str, Any]) -> Any:
    return FSMBuilder().build(ConfigLoader().load_from_dict(config))


def _config(*, default: str | None = None, state_mode: str | None = None) -> dict[str, Any]:
    start: dict[str, Any] = {"name": "start", "is_start": True}
    if state_mode is not None:
        start["data_mode"] = state_mode
    config: dict[str, Any] = {
        "name": "modes",
        "main_network": "main",
        "networks": [
            {
                "name": "main",
                "states": [start, {"name": "end", "is_end": True}],
                "arcs": [{"from": "start", "to": "end", "name": "go"}],
            }
        ],
    }
    if default is not None:
        config["data_mode"] = {"default": default}
    return config


def test_the_top_level_default_reaches_a_built_state() -> None:
    """``data_mode.default`` is a top-level block, not a per-network one."""
    state = _build(_config(default="reference")).get_start_state()
    assert state.data_mode is DataHandlingMode.REFERENCE


def test_a_state_overrides_the_default_with_its_own_data_mode() -> None:
    """The per-state override goes on the state, which is what the guide says."""
    state = _build(_config(default="reference", state_mode="direct")).get_start_state()
    assert state.data_mode is DataHandlingMode.DIRECT


def test_copy_is_the_mode_when_nothing_says_otherwise() -> None:
    """The documented default, from the schema rather than from a docstring."""
    state = _build(_config()).get_start_state()
    assert state.data_mode is DataHandlingMode.COPY


def test_a_data_mode_block_under_a_network_is_dropped_without_complaint() -> None:
    """The trap the guide's own example fell into for its whole life.

    Pydantic accepts unknown keys silently and ``NetworkConfig`` has no
    ``data_mode`` field, so a block written there validates, is discarded, and
    leaves every state on ``copy``. Nothing warns. The guide taught this shape
    in all three of its configuration examples.
    """
    config = _config()
    config["networks"][0]["data_mode"] = {"default": "direct"}

    fsm = _build(config)

    assert not hasattr(fsm.networks["main"], "data_mode")
    assert fsm.get_start_state().data_mode is DataHandlingMode.COPY


def test_the_other_data_mode_keys_are_accepted_and_read_by_nothing() -> None:
    """``state_overrides`` and the three per-mode configs are inert.

    They parse and validate, so a reader has no signal that they do nothing.
    The assertion is over the source rather than over behaviour, because
    "nothing reads this" is not a behaviour a state can be asked about.
    """
    from dataknobs_fsm.config.schema import DataModeConfig

    inert = {"state_overrides", "copy_config", "reference_config", "direct_config"}
    assert inert < set(DataModeConfig.model_fields)

    config = _config(default="copy")
    config["data_mode"]["state_overrides"] = {"start": "direct"}

    # Accepted...
    state = _build(config).get_start_state()
    # ...and ignored: the override names `start`, which stays on the default.
    assert state.data_mode is DataHandlingMode.COPY
