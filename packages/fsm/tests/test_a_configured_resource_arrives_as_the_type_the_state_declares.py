# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""A state's declared resources arrive as the type the field declares.

``StateDefinition.resource_requirements`` is declared
``List[ResourceConfig]`` --- :class:`dataknobs_fsm.functions.base.ResourceConfig`,
the runtime type, which is what ``core`` imports and what ``IResource.initialize``
takes. The config builder put :class:`dataknobs_fsm.config.schema.ResourceConfig`
there instead: a second class of the same name, declaring the same concept under
different field names.

The two arrived in different commits, and neither commit mentions the other
class. The runtime one is as old as ``functions/base.py`` itself (``595f1615``);
the schema one and the builder line that puts it into the core field are the
*same* commit (``42119f99``), so the second class and the boundary violation
were written together and argued nowhere. No commit has ever attempted a
conversion --- ``git log -S'connection_params=' -- builder.py`` returns
nothing.

So the field promises ``connection_params``, ``pool_size``, ``timeout`` and
``retry_policy``, and on any FSM built from configuration it carries an object
with none of them. Everything downstream that needed one of those fields grew a
hedge instead of a fix, and this package now has four:

- two in ``AsyncExecutionEngine``, reading
  ``getattr(rc, "timeout_seconds", None) or getattr(rc, "timeout", None) or 30``
- one in ``StateNetwork._update_resource_requirements``, whose comment names
  "the two-``ResourceConfig`` mismatch this package tracks separately" and works
  around it with ``kind.value if isinstance(kind, Enum) else str(kind)``
- one in ``ResourceManager.configure_from_requirements``, which does *not* hedge
  --- it reads ``config.timeout`` directly and would raise ``AttributeError`` on
  every state the builder produces. It has no caller, which is the only reason
  that has never been seen.

These tests read the field the way its declaration says to.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from dataknobs_fsm.config.builder import FSMBuilder
from dataknobs_fsm.config.loader import ConfigLoader
from dataknobs_fsm.core.fsm import FSM
from dataknobs_fsm.functions.base import ResourceConfig
from dataknobs_fsm.resources.manager import ResourceManager


def configured_fsm() -> FSM:
    """An FSM whose start state declares one fully-specified resource."""
    config: dict[str, Any] = {
        "name": "resourced",
        "main_network": "main",
        "resources": [
            {
                "name": "db",
                "type": "database",
                "config": {"backend": "memory"},
                "connection_pool_size": 7,
                "timeout_seconds": 12,
                "retry_attempts": 5,
                "retry_delay_seconds": 2.5,
                "health_check_interval": 60,
            },
        ],
        "networks": [
            {
                "name": "main",
                "states": [
                    {"name": "start", "is_start": True, "resources": ["db"]},
                    {"name": "end", "is_end": True},
                ],
                "arcs": [{"from": "start", "to": "end"}],
            }
        ],
    }
    return FSMBuilder().build(ConfigLoader().load_from_dict(config))


def only_requirement(fsm: FSM) -> Any:
    """The one resource the start state declares."""
    requirements = fsm.networks["main"].states["start"].resource_requirements
    assert len(requirements) == 1, requirements
    return requirements[0]


def test_the_field_holds_the_type_it_declares() -> None:
    """The whole point: a reader following the annotation is not lied to."""
    rc = only_requirement(configured_fsm())

    assert isinstance(rc, ResourceConfig), (
        f"resource_requirements is declared List[ResourceConfig] "
        f"({ResourceConfig.__module__}.ResourceConfig) and holds "
        f"{type(rc).__module__}.{type(rc).__name__}"
    )


def test_every_configured_field_survives_the_translation() -> None:
    """Not merely present --- carrying what the configuration said.

    A conversion that answered defaults for everything would satisfy the
    isinstance check above and lose the configuration, so each field is
    asserted against a value that is not its default.
    """
    rc = only_requirement(configured_fsm())

    assert rc.name == "db"
    assert rc.type == "database", "the schema's ResourceType becomes its value"
    assert rc.connection_params == {"backend": "memory"}, "schema `config` is the runtime's params"
    assert rc.pool_size == 7, "schema `connection_pool_size`"
    assert rc.timeout == 12.0, "schema `timeout_seconds`"
    assert rc.health_check_interval == 60.0
    assert rc.retry_policy == {"retry_attempts": 5, "retry_delay_seconds": 2.5}, (
        "the two retry fields travel together, under the names the schema gives them"
    )


def test_the_resource_manager_can_read_a_configured_states_requirements() -> None:
    """``configure_from_requirements`` reads ``config.timeout`` and never hedged.

    It is public API on a public class and has no in-tree caller, so this is the
    first thing to call it with what the builder actually produces.
    """
    fsm = configured_fsm()
    requirements = fsm.networks["main"].states["start"].resource_requirements

    manager = ResourceManager()
    manager.register_provider("db", manager.create_simple_provider("db", {"rows": []}))
    try:
        acquired = manager.configure_from_requirements(requirements, owner_id="probe")
        assert set(acquired) == {"db"}
    finally:
        manager.close()


def test_a_state_naming_an_undeclared_resource_is_refused_by_the_schema() -> None:
    """Which is why the builder's fallback arm cannot be reached, or tested.

    ``_build_state`` answers an unknown resource name with a freshly built
    ``ResourceConfig(name=r, type=CUSTOM)`` rather than failing. That arm is
    unreachable: ``FSMConfig``'s model validator refuses any state naming a
    resource the document does not declare, and it runs on every construction.
    ``FSMBuilder._validate_completeness`` carries a second copy of the same
    check, equally unreachable. Both are recorded rather than removed here ---
    deleting them changes what the builder promises a caller who bypasses
    validation, which is a different decision from this one.

    This test pins the reachable behaviour, so that if the validator is ever
    relaxed the arm below it starts being exercised rather than starting to
    matter silently.
    """
    config: dict[str, Any] = {
        "name": "unresourced",
        "main_network": "main",
        "networks": [
            {
                "name": "main",
                "states": [
                    {"name": "start", "is_start": True, "resources": ["nowhere"]},
                    {"name": "end", "is_end": True},
                ],
                "arcs": [{"from": "start", "to": "end"}],
            }
        ],
    }

    with pytest.raises(ValidationError, match="Resource 'nowhere' not found"):
        ConfigLoader().load_from_dict(config)


def test_the_network_buckets_a_configured_resource_by_its_type() -> None:
    """The bucketing reads ``.type`` and must keep working on a plain ``str``.

    ``_update_resource_requirements`` currently carries
    ``kind.value if isinstance(kind, Enum) else str(kind)`` for exactly the
    ambiguity this change removes; the behaviour it protects is asserted here
    so removing the hedge cannot quietly change it.
    """
    totals = configured_fsm().networks["main"].get_resource_requirements()

    assert totals.databases == {"db"}
    assert totals.custom == {}


@pytest.mark.parametrize("field", ["connection_params", "pool_size", "timeout", "retry_policy"])
def test_each_declared_field_is_reachable(field: str) -> None:
    """One failure per field, so a partial conversion reports which half it did."""
    rc = only_requirement(configured_fsm())

    assert hasattr(rc, field), f"the declared type has {field} and this object does not"
