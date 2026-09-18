# SPDX-FileCopyrightText: Copyright 2022-2026 KBS Labs
# SPDX-License-Identifier: Apache-2.0

"""Ontologies over live data: the registry, and the source it binds.

``dataknobs_common.ontology`` loads a vocabulary a person edited. This package
loads one backed by a table someone else owns -- which is a different object
because it has a lifecycle:

    from dataknobs_config import EnvironmentAwareConfig
    from dataknobs_data.ontology import OntologyRegistry

    cfg = EnvironmentAwareConfig.load_app("catalog")
    registry = await OntologyRegistry.from_config_async(
        cfg.resolve_for_build("ontology")
    )
    onto = registry.get("catalog")
    entity = await onto.entity(onto.localize("catalog:sku-4471"))
    ...
    await registry.close()

The vocabulary itself stays a value. It is the registry that opened the
handles, so it is the registry that closes them -- and only the ones it opened:
a handle handed to :meth:`~OntologyRegistry.from_components` belongs to
whoever built it.
"""

from dataknobs_data.ontology.registry import OntologyRegistry
from dataknobs_data.ontology.sources import (
    RECORD_SOURCE_KIND,
    EntityProjection,
    RecordEntitySource,
    SurfaceFormLookup,
)

__all__ = [
    "RECORD_SOURCE_KIND",
    "EntityProjection",
    "OntologyRegistry",
    "RecordEntitySource",
    "SurfaceFormLookup",
]
