"""Bots' provenance module re-exports the records from dataknobs-common.

The records' behaviour is tested where they live,
``packages/common/tests/test_provenance.py``. What is pinned here is that bots'
import path names the same objects, so an artifact's provenance is the common
record and not a copy of it.
"""

from __future__ import annotations

import dataknobs_common.provenance as common

from dataknobs_bots import artifacts
from dataknobs_bots.artifacts import provenance

NAMES = (
    "LLMInvocation",
    "ProvenanceRecord",
    "RevisionRecord",
    "SourceReference",
    "ToolInvocation",
    "create_provenance",
    "now_iso",
)


def test_bots_provenance_re_exports_the_common_records() -> None:
    assert set(provenance.__all__) == set(NAMES)
    for name in NAMES:
        assert getattr(provenance, name) is getattr(common, name), name


def test_the_artifacts_package_exports_the_same_objects() -> None:
    # Every name but ``now_iso``, which the package has never exported.
    for name in NAMES:
        if name != "now_iso":
            assert getattr(artifacts, name) is getattr(common, name), name


def test_an_artifact_carries_the_common_record() -> None:
    from dataknobs_bots.artifacts.models import Artifact

    artifact = Artifact(provenance=provenance.create_provenance("bot:test", "generator"))
    assert isinstance(artifact.provenance, common.ProvenanceRecord)
    restored = Artifact.from_dict(artifact.to_dict())
    assert restored.provenance == artifact.provenance
