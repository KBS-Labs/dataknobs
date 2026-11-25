"""Bots' provenance module re-exports the records from dataknobs-common.

The records' behaviour is tested where they live,
``packages/common/tests/test_provenance.py``. What is pinned here is that bots'
import path names the same objects, so an artifact's provenance is the common
record and not a copy of it, and its ``now_iso`` is the common clock's.
"""

from __future__ import annotations

from dataknobs_common import clock
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
)


def test_bots_provenance_re_exports_the_common_records() -> None:
    assert set(provenance.__all__) == {*NAMES, "now_iso"}
    for name in NAMES:
        assert getattr(provenance, name) is getattr(common, name), name


def test_bots_now_iso_is_the_common_clock() -> None:
    assert provenance.now_iso is clock.now_iso


def test_the_artifacts_package_exports_the_same_objects() -> None:
    # ``now_iso`` is not among them: the package has never exported it.
    for name in NAMES:
        assert getattr(artifacts, name) is getattr(common, name), name


def test_an_artifact_carries_the_common_record() -> None:
    from dataknobs_bots.artifacts.models import Artifact

    artifact = Artifact(provenance=provenance.create_provenance("bot:test", "generator"))
    assert isinstance(artifact.provenance, common.ProvenanceRecord)
    restored = Artifact.from_dict(artifact.to_dict())
    assert restored.provenance == artifact.provenance
