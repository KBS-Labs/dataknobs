"""The ``provisioning`` tool: a count read from a test run's records.

``dataknobs_common.testing.provisioning`` writes one record per suite, naming
the tests that took a fresh Elasticsearch index and never wrote to it. The
contract holds that count per package under a ceiling, like any finding. What
is different, and what these tests pin, follows from the count existing only
once the tests have run:

* a bare ``check`` does not measure it (the gate calls that before any test);
* a package the run did not test is unread, reported as ``None``, and its
  ceiling is never lowered on that run's evidence;
* every way the count could fall without a test changing is refused: a missing
  record, an unreadable one, and tests a down service skipped.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests._workspace import ROOT, load_bin_module

CONTRACT = ROOT / ".dataknobs" / "quality-contract.json"
TOOL = ROOT / "bin" / "quality-contract.py"

contract_module = load_bin_module("quality-contract")


def _contract() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(CONTRACT.read_text(encoding="utf-8"))
    return loaded


def _record(unwritten: dict[str, int], skipped: int = 0) -> dict[str, Any]:
    return {
        "version": 1,
        "files": {
            name: {"took": count, "unwritten": [f"{name}::test_{n}" for n in range(count)]}
            for name, count in unwritten.items()
        },
        "service_skipped": {"elasticsearch": skipped},
    }


def _write(records: Path, stage: str, package: str, record: dict[str, Any]) -> None:
    records.mkdir(parents=True, exist_ok=True)
    (records / f"provisioning-{stage}-{package}.json").write_text(json.dumps(record))


@pytest.fixture
def records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """An empty records directory the measurer is pointed at, testing nothing yet."""
    path = tmp_path / "records"
    path.mkdir()
    monkeypatch.setenv(contract_module.PROVISIONING_RECORDS_ENV, str(path))
    monkeypatch.setenv(contract_module.PROVISIONING_TESTED_ENV, "")
    yield path


def _tested(monkeypatch: pytest.MonkeyPatch, *packages: str) -> None:
    monkeypatch.setenv(contract_module.PROVISIONING_TESTED_ENV, " ".join(packages))


DATA_UNIT = "packages/data/tests/test_x.py"
DATA_INTEGRATION = "packages/data/tests/integration/test_y.py"


def test_a_package_s_stages_are_summed_per_file(
    records: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(records, "unit", "data", _record({DATA_UNIT: 2}))
    _write(records, "integration", "data", _record({DATA_INTEGRATION: 3, DATA_UNIT: 0}))
    _tested(monkeypatch, "data")

    measured = contract_module.measure_provisioning(_contract(), [])

    assert dict(measured.by_cell["packages/data/tests"]) == {DATA_UNIT: 2, DATA_INTEGRATION: 3}
    assert "packages/data/tests" not in measured.unread


def test_a_package_the_run_did_not_test_is_unread(
    records: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(records, "unit", "common", _record({}))
    _tested(monkeypatch, "common")

    measured = contract_module.measure_provisioning(_contract(), [])

    assert "packages/data/tests" in measured.unread
    assert "packages/common/tests" not in measured.unread
    # Cells no session writes a record for are never read either.
    assert "bin" in measured.unread


def test_check_reports_an_unread_cell_as_none_and_never_as_cleared(
    records: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The data cell's ceiling is above zero; read as a zero it would report as cleared."""
    _write(records, "unit", "common", _record({}))
    _tested(monkeypatch, "common")

    report = contract_module.check(_contract(), ["provisioning"])

    assert report["cells"]["provisioning/packages/data/tests"]["measured"] is None
    assert "provisioning/packages/data/tests" in {e["cell"] for e in report["unmeasured"]}
    assert not [e for e in report["cleared"] if e["path"] == "packages/data/tests"]
    assert report["cells"]["provisioning/packages/common/tests"]["measured"] == 0


def test_a_baseline_update_never_lowers_an_unread_cell(
    records: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    contract = _contract()
    cells = {cell["path"]: cell for cell in contract["tools"]["provisioning"]["cells"]}
    cells["packages/common/tests"]["ceiling"] = 4
    data_ceiling = cells["packages/data/tests"]["ceiling"]
    assert data_ceiling > 0, "the test needs an unread cell whose ceiling a zero would lower"
    _write(records, "unit", "common", _record({"packages/common/tests/test_z.py": 1}))
    _tested(monkeypatch, "common")
    written = tmp_path / "contract.json"

    lowered, _exceeded = contract_module.update_baseline(contract, ["provisioning"], written)

    assert lowered == ["provisioning/packages/common/tests: 4 -> 1"]
    after = json.loads(written.read_text())["tools"]["provisioning"]["cells"]
    assert {c["path"]: c["ceiling"] for c in after}["packages/data/tests"] == data_ceiling


def test_a_tested_package_without_a_record_for_a_stage_it_has_is_refused(
    records: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The data package has an integration directory, so a unit record alone is half a count."""
    _write(records, "unit", "data", _record({DATA_UNIT: 1}))
    _tested(monkeypatch, "data")

    with pytest.raises(SystemExit, match=r"provisioning-integration-data\.json"):
        contract_module.measure_provisioning(_contract(), [])


def test_a_package_without_an_integration_directory_needs_only_its_unit_record(
    records: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert not (ROOT / "packages" / "common" / "tests" / "integration").exists()
    _write(records, "unit", "common", _record({}))
    _tested(monkeypatch, "common")

    contract_module.measure_provisioning(_contract(), [])


def test_tests_a_down_service_skipped_refuse_the_count(
    records: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(records, "unit", "common", _record({}, skipped=3))
    _tested(monkeypatch, "common")

    with pytest.raises(SystemExit, match=r"3 test.*elasticsearch unreachable"):
        contract_module.measure_provisioning(_contract(), [])


@pytest.mark.parametrize(
    "content",
    ["not json", json.dumps({"version": 2, "files": {}, "service_skipped": {}}), "[]"],
    ids=["unparseable", "another-version", "not-an-object"],
)
def test_a_record_of_another_shape_is_refused(
    records: Path, monkeypatch: pytest.MonkeyPatch, content: str
) -> None:
    (records / "provisioning-unit-common.json").write_text(content)
    _tested(monkeypatch, "common")

    with pytest.raises(SystemExit, match=r"provisioning-unit-common\.json"):
        contract_module.measure_provisioning(_contract(), [])


def test_measuring_with_no_records_named_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(contract_module.PROVISIONING_RECORDS_ENV, raising=False)
    monkeypatch.delenv(contract_module.PROVISIONING_TESTED_ENV, raising=False)

    with pytest.raises(SystemExit, match="--records"):
        contract_module.measure_provisioning(_contract(), [])


def test_a_bare_check_does_not_measure_it_and_a_named_one_does() -> None:
    assert "provisioning" not in contract_module._selected(None)
    assert set(contract_module._selected(None)) == {"ruff", "mypy", "format"}
    assert contract_module._selected("provisioning") == ["provisioning"]


def test_a_charge_refuses_a_path_whose_package_was_not_tested(
    records: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(records, "unit", "common", _record({}))
    _tested(monkeypatch, "common")

    with pytest.raises(SystemExit, match="not read"):
        contract_module.charge(_contract(), "provisioning", ["packages/data/tests"])


def _cli(*args: str) -> subprocess.CompletedProcess[str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key
        not in (contract_module.PROVISIONING_RECORDS_ENV, contract_module.PROVISIONING_TESTED_ENV)
    }
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env={**environment, "NO_COLOR": "1"},
    )


def test_the_command_line_reads_the_records_it_is_handed(tmp_path: Path) -> None:
    """End to end: the flags reach the measurer, and the verdict is the exit status."""
    records = tmp_path / "records"
    ceiling = {
        cell["path"]: cell["ceiling"] for cell in _contract()["tools"]["provisioning"]["cells"]
    }["packages/utils/tests"]
    _write(records, "unit", "utils", _record({}))
    _write(
        records,
        "integration",
        "utils",
        _record({"packages/utils/tests/integration/test_u.py": ceiling}),
    )

    at = _cli("check", "--tool", "provisioning", "--records", str(records), "--tested", "utils")
    assert at.returncode == 0, at.stdout + at.stderr

    _write(records, "unit", "utils", _record({"packages/utils/tests/test_new.py": 1}))
    over = _cli("check", "--tool", "provisioning", "--records", str(records), "--tested", "utils")
    assert over.returncode != 0
    assert "packages/utils/tests/test_new.py" in over.stdout + over.stderr
