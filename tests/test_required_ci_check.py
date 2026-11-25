"""The one status check `main` requires must not pass when a job it stands for did not run.

`All Checks Complete` runs whatever happened before it, so its own script is all
that decides. These tests run that script, taken from the workflow, with each
combination of results GitHub can hand it.
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Any

import pytest
import yaml

from tests._workspace import ROOT

WORKFLOW = ROOT / ".github" / "workflows" / "quality-validation.yml"
REQUIRED = "all-checks-complete"
DETECTION = "check-changes"

#: The jobs that run only when change detection says so, and the output they read.
GATED = {"validate-quality-artifacts": "code-changed", "build-docs": "docs-changed"}

EXPRESSION = re.compile(r"\$\{\{\s*needs\.([\w-]+)\.(result|outputs\.[\w-]+)\s*\}\}")


def _jobs() -> dict[str, Any]:
    jobs: dict[str, Any] = yaml.safe_load(WORKFLOW.read_text())["jobs"]
    return jobs


def _outcome(detection: str, code: str, docs: str, validation: str, build: str) -> int:
    """The exit code of the required check's script, given what its jobs reported."""
    values = {
        (DETECTION, "result"): detection,
        (DETECTION, "outputs.code-changed"): code,
        (DETECTION, "outputs.docs-changed"): docs,
        ("validate-quality-artifacts", "result"): validation,
        ("build-docs", "result"): build,
    }

    def fill(text: str) -> str:
        filled = EXPRESSION.sub(lambda m: values[m[1], m[2]], text)
        assert "${{" not in filled, filled
        return filled

    (step,) = _jobs()[REQUIRED]["steps"]
    env = {name: fill(value) for name, value in step.get("env", {}).items()}
    return subprocess.run(
        ["bash", "-e", "-c", fill(step["run"])],
        env={**os.environ, **env},
        capture_output=True,
        check=False,
    ).returncode


def test_the_required_check_waits_for_every_other_job() -> None:
    jobs = _jobs()
    assert set(jobs[REQUIRED]["needs"]) == set(jobs) - {REQUIRED} == {DETECTION, *GATED}
    assert jobs[REQUIRED]["if"] == "always()"
    for job, output in GATED.items():
        assert jobs[job]["if"] == f"needs.{DETECTION}.outputs.{output} == 'true'"


@pytest.mark.parametrize("code", ["true", "false"])
@pytest.mark.parametrize("docs", ["true", "false"])
def test_it_passes_when_each_job_did_what_was_decided(code: str, docs: str) -> None:
    ran = {"true": "success", "false": "skipped"}
    assert _outcome("success", code, docs, ran[code], ran[docs]) == 0


@pytest.mark.parametrize("detection", ["failure", "cancelled", "skipped"])
def test_it_fails_when_change_detection_did_not_succeed(detection: str) -> None:
    """The other jobs are skipped then, and the outputs are empty."""
    assert _outcome(detection, "", "", "skipped", "skipped") != 0


@pytest.mark.parametrize("detection", ["failure", "cancelled"])
def test_it_fails_on_detection_alone(detection: str) -> None:
    """A step can set its outputs before a later one fails the job."""
    assert _outcome(detection, "false", "false", "skipped", "skipped") != 0


@pytest.mark.parametrize(
    ("changed", "result"),
    [
        (changed, result)
        for changed in ("true", "false", "")
        for result in ("success", "failure", "cancelled", "skipped")
        if (changed, result) not in {("true", "success"), ("false", "skipped")}
    ],
)
def test_it_fails_when_a_job_did_not_do_what_was_decided(changed: str, result: str) -> None:
    assert _outcome("success", changed, "false", result, "skipped") != 0
    assert _outcome("success", "false", changed, "skipped", result) != 0
