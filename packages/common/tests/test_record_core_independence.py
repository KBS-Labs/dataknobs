"""The record vocabulary must not reach back into ``dataknobs-data``.

``Field``, ``FieldType`` and ``Record`` live here, and ``VectorField`` does
not: it needs ``numpy`` at runtime, which ``dataknobs-common`` does not
declare. That split only holds while nothing in these two modules names the
subclass — and both sites that did (a ``__name__`` comparison with a runtime
import beneath it, and a hardcoded ``from_dict`` branch) were invisible to
every import-level check, because neither appears in an import statement this
package can see.

So the guard runs in a fresh interpreter and asks what actually got imported.
"""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
import tomllib
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]

#: The one entry the base install may declare, and the only one it ever has.
#:
#: PEP 696 type-parameter defaults are ``typing``'s only from 3.13 while
#: ``requires-python`` is >=3.12, and ``dataknobs_common.hierarchy`` needs one
#: at *runtime* -- ``typing.TypeVar`` raises on ``default=``. So the marker
#: scopes it to the versions that cannot do without it, a 3.13 install already
#: resolves to nothing, and the entry deletes itself when the floor rises.
#:
#: It does not weaken what this file is guarding. The constraint is that
#: installing this package pulls in no third-party *code*: typing-extensions is
#: pure typing support, declares no dependencies of its own, and reaches
#: nothing at runtime. The three tests below are where the real teeth are --
#: they ask a fresh interpreter what actually got imported.
TYPING_BACKPORT = "typing-extensions>=4.4.0; python_full_version < '3.13'"


def _run(script: str) -> subprocess.CompletedProcess[str]:
    """Run a script in a fresh interpreter, returning the completed process.

    Not ``check=True``: the assertion carries stdout and stderr, which a
    ``CalledProcessError`` would replace with an exit status.
    """
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_the_base_install_declares_nothing_but_the_typing_backport() -> None:
    """The constraint the rest of this file is enforcing, read from the source.

    Both spellings pass, and that is deliberate: the empty list is where this
    package was and where it returns the day ``requires-python`` reaches 3.13.
    A *second* entry fails, which is the ratchet -- the guard still refuses the
    thing it was written to refuse, and this is not a door left open.
    """
    manifest = tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text())

    assert manifest["project"]["dependencies"] in ([], [TYPING_BACKPORT])


def test_the_typing_backport_pulls_in_nothing_of_its_own() -> None:
    """Declaring it must not put third-party *code* on a consumer's path.

    The letter of the old assertion was ``dependencies == []``; its purpose was
    that installing this package installs nothing else. That purpose is what
    gets asserted here, so relaxing the letter costs nothing: typing-extensions
    declares no dependencies at all, so the base install grows by one pure
    typing module and by nothing behind it.

    Read from the installed metadata rather than recalled, and asked in the
    direction that could be inconvenient -- a release that grew a dependency
    would fail this rather than being taken on trust.
    """
    requires = importlib.metadata.metadata("typing_extensions").get_all("Requires-Dist")

    assert not requires, requires


def test_importing_records_does_not_import_dataknobs_data() -> None:
    """The whole point of the split, asked of a real interpreter."""
    result = _run(
        "import sys\n"
        "import dataknobs_common.records\n"
        "import dataknobs_common.fields\n"
        "leaked = sorted(m for m in sys.modules if m.startswith('dataknobs_data'))\n"
        "assert not leaked, leaked\n"
    )

    assert result.returncode == 0, (
        f"dataknobs_data leaked into the dataknobs_common import path:\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )


def test_copying_a_field_does_not_import_dataknobs_data() -> None:
    """``Field.copy`` used to name its own class; a subclass override would too."""
    result = _run(
        "import sys\n"
        "from dataknobs_common import Field, Record\n"
        "record = Record({'a': 1, 'b': 'two'})\n"
        "duplicate = record.copy(deep=True)\n"
        "assert duplicate.get_value('a') == 1\n"
        "assert type(duplicate.fields['b']) is Field\n"
        "leaked = sorted(m for m in sys.modules if m.startswith('dataknobs_data'))\n"
        "assert not leaked, leaked\n"
    )

    assert result.returncode == 0, (
        f"Record.copy reached dataknobs_data:\nstdout={result.stdout}\nstderr={result.stderr}"
    )


def test_a_vector_payload_builds_a_plain_field_when_nothing_registered_one() -> None:
    """The one behaviour difference the split introduces, asserted not discovered.

    ``Field.from_dict`` dispatches through ``field_type_backends``, and nothing
    registers ``vector`` until ``dataknobs_data.fields`` is imported. Answering
    with a plain ``Field`` is the honest result: this package cannot construct a
    ``VectorField``. The other half of the pair — the same call returning a
    ``VectorField`` once ``dataknobs_data`` is imported — is asserted in that
    package's suite.
    """
    result = _run(
        "import sys\n"
        "from dataknobs_common import Field, FieldType, field_type_backends\n"
        "assert field_type_backends.get_optional(FieldType.VECTOR.value) is None\n"
        "built = Field.from_dict(\n"
        "    {'name': 'e', 'value': [0.1, 0.2], 'type': 'vector', 'metadata': {}}\n"
        ")\n"
        "assert type(built) is Field, type(built)\n"
        "assert built.type is FieldType.VECTOR\n"
        "assert built.value == [0.1, 0.2]\n"
        "leaked = sorted(m for m in sys.modules if m.startswith('dataknobs_data'))\n"
        "assert not leaked, leaked\n"
    )

    assert result.returncode == 0, (
        f"the unregistered-type path is not what it claims:\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
