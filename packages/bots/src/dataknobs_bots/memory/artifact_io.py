"""File import/export for ArtifactBank artifacts.

Provides functions to save artifacts as JSON files and load them back,
as well as JSONL "book" files containing multiple artifacts.

Two export formats are supported:

- **Compiled** (default): Clean, shareable output from ``compile()``.
  Top-level keys include ``_artifact_name`` and ``_compiled_at``.
- **Full-state**: Complete internal state from ``to_dict()``, including
  provenance, finalization status, and section configs.  Suitable for
  exact round-trip restore.

Example::

    >>> from dataknobs_bots.memory.artifact_io import save_artifact, load_artifact
    >>> save_artifact(artifact, "recipe.json")
    >>> restored = load_artifact("recipe.json", artifact_config=RECIPE_CONFIG)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from .artifact_bank import ArtifactBank

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

_COMPILED_MARKER = "_artifact_name"
_FULL_STATE_KEYS = {"name", "sections"}


def _detect_format(data: dict[str, Any]) -> str:
    """Detect whether *data* is compiled or full-state format.

    Returns:
        ``"compiled"`` or ``"full_state"``.

    Raises:
        ValueError: If the format cannot be determined.
    """
    if _COMPILED_MARKER in data:
        return "compiled"
    if _FULL_STATE_KEYS.issubset(data):
        return "full_state"
    raise ValueError(
        "Cannot detect artifact format: data has neither "
        f"'{_COMPILED_MARKER}' (compiled) nor {_FULL_STATE_KEYS} (full-state) keys."
    )


def _artifact_name_from_data(data: dict[str, Any]) -> str:
    """Extract the artifact name from either format."""
    if _COMPILED_MARKER in data:
        return str(data[_COMPILED_MARKER])
    return str(data.get("name", ""))


# ---------------------------------------------------------------------------
# Compiled-format reverse mapping
# ---------------------------------------------------------------------------


def _infer_artifact_config(data: dict[str, Any]) -> dict[str, Any]:
    """Heuristically build an artifact config from compiled data.

    Keys prefixed with ``_`` are treated as metadata and skipped.
    Values that are ``list[dict]`` are classified as sections;
    everything else becomes a field (``required: False``).

    This is best-effort — a ``list[dict]`` field would be
    misclassified as a section.
    """
    fields: dict[str, dict[str, Any]] = {}
    sections: dict[str, dict[str, Any]] = {}
    name = str(data.get(_COMPILED_MARKER, "artifact"))

    for key, value in data.items():
        if key.startswith("_"):
            continue
        if isinstance(value, list) and all(isinstance(v, dict) for v in value):
            sections[key] = {"schema": {}}
        else:
            fields[key] = {"required": False}

    return {"name": name, "fields": fields, "sections": sections}


def _load_from_compiled(
    data: dict[str, Any],
    config: dict[str, Any] | None,
) -> ArtifactBank:
    """Build an ArtifactBank from compiled-format data.

    If *config* is provided, ``from_config`` creates a properly-typed
    artifact.  Otherwise the config is inferred heuristically.
    """
    effective_config = config if config is not None else _infer_artifact_config(data)
    artifact = ArtifactBank.from_config(effective_config)
    _populate_from_compiled(artifact, data)
    return artifact


def _populate_from_compiled(
    artifact: ArtifactBank,
    data: dict[str, Any],
) -> None:
    """Set fields and add section records from compiled data."""
    # Set field values
    for field_name in artifact.field_defs:
        if field_name in data:
            artifact.set_field(field_name, data[field_name])

    # Add section records
    for section_name, bank in artifact.sections.items():
        records = data.get(section_name, [])
        if isinstance(records, list):
            for record_data in records:
                if isinstance(record_data, dict):
                    bank.add(record_data, source_stage="import")


# ---------------------------------------------------------------------------
# Atomic write helper
# ---------------------------------------------------------------------------


def _atomic_write_text(path: Path, content: str) -> None:
    """Write *content* to *path* atomically via temp-file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, path)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Parse a JSONL file into a list of dicts.

    Blank lines are silently skipped.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If any non-blank line is not valid JSON.
    """
    entries: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_no} of {path}: {exc}"
                ) from exc
            if not isinstance(obj, dict):
                raise ValueError(
                    f"Expected JSON object on line {line_no} of {path}, "
                    f"got {type(obj).__name__}"
                )
            entries.append(obj)
    return entries


# ---------------------------------------------------------------------------
# Public API — Export
# ---------------------------------------------------------------------------


def save_artifact(
    artifact: ArtifactBank,
    path: str | Path,
    *,
    compiled: bool = True,
) -> None:
    """Write a single artifact to a JSON file.

    Args:
        artifact: The artifact to save.
        path: Destination file path.
        compiled: If ``True`` (default), use ``compile()`` output (clean,
            shareable).  If ``False``, use ``to_dict()`` (full internal
            state with provenance).
    """
    path = Path(path)
    data = artifact.compile() if compiled else artifact.to_dict()
    content = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    _atomic_write_text(path, content)
    logger.info(
        "Saved artifact '%s' to %s (format=%s)",
        artifact.name,
        path,
        "compiled" if compiled else "full_state",
    )


def append_to_book(
    artifact: ArtifactBank,
    path: str | Path,
    *,
    compiled: bool = True,
) -> None:
    """Append a single artifact as one JSONL line to a book file.

    Creates the file (and parent directories) if it does not exist.

    Args:
        artifact: The artifact to append.
        path: JSONL book file path.
        compiled: If ``True`` (default), use ``compile()`` output.
    """
    path = Path(path)
    data = artifact.compile() if compiled else artifact.to_dict()
    line = json.dumps(data, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
    logger.info("Appended artifact '%s' to book %s", artifact.name, path)


def save_book(
    artifacts: list[ArtifactBank],
    path: str | Path,
    *,
    compiled: bool = True,
) -> None:
    """Write multiple artifacts to a JSONL book file (atomic overwrite).

    Args:
        artifacts: List of artifacts to save.
        path: Destination JSONL file path.
        compiled: If ``True`` (default), use ``compile()`` output.
    """
    path = Path(path)
    lines: list[str] = []
    for artifact in artifacts:
        data = artifact.compile() if compiled else artifact.to_dict()
        lines.append(json.dumps(data, ensure_ascii=False))
    content = "\n".join(lines) + "\n" if lines else ""
    _atomic_write_text(path, content)
    logger.info("Saved %d artifacts to book %s", len(artifacts), path)


# ---------------------------------------------------------------------------
# Public API — Import
# ---------------------------------------------------------------------------


def load_artifact(
    path: str | Path,
    *,
    artifact_config: dict[str, Any] | None = None,
) -> ArtifactBank:
    """Load a single artifact from a JSON file.

    Auto-detects compiled vs full-state format.

    Args:
        path: Source JSON file path.
        artifact_config: Artifact configuration dict (``name``, ``fields``,
            ``sections``).  Required for precise compiled-format import;
            if omitted, the structure is inferred heuristically.

    Returns:
        Restored ``ArtifactBank`` instance.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file contains invalid JSON or unrecognised format.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected JSON object in {path}, got {type(data).__name__}"
        )

    fmt = _detect_format(data)
    if fmt == "full_state":
        artifact = ArtifactBank.from_dict(data)
    else:
        artifact = _load_from_compiled(data, artifact_config)

    logger.info("Loaded artifact '%s' from %s (format=%s)", artifact.name, path, fmt)
    return artifact


def load_from_book(
    path: str | Path,
    *,
    name: str | None = None,
    index: int | None = None,
    artifact_config: dict[str, Any] | None = None,
) -> ArtifactBank:
    """Load one artifact from a JSONL book file.

    Exactly one of *name* or *index* must be provided.

    Args:
        path: JSONL book file path.
        name: Artifact name to match.
        index: Zero-based line index.
        artifact_config: Artifact configuration for compiled-format entries.

    Returns:
        Restored ``ArtifactBank`` instance.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If selection criteria are invalid or no match is found.
    """
    if name is not None and index is not None:
        raise ValueError("Provide either 'name' or 'index', not both.")
    if name is None and index is None:
        raise ValueError("Provide either 'name' or 'index'.")

    path = Path(path)
    entries = _read_jsonl(path)

    if index is not None:
        if index < 0 or index >= len(entries):
            raise ValueError(
                f"Index {index} out of range for book with {len(entries)} entries."
            )
        data = entries[index]
    else:
        # name is not None
        assert name is not None  # for type narrowing
        matched: dict[str, Any] | None = None
        for entry in entries:
            if _artifact_name_from_data(entry) == name:
                matched = entry
                break
        if matched is None:
            available = [_artifact_name_from_data(e) for e in entries]
            raise ValueError(
                f"No artifact named '{name}' in {path}. "
                f"Available: {available}"
            )
        data = matched

    fmt = _detect_format(data)
    if fmt == "full_state":
        artifact = ArtifactBank.from_dict(data)
    else:
        artifact = _load_from_compiled(data, artifact_config)

    logger.info("Loaded artifact '%s' from book %s", artifact.name, path)
    return artifact


def list_book(path: str | Path) -> list[dict[str, Any]]:
    """List artifacts in a JSONL book file.

    Args:
        path: JSONL book file path.

    Returns:
        List of dicts with ``index``, ``name``, and ``format`` keys.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path)
    entries = _read_jsonl(path)
    result: list[dict[str, Any]] = []
    for idx, entry in enumerate(entries):
        fmt = _detect_format(entry)
        result.append({
            "index": idx,
            "name": _artifact_name_from_data(entry),
            "format": fmt,
        })
    return result
