"""The S3 atomic-insert conflict classifier recognizes both fail-closed shapes.

A conditional PUT (``IfNoneMatch="*"``) that loses to an existing object fails
closed, and AWS S3 surfaces the loss two ways:

- ``412 PreconditionFailed`` — the object already existed at request time.
- ``409 ConditionalRequestConflict`` — concurrent conditional writes to the
  same key raced, and this one lost.

Both mean the id is already taken, so ``create()`` on the sync and async S3
backends must map both to ``DuplicateRecordError``. An earlier version mapped
only 412, so a racing loser under real concurrency would have surfaced a raw
``ClientError`` instead. ``is_s3_conditional_conflict`` closes that gap and is
the single classifier both backends share (mirroring the SQL backends'
``is_duplicate_key_error``).

Exercised against **real** ``botocore.exceptions.ClientError`` instances (no
mocks): each error carries the exact ``response`` shape boto populates, pinning
the classifier against the real error surface.
"""

from __future__ import annotations

from botocore.exceptions import ClientError

from dataknobs_data.pooling.s3 import is_s3_conditional_conflict


def _client_error(*, code: str, status: int) -> ClientError:
    """A real ClientError shaped exactly as boto populates it for a failed PUT."""
    return ClientError(
        {
            "Error": {"Code": code, "Message": "x"},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        "PutObject",
    )


def test_precondition_failed_412_is_conflict() -> None:
    err = _client_error(code="PreconditionFailed", status=412)
    assert is_s3_conditional_conflict(err) is True


def test_conditional_request_conflict_409_is_conflict() -> None:
    err = _client_error(code="ConditionalRequestConflict", status=409)
    assert is_s3_conditional_conflict(err) is True


def test_status_only_412_is_conflict() -> None:
    """Match on HTTP status even when the error Code is absent/unexpected."""
    err = _client_error(code="", status=412)
    assert is_s3_conditional_conflict(err) is True


def test_status_only_409_is_conflict() -> None:
    err = _client_error(code="", status=409)
    assert is_s3_conditional_conflict(err) is True


def test_unrelated_client_error_is_not_conflict() -> None:
    """A 404/NoSuchKey (or any non-conflict status) is not an insert conflict."""
    err = _client_error(code="NoSuchKey", status=404)
    assert is_s3_conditional_conflict(err) is False


def test_access_denied_403_is_not_conflict() -> None:
    err = _client_error(code="AccessDenied", status=403)
    assert is_s3_conditional_conflict(err) is False


def test_non_client_error_is_not_conflict() -> None:
    """An exception with no ClientError ``response`` dict is never a conflict."""
    assert is_s3_conditional_conflict(RuntimeError("boom")) is False
    assert is_s3_conditional_conflict(ValueError("x")) is False
