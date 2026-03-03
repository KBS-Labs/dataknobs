"""Tests for tools/response.py standardized response utilities."""

from __future__ import annotations

from dataknobs_bots.tools.response import error_response, success_response


class TestSuccessResponse:
    """Tests for success_response()."""

    def test_basic(self) -> None:
        result = success_response()
        assert result == {"success": True}

    def test_with_fields(self) -> None:
        result = success_response(record_id="abc", data={"name": "flour"})
        assert result == {
            "success": True,
            "record_id": "abc",
            "data": {"name": "flour"},
        }

    def test_success_key_always_true(self) -> None:
        result = success_response(count=0, items=[])
        assert result["success"] is True


class TestErrorResponse:
    """Tests for error_response()."""

    def test_basic(self) -> None:
        result = error_response("Something went wrong")
        assert result == {"success": False, "error": "Something went wrong"}

    def test_with_details(self) -> None:
        result = error_response("Validation failed", details=["name required"])
        assert result == {
            "success": False,
            "error": "Validation failed",
            "details": ["name required"],
        }

    def test_without_details_omits_key(self) -> None:
        result = error_response("Bad request")
        assert "details" not in result

    def test_empty_details_included(self) -> None:
        result = error_response("Error", details=[])
        assert result["details"] == []

    def test_with_extra_fields(self) -> None:
        result = error_response(
            "Duplicate found",
            existing_record={"record_id": "r1", "name": "flour"},
        )
        assert result == {
            "success": False,
            "error": "Duplicate found",
            "existing_record": {"record_id": "r1", "name": "flour"},
        }
