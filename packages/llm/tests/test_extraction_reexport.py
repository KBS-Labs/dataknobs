"""Tests for SimpleExtractionResult stable re-export (Gap 15)."""


def test_import_from_stable_path() -> None:
    """SimpleExtractionResult is importable from dataknobs_llm.extraction."""
    from dataknobs_llm.extraction import SimpleExtractionResult

    # Basic sanity — can construct and use is_confident
    result = SimpleExtractionResult(data={"name": "Alice"}, confidence=0.9)
    assert result.is_confident is True
    assert result.data == {"name": "Alice"}


def test_import_from_testing_still_works() -> None:
    """Original import path (dataknobs_llm.testing) remains valid."""
    from dataknobs_llm.testing import SimpleExtractionResult

    result = SimpleExtractionResult(data={}, confidence=0.5)
    assert result.is_confident is False


def test_same_class_from_both_paths() -> None:
    """Both import paths resolve to the same class object."""
    from dataknobs_llm.extraction import (
        SimpleExtractionResult as FromExtraction,
    )
    from dataknobs_llm.testing import SimpleExtractionResult as FromTesting

    assert FromExtraction is FromTesting
