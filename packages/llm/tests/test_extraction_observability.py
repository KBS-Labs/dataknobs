"""Tests for extraction observability module."""

import time

import pytest

from dataknobs_llm.extraction.observability import (
    ExtractionHistoryQuery,
    ExtractionRecord,
    ExtractionStats,
    ExtractionTracker,
    create_extraction_record,
)


class TestExtractionRecord:
    """Tests for ExtractionRecord dataclass."""

    def test_create_record(self) -> None:
        """Test creating an extraction record with required fields."""
        timestamp = time.time()
        record = ExtractionRecord(
            timestamp=timestamp,
            input_text="My name is Alice",
            extracted_data={"name": "Alice"},
            confidence=0.95,
            validation_errors=[],
            duration_ms=150.5,
            success=True,
        )

        assert record.timestamp == timestamp
        assert record.input_text == "My name is Alice"
        assert record.extracted_data == {"name": "Alice"}
        assert record.confidence == 0.95
        assert record.validation_errors == []
        assert record.duration_ms == 150.5
        assert record.success is True
        # Optional fields should have defaults
        assert record.schema_name is None
        assert record.schema_hash is None
        assert record.model_used is None
        assert record.provider is None
        assert record.context is None
        assert record.raw_response is None
        assert record.input_length == 0
        assert record.truncated is False

    def test_create_record_with_all_fields(self) -> None:
        """Test creating a record with all fields specified."""
        record = ExtractionRecord(
            timestamp=1234567890.0,
            input_text="I want 3 large pizzas",
            schema_name="OrderSchema",
            schema_hash="abc12345",
            extracted_data={"quantity": 3, "size": "large"},
            confidence=0.92,
            validation_errors=[],
            duration_ms=200.0,
            success=True,
            model_used="claude-3-haiku",
            provider="anthropic",
            context={"stage": "order_details"},
            raw_response='{"quantity": 3, "size": "large"}',
            input_length=25,
            truncated=False,
        )

        assert record.schema_name == "OrderSchema"
        assert record.schema_hash == "abc12345"
        assert record.model_used == "claude-3-haiku"
        assert record.provider == "anthropic"
        assert record.context == {"stage": "order_details"}
        assert record.raw_response == '{"quantity": 3, "size": "large"}'
        assert record.input_length == 25
        assert record.truncated is False

    def test_create_failed_record(self) -> None:
        """Test creating a record for a failed extraction."""
        record = ExtractionRecord(
            timestamp=time.time(),
            input_text="gibberish",
            extracted_data={},
            confidence=0.0,
            validation_errors=["Could not parse JSON from response"],
            duration_ms=100.0,
            success=False,
        )

        assert record.success is False
        assert record.extracted_data == {}
        assert record.confidence == 0.0
        assert len(record.validation_errors) == 1

    def test_create_record_with_validation_errors(self) -> None:
        """Test creating a record with validation errors."""
        record = ExtractionRecord(
            timestamp=time.time(),
            input_text="Alice",
            extracted_data={"name": "Alice"},
            confidence=0.5,
            validation_errors=["Missing required field: age", "Missing required field: email"],
            duration_ms=150.0,
            success=False,
        )

        assert record.success is False
        assert len(record.validation_errors) == 2
        assert "Missing required field: age" in record.validation_errors

    def test_to_dict(self) -> None:
        """Test converting record to dictionary."""
        record = ExtractionRecord(
            timestamp=1234567890.0,
            input_text="test input",
            schema_name="TestSchema",
            extracted_data={"key": "value"},
            confidence=0.9,
            validation_errors=[],
            duration_ms=100.0,
            success=True,
            model_used="test-model",
        )

        data = record.to_dict()

        assert data["timestamp"] == 1234567890.0
        assert data["input_text"] == "test input"
        assert data["schema_name"] == "TestSchema"
        assert data["extracted_data"] == {"key": "value"}
        assert data["confidence"] == 0.9
        assert data["validation_errors"] == []
        assert data["duration_ms"] == 100.0
        assert data["success"] is True
        assert data["model_used"] == "test-model"

    def test_from_dict(self) -> None:
        """Test creating record from dictionary."""
        data = {
            "timestamp": 1000.0,
            "input_text": "input text",
            "extracted_data": {"name": "Bob"},
            "confidence": 0.85,
            "validation_errors": ["warning"],
            "duration_ms": 200.0,
            "success": False,
            "schema_name": "PersonSchema",
            "schema_hash": "hash123",
            "model_used": "qwen3",
            "provider": "ollama",
            "context": {"stage": "gather"},
            "raw_response": '{"name": "Bob"}',
            "input_length": 10,
            "truncated": False,
        }

        record = ExtractionRecord.from_dict(data)

        assert record.timestamp == 1000.0
        assert record.input_text == "input text"
        assert record.extracted_data == {"name": "Bob"}
        assert record.confidence == 0.85
        assert record.validation_errors == ["warning"]
        assert record.success is False
        assert record.schema_name == "PersonSchema"
        assert record.provider == "ollama"


class TestExtractionStats:
    """Tests for ExtractionStats dataclass."""

    def test_default_stats(self) -> None:
        """Test stats with default values."""
        stats = ExtractionStats()

        assert stats.total_extractions == 0
        assert stats.successful_extractions == 0
        assert stats.failed_extractions == 0
        assert stats.avg_confidence == 0.0
        assert stats.avg_duration_ms == 0.0
        assert stats.success_rate == 0.0
        assert stats.most_common_errors == []
        assert stats.first_extraction is None
        assert stats.last_extraction is None
        assert stats.by_schema == {}
        assert stats.by_model == {}

    def test_stats_with_values(self) -> None:
        """Test stats with custom values."""
        stats = ExtractionStats(
            total_extractions=100,
            successful_extractions=85,
            failed_extractions=15,
            avg_confidence=0.87,
            avg_duration_ms=150.0,
            success_rate=0.85,
            most_common_errors=[("Missing field", 10), ("Parse error", 5)],
            first_extraction=1000.0,
            last_extraction=2000.0,
            by_schema={"PersonSchema": 50, "OrderSchema": 50},
            by_model={"claude-3-haiku": 60, "qwen3": 40},
        )

        assert stats.total_extractions == 100
        assert stats.successful_extractions == 85
        assert stats.failed_extractions == 15
        assert stats.success_rate == 0.85
        assert len(stats.most_common_errors) == 2
        assert stats.by_schema["PersonSchema"] == 50


class TestExtractionHistoryQuery:
    """Tests for ExtractionHistoryQuery dataclass."""

    def test_default_query(self) -> None:
        """Test query with default values."""
        query = ExtractionHistoryQuery()

        assert query.schema_name is None
        assert query.model is None
        assert query.success_only is False
        assert query.failed_only is False
        assert query.min_confidence is None
        assert query.since is None
        assert query.until is None
        assert query.limit is None

    def test_custom_query(self) -> None:
        """Test query with custom values."""
        query = ExtractionHistoryQuery(
            schema_name="PersonSchema",
            model="claude-3-haiku",
            success_only=True,
            min_confidence=0.8,
            since=1000.0,
            until=2000.0,
            limit=50,
        )

        assert query.schema_name == "PersonSchema"
        assert query.model == "claude-3-haiku"
        assert query.success_only is True
        assert query.min_confidence == 0.8
        assert query.since == 1000.0
        assert query.limit == 50


class TestExtractionTracker:
    """Tests for ExtractionTracker class."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        tracker = ExtractionTracker()
        assert len(tracker) == 0

    def test_init_with_max_history(self) -> None:
        """Test initialization with custom max history."""
        tracker = ExtractionTracker(max_history=50)
        assert len(tracker) == 0

    def test_record_extraction(self) -> None:
        """Test recording an extraction."""
        tracker = ExtractionTracker()

        record = ExtractionRecord(
            timestamp=time.time(),
            input_text="test",
            extracted_data={"key": "value"},
            confidence=0.9,
            validation_errors=[],
            duration_ms=100.0,
            success=True,
        )

        tracker.record(record)
        assert len(tracker) == 1

    def test_max_history_enforcement(self) -> None:
        """Test that max history is enforced."""
        tracker = ExtractionTracker(max_history=5)

        for i in range(10):
            tracker.record(
                ExtractionRecord(
                    timestamp=time.time() + i,
                    input_text=f"input_{i}",
                    extracted_data={"i": i},
                    confidence=0.9,
                    validation_errors=[],
                    duration_ms=100.0,
                    success=True,
                )
            )

        assert len(tracker) == 5
        # Should keep the most recent 5
        results = tracker.query()
        assert results[0].input_text == "input_5"
        assert results[-1].input_text == "input_9"

    def test_query_all(self) -> None:
        """Test querying all records."""
        tracker = ExtractionTracker()

        for i in range(3):
            tracker.record(
                ExtractionRecord(
                    timestamp=time.time(),
                    input_text=f"input_{i}",
                    extracted_data={},
                    confidence=0.9,
                    validation_errors=[],
                    duration_ms=100.0,
                    success=True,
                )
            )

        results = tracker.query()
        assert len(results) == 3

    def test_query_by_schema_name(self) -> None:
        """Test filtering by schema name."""
        tracker = ExtractionTracker()

        tracker.record(
            ExtractionRecord(
                timestamp=time.time(),
                input_text="Alice",
                schema_name="PersonSchema",
                extracted_data={"name": "Alice"},
                confidence=0.9,
                validation_errors=[],
                duration_ms=100.0,
                success=True,
            )
        )
        tracker.record(
            ExtractionRecord(
                timestamp=time.time(),
                input_text="3 pizzas",
                schema_name="OrderSchema",
                extracted_data={"quantity": 3},
                confidence=0.9,
                validation_errors=[],
                duration_ms=100.0,
                success=True,
            )
        )

        query = ExtractionHistoryQuery(schema_name="PersonSchema")
        results = tracker.query(query)

        assert len(results) == 1
        assert results[0].schema_name == "PersonSchema"

    def test_query_by_model(self) -> None:
        """Test filtering by model."""
        tracker = ExtractionTracker()

        tracker.record(
            ExtractionRecord(
                timestamp=time.time(),
                input_text="test1",
                extracted_data={},
                confidence=0.9,
                validation_errors=[],
                duration_ms=100.0,
                success=True,
                model_used="claude-3-haiku",
            )
        )
        tracker.record(
            ExtractionRecord(
                timestamp=time.time(),
                input_text="test2",
                extracted_data={},
                confidence=0.9,
                validation_errors=[],
                duration_ms=100.0,
                success=True,
                model_used="qwen3",
            )
        )

        query = ExtractionHistoryQuery(model="qwen3")
        results = tracker.query(query)

        assert len(results) == 1
        assert results[0].model_used == "qwen3"

    def test_query_success_only(self) -> None:
        """Test filtering for successful extractions only."""
        tracker = ExtractionTracker()

        tracker.record(
            ExtractionRecord(
                timestamp=time.time(),
                input_text="good",
                extracted_data={"data": "value"},
                confidence=0.9,
                validation_errors=[],
                duration_ms=100.0,
                success=True,
            )
        )
        tracker.record(
            ExtractionRecord(
                timestamp=time.time(),
                input_text="bad",
                extracted_data={},
                confidence=0.0,
                validation_errors=["error"],
                duration_ms=100.0,
                success=False,
            )
        )

        query = ExtractionHistoryQuery(success_only=True)
        results = tracker.query(query)

        assert len(results) == 1
        assert results[0].success is True

    def test_query_failed_only(self) -> None:
        """Test filtering for failed extractions only."""
        tracker = ExtractionTracker()

        tracker.record(
            ExtractionRecord(
                timestamp=time.time(),
                input_text="good",
                extracted_data={"data": "value"},
                confidence=0.9,
                validation_errors=[],
                duration_ms=100.0,
                success=True,
            )
        )
        tracker.record(
            ExtractionRecord(
                timestamp=time.time(),
                input_text="bad",
                extracted_data={},
                confidence=0.0,
                validation_errors=["error"],
                duration_ms=100.0,
                success=False,
            )
        )

        query = ExtractionHistoryQuery(failed_only=True)
        results = tracker.query(query)

        assert len(results) == 1
        assert results[0].success is False

    def test_query_min_confidence(self) -> None:
        """Test filtering by minimum confidence."""
        tracker = ExtractionTracker()

        tracker.record(
            ExtractionRecord(
                timestamp=time.time(),
                input_text="high",
                extracted_data={},
                confidence=0.95,
                validation_errors=[],
                duration_ms=100.0,
                success=True,
            )
        )
        tracker.record(
            ExtractionRecord(
                timestamp=time.time(),
                input_text="low",
                extracted_data={},
                confidence=0.5,
                validation_errors=[],
                duration_ms=100.0,
                success=True,
            )
        )

        query = ExtractionHistoryQuery(min_confidence=0.8)
        results = tracker.query(query)

        assert len(results) == 1
        assert results[0].confidence >= 0.8

    def test_query_by_time_range(self) -> None:
        """Test filtering by time range."""
        tracker = ExtractionTracker()

        now = time.time()
        tracker.record(
            ExtractionRecord(
                timestamp=now - 100,
                input_text="old",
                extracted_data={},
                confidence=0.9,
                validation_errors=[],
                duration_ms=100.0,
                success=True,
            )
        )
        tracker.record(
            ExtractionRecord(
                timestamp=now,
                input_text="recent",
                extracted_data={},
                confidence=0.9,
                validation_errors=[],
                duration_ms=100.0,
                success=True,
            )
        )

        # Query for recent records only
        query = ExtractionHistoryQuery(since=now - 50)
        results = tracker.query(query)

        assert len(results) == 1
        assert results[0].input_text == "recent"

    def test_query_with_limit(self) -> None:
        """Test limiting query results."""
        tracker = ExtractionTracker()

        for i in range(10):
            tracker.record(
                ExtractionRecord(
                    timestamp=time.time() + i,
                    input_text=f"input_{i}",
                    extracted_data={},
                    confidence=0.9,
                    validation_errors=[],
                    duration_ms=100.0,
                    success=True,
                )
            )

        query = ExtractionHistoryQuery(limit=3)
        results = tracker.query(query)

        assert len(results) == 3
        # Should return last 3 (most recent)
        assert results[0].input_text == "input_7"
        assert results[2].input_text == "input_9"

    def test_get_stats_empty(self) -> None:
        """Test getting stats with no records."""
        tracker = ExtractionTracker()
        stats = tracker.get_stats()

        assert stats.total_extractions == 0
        assert stats.successful_extractions == 0

    def test_get_stats_with_data(self) -> None:
        """Test getting stats with records."""
        tracker = ExtractionTracker()

        # Add some extractions
        tracker.record(
            ExtractionRecord(
                timestamp=1000.0,
                input_text="test1",
                schema_name="SchemaA",
                extracted_data={"a": 1},
                confidence=0.9,
                validation_errors=[],
                duration_ms=100.0,
                success=True,
                model_used="model1",
            )
        )
        tracker.record(
            ExtractionRecord(
                timestamp=2000.0,
                input_text="test2",
                schema_name="SchemaA",
                extracted_data={},
                confidence=0.5,
                validation_errors=["Missing field: x"],
                duration_ms=200.0,
                success=False,
                model_used="model1",
            )
        )
        tracker.record(
            ExtractionRecord(
                timestamp=3000.0,
                input_text="test3",
                schema_name="SchemaB",
                extracted_data={"b": 2},
                confidence=0.85,
                validation_errors=[],
                duration_ms=150.0,
                success=True,
                model_used="model2",
            )
        )

        stats = tracker.get_stats()

        assert stats.total_extractions == 3
        assert stats.successful_extractions == 2
        assert stats.failed_extractions == 1
        assert stats.avg_confidence == pytest.approx((0.9 + 0.5 + 0.85) / 3)
        assert stats.avg_duration_ms == pytest.approx((100 + 200 + 150) / 3)
        assert stats.success_rate == pytest.approx(2 / 3)
        assert stats.first_extraction == 1000.0
        assert stats.last_extraction == 3000.0
        assert stats.by_schema == {"SchemaA": 2, "SchemaB": 1}
        assert stats.by_model == {"model1": 2, "model2": 1}

    def test_get_stats_most_common_errors(self) -> None:
        """Test that most common errors are tracked."""
        tracker = ExtractionTracker()

        # Add records with various errors
        for _ in range(5):
            tracker.record(
                ExtractionRecord(
                    timestamp=time.time(),
                    input_text="test",
                    extracted_data={},
                    confidence=0.0,
                    validation_errors=["Missing field: name"],
                    duration_ms=100.0,
                    success=False,
                )
            )
        for _ in range(3):
            tracker.record(
                ExtractionRecord(
                    timestamp=time.time(),
                    input_text="test",
                    extracted_data={},
                    confidence=0.0,
                    validation_errors=["Parse error"],
                    duration_ms=100.0,
                    success=False,
                )
            )

        stats = tracker.get_stats()

        assert len(stats.most_common_errors) >= 2
        # Most common should be first
        assert stats.most_common_errors[0] == ("Missing field: name", 5)
        assert stats.most_common_errors[1] == ("Parse error", 3)

    def test_get_recent(self) -> None:
        """Test getting most recent records."""
        tracker = ExtractionTracker()

        for i in range(10):
            tracker.record(
                ExtractionRecord(
                    timestamp=time.time() + i,
                    input_text=f"input_{i}",
                    extracted_data={},
                    confidence=0.9,
                    validation_errors=[],
                    duration_ms=100.0,
                    success=True,
                )
            )

        recent = tracker.get_recent(3)

        assert len(recent) == 3
        assert recent[0].input_text == "input_7"
        assert recent[2].input_text == "input_9"

    def test_clear(self) -> None:
        """Test clearing extraction history."""
        tracker = ExtractionTracker()

        for i in range(5):
            tracker.record(
                ExtractionRecord(
                    timestamp=time.time(),
                    input_text=f"input_{i}",
                    extracted_data={},
                    confidence=0.9,
                    validation_errors=[],
                    duration_ms=100.0,
                    success=True,
                )
            )

        assert len(tracker) == 5

        tracker.clear()

        assert len(tracker) == 0
        assert tracker.query() == []


class TestCreateExtractionRecord:
    """Tests for the create_extraction_record factory function."""

    def test_create_with_factory(self) -> None:
        """Test the factory function sets timestamp automatically."""
        before = time.time()
        record = create_extraction_record(
            input_text="My name is Alice",
            extracted_data={"name": "Alice"},
            confidence=0.95,
            validation_errors=[],
            duration_ms=150.0,
        )
        after = time.time()

        assert before <= record.timestamp <= after
        assert record.input_text == "My name is Alice"
        assert record.extracted_data == {"name": "Alice"}
        assert record.confidence == 0.95
        assert record.success is True  # High confidence, no errors

    def test_create_with_schema(self) -> None:
        """Test factory with schema provided."""
        record = create_extraction_record(
            input_text="test",
            extracted_data={"name": "Test"},
            confidence=0.9,
            validation_errors=[],
            duration_ms=100.0,
            schema={"title": "PersonSchema", "type": "object"},
        )

        assert record.schema_name == "PersonSchema"
        assert record.schema_hash is not None
        assert len(record.schema_hash) == 8

    def test_create_with_all_fields(self) -> None:
        """Test factory with all optional fields."""
        record = create_extraction_record(
            input_text="test input",
            extracted_data={"key": "value"},
            confidence=0.85,
            validation_errors=[],
            duration_ms=200.0,
            schema={"title": "TestSchema", "type": "object"},
            model_used="claude-3-haiku",
            provider="anthropic",
            context={"stage": "configure"},
            raw_response='{"key": "value"}',
        )

        assert record.model_used == "claude-3-haiku"
        assert record.provider == "anthropic"
        assert record.context == {"stage": "configure"}
        assert record.raw_response == '{"key": "value"}'

    def test_create_truncates_long_input(self) -> None:
        """Test that long input text is truncated."""
        long_text = "x" * 500
        record = create_extraction_record(
            input_text=long_text,
            extracted_data={},
            confidence=0.9,
            validation_errors=[],
            duration_ms=100.0,
            truncate_at=200,
        )

        assert len(record.input_text) == 203  # 200 + "..."
        assert record.input_text.endswith("...")
        assert record.input_length == 500
        assert record.truncated is True

    def test_create_truncates_long_response(self) -> None:
        """Test that long raw response is truncated."""
        long_response = "y" * 500
        record = create_extraction_record(
            input_text="short",
            extracted_data={},
            confidence=0.9,
            validation_errors=[],
            duration_ms=100.0,
            raw_response=long_response,
            truncate_at=200,
        )

        assert len(record.raw_response) == 203  # 200 + "..."
        assert record.raw_response.endswith("...")
        assert record.truncated is True

    def test_create_success_based_on_confidence_and_errors(self) -> None:
        """Test that success is determined by confidence >= 0.8 and no errors."""
        # High confidence, no errors -> success
        record1 = create_extraction_record(
            input_text="test",
            extracted_data={"key": "value"},
            confidence=0.9,
            validation_errors=[],
            duration_ms=100.0,
        )
        assert record1.success is True

        # High confidence with errors -> not success
        record2 = create_extraction_record(
            input_text="test",
            extracted_data={"key": "value"},
            confidence=0.9,
            validation_errors=["warning"],
            duration_ms=100.0,
        )
        assert record2.success is False

        # Low confidence, no errors -> not success
        record3 = create_extraction_record(
            input_text="test",
            extracted_data={"key": "value"},
            confidence=0.5,
            validation_errors=[],
            duration_ms=100.0,
        )
        assert record3.success is False

    def test_create_at_confidence_threshold(self) -> None:
        """Test success at exactly 0.8 confidence threshold."""
        record = create_extraction_record(
            input_text="test",
            extracted_data={"key": "value"},
            confidence=0.8,
            validation_errors=[],
            duration_ms=100.0,
        )
        assert record.success is True

        record_below = create_extraction_record(
            input_text="test",
            extracted_data={"key": "value"},
            confidence=0.79,
            validation_errors=[],
            duration_ms=100.0,
        )
        assert record_below.success is False
