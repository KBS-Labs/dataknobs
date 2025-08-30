"""Unit tests for vector field functionality."""

import pytest

from dataknobs_data import Field, FieldType, Record, VectorField
from dataknobs_data.vector import (
    DistanceMetric,
    VectorConfig,
    VectorDimensionError,
    VectorIndexConfig,
    VectorMetadata,
    VectorSearchResult,
    compute_distance,
    compute_similarity,
    normalize_vector,
    validate_vector_dimensions,
)

# Skip tests if numpy is not available
np = pytest.importorskip("numpy")


class TestVectorField:
    """Test VectorField class functionality."""

    def test_vector_field_creation_from_list(self):
        """Test creating a vector field from a list."""
        vector_data = [0.1, 0.2, 0.3, 0.4]
        field = VectorField(
            value=vector_data,
            name="embedding",
            dimensions=4,
            source_field="text",
            model_name="test-model",
            model_version="1.0",
        )
        
        assert field.name == "embedding"
        assert field.type == FieldType.VECTOR
        assert field.dimensions == 4
        assert field.source_field == "text"
        assert field.model_name == "test-model"
        assert field.model_version == "1.0"
        assert isinstance(field.value, np.ndarray)
        assert field.value.dtype == np.float32
        assert np.allclose(field.value, vector_data)
    
    def test_vector_field_creation_from_numpy(self):
        """Test creating a vector field from numpy array."""
        vector_data = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
        field = VectorField(
            value=vector_data,
            name="embedding",
        )
        
        assert field.dimensions == 4
        assert field.value.dtype == np.float32  # Should convert to float32
        assert np.allclose(field.value, vector_data)
    
    def test_vector_field_dimension_validation(self):
        """Test dimension validation."""
        vector_data = [0.1, 0.2, 0.3, 0.4]
        
        # Should succeed with matching dimensions
        field = VectorField(
            value=vector_data,
            name="embedding",
            dimensions=4,
        )
        assert field.dimensions == 4
        
        # Should fail with mismatched dimensions
        with pytest.raises(ValueError, match="dimension mismatch"):
            VectorField(
                value=vector_data,
                name="embedding",
                dimensions=5,
            )
    
    def test_vector_field_auto_dimension_detection(self):
        """Test automatic dimension detection."""
        vector_data = [0.1, 0.2, 0.3, 0.4, 0.5]
        field = VectorField(
            value=vector_data,
            name="embedding",
        )
        assert field.dimensions == 5
    
    def test_vector_field_invalid_type(self):
        """Test error handling for invalid vector types."""
        with pytest.raises(TypeError, match="must be numpy array or list"):
            VectorField(
                value="not a vector",
                name="embedding",
            )
    
    def test_vector_field_metadata(self):
        """Test vector field metadata handling."""
        field = VectorField(
            value=[0.1, 0.2],
            name="embedding",
            source_field="text",
            model_name="test-model",
            model_version="1.0",
            metadata={"custom": "value"},
        )
        
        assert "dimensions" in field.metadata
        assert field.metadata["dimensions"] == 2
        assert "source_field" in field.metadata
        assert field.metadata["source_field"] == "text"
        assert "model" in field.metadata
        assert field.metadata["model"]["name"] == "test-model"
        assert field.metadata["model"]["version"] == "1.0"
        assert field.metadata["custom"] == "value"
    
    def test_vector_field_validate(self):
        """Test vector field validation."""
        # Valid vector
        field = VectorField(
            value=[0.1, 0.2, 0.3],
            name="embedding",
        )
        assert field.validate() is True
        
        # Test None value
        field.value = None
        assert field.validate() is True
        
        # Test invalid type (manually set)
        field.value = "not a vector"
        assert field.validate() is False
    
    def test_vector_field_to_list(self):
        """Test converting vector to list."""
        vector_data = [0.1, 0.2, 0.3]
        field = VectorField(
            value=vector_data,
            name="embedding",
        )
        
        result = field.to_list()
        assert isinstance(result, list)
        # Use np.allclose for floating-point comparison
        assert np.allclose(result, vector_data)
    
    def test_vector_field_cosine_similarity(self):
        """Test cosine similarity computation."""
        vec1 = VectorField([1.0, 0.0, 0.0], name="v1")
        vec2 = VectorField([1.0, 0.0, 0.0], name="v2")
        vec3 = VectorField([0.0, 1.0, 0.0], name="v3")
        vec4 = VectorField([0.5, 0.5, 0.0], name="v4")
        
        # Same vectors should have similarity 1
        assert np.isclose(vec1.cosine_similarity(vec2), 1.0)
        
        # Orthogonal vectors should have similarity 0
        assert np.isclose(vec1.cosine_similarity(vec3), 0.0)
        
        # Test with different input types
        assert np.isclose(vec1.cosine_similarity([1.0, 0.0, 0.0]), 1.0)
        assert np.isclose(vec1.cosine_similarity(np.array([1.0, 0.0, 0.0])), 1.0)
        
        # Test intermediate similarity
        similarity = vec1.cosine_similarity(vec4)
        assert 0 < similarity < 1
    
    def test_vector_field_euclidean_distance(self):
        """Test Euclidean distance computation."""
        vec1 = VectorField([0.0, 0.0], name="v1")
        vec2 = VectorField([3.0, 4.0], name="v2")
        
        # Distance should be 5 (3-4-5 triangle)
        assert np.isclose(vec1.euclidean_distance(vec2), 5.0)
        
        # Same vector should have distance 0
        assert np.isclose(vec1.euclidean_distance(vec1), 0.0)
        
        # Test with different input types
        assert np.isclose(vec1.euclidean_distance([3.0, 4.0]), 5.0)
        assert np.isclose(vec1.euclidean_distance(np.array([3.0, 4.0])), 5.0)
    
    def test_vector_field_serialization(self):
        """Test vector field serialization and deserialization."""
        original = VectorField(
            value=[0.1, 0.2, 0.3],
            name="embedding",
            source_field="text",
            model_name="test-model",
            model_version="1.0",
        )
        
        # Convert to dict
        data = original.to_dict()
        assert data["name"] == "embedding"
        assert data["type"] == "vector"
        # Use np.allclose for floating-point comparison
        assert np.allclose(data["value"], [0.1, 0.2, 0.3])
        assert data["dimensions"] == 3
        
        # Recreate from dict
        restored = VectorField.from_dict(data)
        assert restored.name == original.name
        assert restored.dimensions == original.dimensions
        assert restored.source_field == original.source_field
        assert restored.model_name == original.model_name
        assert restored.model_version == original.model_version
        assert np.allclose(restored.value, original.value)
    
    def test_vector_field_in_record(self):
        """Test vector field integration with Record."""
        record = Record(
            data={
                "id": "123",
                "text": "sample text",
            }
        )
        
        # Add vector field
        vector_field = VectorField(
            value=[0.1, 0.2, 0.3],
            name="embedding",
            source_field="text",
        )
        record.fields["embedding"] = vector_field
        
        assert "embedding" in record
        assert record.has_field("embedding")
        assert isinstance(record.fields["embedding"], VectorField)
        assert record.fields["embedding"].source_field == "text"
    
    def test_field_from_dict_with_vector(self):
        """Test Field.from_dict properly creates VectorField."""
        data = {
            "name": "embedding",
            "value": [0.1, 0.2, 0.3],
            "type": "vector",
            "metadata": {
                "dimensions": 3,
                "source_field": "text",
            },
        }
        
        field = Field.from_dict(data)
        assert isinstance(field, VectorField)
        assert field.dimensions == 3
        assert field.source_field == "text"


class TestVectorTypes:
    """Test vector type definitions."""

    def test_distance_metric_enum(self):
        """Test DistanceMetric enumeration."""
        assert DistanceMetric.COSINE.value == "cosine"
        assert DistanceMetric.EUCLIDEAN.value == "euclidean"
        assert DistanceMetric.DOT_PRODUCT.value == "dot_product"
        
        # Test aliases
        aliases = DistanceMetric.COSINE.get_aliases()
        assert "cosine_similarity" in aliases
        assert "cos" in aliases
    
    def test_vector_config(self):
        """Test VectorConfig validation."""
        # Valid config
        config = VectorConfig(
            dimensions=768,
            metric=DistanceMetric.COSINE,
            normalize=True,
            source_field="text",
            model_name="bert",
        )
        config.validate()  # Should not raise
        
        # Invalid dimensions
        with pytest.raises(ValueError, match="must be positive"):
            config = VectorConfig(dimensions=0)
            config.validate()
        
        with pytest.raises(ValueError, match="exceeds maximum"):
            config = VectorConfig(dimensions=100000)
            config.validate()
    
    def test_vector_index_config(self):
        """Test VectorIndexConfig optimal parameters."""
        config = VectorIndexConfig(index_type="auto")
        
        # Small dataset - flat index
        params = config.get_optimal_params(1000)
        assert params["type"] == "flat"
        
        # Medium dataset - IVFFlat
        params = config.get_optimal_params(100000)
        assert params["type"] == "ivfflat"
        assert "lists" in params
        assert "probes" in params
        
        # Large dataset - HNSW
        params = config.get_optimal_params(10000000)
        assert params["type"] == "hnsw"
        assert "m" in params
        assert "ef_construction" in params
        assert "ef_search" in params
    
    def test_vector_metadata(self):
        """Test VectorMetadata conversion."""
        metadata = VectorMetadata(
            dimensions=768,
            source_field="text",
            model_name="bert",
            model_version="1.0",
            created_at="2024-01-01T00:00:00Z",
            index_type="hnsw",
            metric="cosine",
        )
        
        # Convert to dict
        data = metadata.to_dict()
        assert data["dimensions"] == 768
        assert data["source_field"] == "text"
        assert data["model"]["name"] == "bert"
        assert data["model"]["version"] == "1.0"
        
        # Restore from dict
        restored = VectorMetadata.from_dict(data)
        assert restored.dimensions == metadata.dimensions
        assert restored.source_field == metadata.source_field
        assert restored.model_name == metadata.model_name
        assert restored.model_version == metadata.model_version
    
    def test_vector_search_result(self):
        """Test VectorSearchResult."""
        record = Record({"id": "123", "text": "sample"})
        
        result = VectorSearchResult(
            record=record,
            score=0.95,
            source_text="sample",
            vector_field="embedding",
            metadata={"custom": "data"},
        )
        
        assert result.record.id == "123"
        assert result.score == 0.95
        assert result.source_text == "sample"
        assert result.vector_field == "embedding"
        assert result.metadata["custom"] == "data"
        
        # Test comparison
        result2 = VectorSearchResult(record=record, score=0.85)
        assert result2 < result  # Lower score should be "less than"


class TestVectorOperations:
    """Test vector operation utilities."""

    def test_normalize_vector(self):
        """Test vector normalization."""
        vector = np.array([3.0, 4.0])
        normalized = normalize_vector(vector)
        
        # Should have unit length
        assert np.isclose(np.linalg.norm(normalized), 1.0)
        
        # Should maintain direction
        assert np.allclose(normalized, [0.6, 0.8])
        
        # Zero vector should remain zero
        zero_vec = np.array([0.0, 0.0])
        assert np.allclose(normalize_vector(zero_vec), zero_vec)
    
    def test_compute_distance(self):
        """Test distance computation."""
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([0.0, 1.0])
        
        # Cosine distance
        dist = compute_distance(vec1, vec2, DistanceMetric.COSINE)
        assert np.isclose(dist, 1.0)  # Orthogonal vectors
        
        # Euclidean distance
        dist = compute_distance(vec1, vec2, DistanceMetric.EUCLIDEAN)
        assert np.isclose(dist, np.sqrt(2))
        
        # L1 distance
        dist = compute_distance(vec1, vec2, DistanceMetric.L1)
        assert np.isclose(dist, 2.0)
        
        # Dot product (negative for distance)
        dist = compute_distance(vec1, vec2, DistanceMetric.DOT_PRODUCT)
        assert np.isclose(dist, 0.0)
    
    def test_compute_similarity(self):
        """Test similarity computation."""
        vec1 = np.array([1.0, 0.0])
        vec2 = np.array([1.0, 0.0])
        
        # Cosine similarity
        sim = compute_similarity(vec1, vec2, DistanceMetric.COSINE)
        assert np.isclose(sim, 1.0)  # Same vectors
        
        # Euclidean-based similarity
        sim = compute_similarity(vec1, vec2, DistanceMetric.EUCLIDEAN)
        assert np.isclose(sim, 1.0)  # Same vectors
        
        # Dot product similarity
        sim = compute_similarity(vec1, vec2, DistanceMetric.DOT_PRODUCT)
        assert np.isclose(sim, 1.0)
    
    def test_validate_vector_dimensions(self):
        """Test dimension validation utility."""
        # Valid dimensions
        vector = validate_vector_dimensions([0.1, 0.2, 0.3], 3)
        assert isinstance(vector, np.ndarray)
        assert len(vector) == 3
        
        # Invalid dimensions
        with pytest.raises(ValueError, match="dimension mismatch"):
            validate_vector_dimensions([0.1, 0.2], 3, field_name="test")
    
    def test_vector_exceptions(self):
        """Test vector-specific exceptions."""
        # Dimension error
        error = VectorDimensionError(expected=768, actual=512, field_name="embedding")
        assert "768" in str(error)
        assert "512" in str(error)
        assert "embedding" in str(error)