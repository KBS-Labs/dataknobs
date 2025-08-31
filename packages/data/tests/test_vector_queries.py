"""Tests for vector query functionality."""

import json
import numpy as np
import pytest

from dataknobs_data.query import Operator, Query, SortOrder, VectorQuery
from dataknobs_data.query_logic import ComplexQuery, QueryBuilder
from dataknobs_data.vector.types import DistanceMetric


class TestVectorQuery:
    """Test VectorQuery dataclass."""
    
    def test_vector_query_creation(self):
        """Test creating a VectorQuery with various parameters."""
        vector = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        
        # Basic creation
        vq = VectorQuery(
            vector=vector,
            field_name="embedding",
            k=10,
            metric=DistanceMetric.COSINE,
        )
        
        assert np.array_equal(vq.vector, vector)
        assert vq.field_name == "embedding"
        assert vq.k == 10
        assert vq.metric == DistanceMetric.COSINE
        assert vq.include_source is True
        assert vq.score_threshold is None
        assert vq.rerank is False
        assert vq.rerank_k is None
    
    def test_vector_query_with_list(self):
        """Test VectorQuery with list input."""
        vector_list = [0.1, 0.2, 0.3]
        
        vq = VectorQuery(
            vector=vector_list,
            field_name="vec",
            k=5,
            metric="euclidean"
        )
        
        assert vq.vector == vector_list
        assert vq.metric == "euclidean"
    
    def test_vector_query_with_score_threshold(self):
        """Test VectorQuery with score threshold."""
        vector = np.array([0.1, 0.2, 0.3])
        
        vq = VectorQuery(
            vector=vector,
            score_threshold=0.8,
            rerank=True,
            rerank_k=20
        )
        
        assert vq.score_threshold == 0.8
        assert vq.rerank is True
        assert vq.rerank_k == 20
    
    def test_vector_query_serialization(self):
        """Test VectorQuery to_dict and from_dict."""
        vector = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        
        original = VectorQuery(
            vector=vector,
            field_name="custom_embedding",
            k=15,
            metric=DistanceMetric.DOT_PRODUCT,
            include_source=False,
            score_threshold=0.7,
            rerank=True,
            rerank_k=30,
            metadata={"alpha": 0.5}
        )
        
        # Serialize to dict
        data = original.to_dict()
        
        assert data["vector"] == vector.tolist()
        assert data["field"] == "custom_embedding"
        assert data["k"] == 15
        assert data["metric"] == "dot_product"
        assert data["include_source"] is False
        assert data["score_threshold"] == 0.7
        assert data["rerank"] is True
        assert data["rerank_k"] == 30
        assert data["metadata"] == {"alpha": 0.5}
        
        # Deserialize from dict
        restored = VectorQuery.from_dict(data)
        
        assert np.array_equal(restored.vector, vector)
        assert restored.field_name == "custom_embedding"
        assert restored.k == 15
        assert restored.metric == DistanceMetric.DOT_PRODUCT
        assert restored.include_source is False
        assert restored.score_threshold == 0.7
        assert restored.rerank is True
        assert restored.rerank_k == 30
        assert restored.metadata == {"alpha": 0.5}


class TestQueryVectorIntegration:
    """Test Query class with vector search support."""
    
    def test_query_similar_to(self):
        """Test Query.similar_to() method."""
        vector = np.array([0.1, 0.2, 0.3])
        
        query = Query().similar_to(
            vector=vector,
            field="embedding",
            k=20,
            metric=DistanceMetric.COSINE,
            score_threshold=0.5
        )
        
        assert query.vector_query is not None
        assert np.array_equal(query.vector_query.vector, vector)
        assert query.vector_query.field_name == "embedding"
        assert query.vector_query.k == 20
        assert query.vector_query.metric == DistanceMetric.COSINE
        assert query.vector_query.score_threshold == 0.5
        assert query.limit_value == 20  # Auto-set from k
    
    def test_query_near_text(self):
        """Test Query.near_text() method."""
        # Mock embedding function
        def mock_embedding_fn(text: str) -> np.ndarray:
            # Return a fixed vector based on text length (for testing)
            return np.array([float(len(text)) * 0.01] * 3)
        
        query = Query().near_text(
            text="search query",
            embedding_fn=mock_embedding_fn,
            field="text_embedding",
            k=15,
            metric="euclidean"
        )
        
        assert query.vector_query is not None
        expected_vector = mock_embedding_fn("search query")
        assert np.array_equal(query.vector_query.vector, expected_vector)
        assert query.vector_query.field_name == "text_embedding"
        assert query.vector_query.k == 15
        assert query.limit_value == 15
    
    def test_query_hybrid(self):
        """Test Query.hybrid() method for combined text and vector search."""
        vector = np.array([0.1, 0.2, 0.3])
        
        query = Query().hybrid(
            text_query="important document",
            vector=vector,
            text_field="content",
            vector_field="embedding",
            alpha=0.7,
            k=25
        )
        
        # Check text filter was added
        assert len(query.filters) == 1
        assert query.filters[0].field == "content"
        assert query.filters[0].operator == Operator.LIKE
        assert query.filters[0].value == "%important document%"
        
        # Check vector search was added
        assert query.vector_query is not None
        assert np.array_equal(query.vector_query.vector, vector)
        assert query.vector_query.field_name == "embedding"
        assert query.vector_query.k == 25
        assert query.vector_query.metadata["hybrid_alpha"] == 0.7
        assert query.limit_value == 25
    
    def test_query_with_reranking(self):
        """Test Query.with_reranking() method."""
        vector = np.array([0.1, 0.2, 0.3])
        
        query = (Query()
                .similar_to(vector=vector, k=10)
                .with_reranking(rerank_k=30))
        
        assert query.vector_query.rerank is True
        assert query.vector_query.rerank_k == 30
        
        # Test default rerank_k (2 * k)
        query2 = (Query()
                 .similar_to(vector=vector, k=10)
                 .with_reranking())
        
        assert query2.vector_query.rerank is True
        assert query2.vector_query.rerank_k == 20
    
    def test_query_clear_vector(self):
        """Test Query.clear_vector() method."""
        vector = np.array([0.1, 0.2, 0.3])
        
        query = Query().similar_to(vector=vector, k=10)
        assert query.vector_query is not None
        
        query.clear_vector()
        assert query.vector_query is None
    
    def test_combined_filters_and_vectors(self):
        """Test combining traditional filters with vector search."""
        vector = np.array([0.1, 0.2, 0.3])
        
        query = (Query()
                .filter("status", Operator.EQ, "published")
                .filter("category", Operator.IN, ["tech", "science"])
                .similar_to(vector=vector, field="embedding", k=10)
                .sort_by("created_at", SortOrder.DESC))
        
        assert len(query.filters) == 2
        assert query.vector_query is not None
        assert query.vector_query.k == 10
        assert len(query.sort_specs) == 1
    
    def test_query_vector_serialization(self):
        """Test Query serialization with vector_query."""
        vector = np.array([0.1, 0.2])
        
        original = (Query()
                   .filter("active", Operator.EQ, True)
                   .similar_to(vector=vector, k=5, score_threshold=0.9)
                   .sort_by("score"))
        
        # Serialize to dict
        data = original.to_dict()
        
        assert "vector_query" in data
        assert data["vector_query"]["vector"] == vector.tolist()
        assert data["vector_query"]["k"] == 5
        assert data["vector_query"]["score_threshold"] == 0.9
        assert len(data["filters"]) == 1
        assert len(data["sort"]) == 1
        
        # Deserialize from dict
        restored = Query.from_dict(data)
        
        assert restored.vector_query is not None
        assert np.allclose(restored.vector_query.vector, vector)
        assert restored.vector_query.k == 5
        assert restored.vector_query.score_threshold == 0.9
        assert len(restored.filters) == 1
        assert len(restored.sort_specs) == 1
    
    def test_query_copy_with_vector(self):
        """Test Query.copy() preserves vector_query."""
        vector = np.array([0.1, 0.2, 0.3])
        
        original = Query().similar_to(vector=vector, k=10)
        copy = original.copy()
        
        assert copy.vector_query is not None
        assert np.array_equal(copy.vector_query.vector, vector)
        assert copy.vector_query.k == 10
        
        # Ensure deep copy
        copy.vector_query.k = 20
        assert original.vector_query.k == 10


class TestComplexQueryVectorSupport:
    """Test ComplexQuery with vector support."""
    
    def test_complex_query_with_vector(self):
        """Test ComplexQuery can hold vector_query."""
        vector = np.array([0.5, 0.5])
        
        vq = VectorQuery(vector=vector, k=10)
        
        complex_query = ComplexQuery(
            vector_query=vq,
            limit_value=10
        )
        
        assert complex_query.vector_query is not None
        assert np.array_equal(complex_query.vector_query.vector, vector)
    
    def test_complex_query_to_simple_with_vector(self):
        """Test converting ComplexQuery to simple Query preserves vector_query."""
        from dataknobs_data.query import Filter
        from dataknobs_data.query_logic import FilterCondition, LogicCondition, LogicOperator
        
        vector = np.array([0.1, 0.2])
        vq = VectorQuery(vector=vector, k=5)
        
        # Create simple AND conditions
        condition = LogicCondition(
            operator=LogicOperator.AND,
            conditions=[
                FilterCondition(Filter("status", Operator.EQ, "active")),
                FilterCondition(Filter("type", Operator.IN, ["A", "B"]))
            ]
        )
        
        complex_query = ComplexQuery(
            condition=condition,
            vector_query=vq,
            limit_value=5
        )
        
        # Convert to simple query
        simple_query = complex_query.to_simple_query()
        
        assert len(simple_query.filters) == 2
        assert simple_query.vector_query is not None
        assert np.array_equal(simple_query.vector_query.vector, vector)
        assert simple_query.vector_query.k == 5
    
    def test_complex_query_serialization_with_vector(self):
        """Test ComplexQuery serialization with vector_query."""
        vector = np.array([0.3, 0.4, 0.5])
        
        original = ComplexQuery(
            vector_query=VectorQuery(vector=vector, k=15),
            limit_value=15
        )
        
        # Serialize
        data = original.to_dict()
        
        assert "vector_query" in data
        assert data["vector_query"]["vector"] == vector.tolist()
        assert data["vector_query"]["k"] == 15
        
        # Deserialize
        restored = ComplexQuery.from_dict(data)
        
        assert restored.vector_query is not None
        assert np.allclose(restored.vector_query.vector, vector)
        assert restored.vector_query.k == 15
        assert restored.limit_value == 15


class TestQueryBuilderVectorSupport:
    """Test QueryBuilder with vector support."""
    
    def test_query_builder_similar_to(self):
        """Test QueryBuilder.similar_to() method."""
        vector = np.array([0.1, 0.2, 0.3])
        
        query = (QueryBuilder()
                .where("status", Operator.EQ, "published")
                .similar_to(vector=vector, field="vec", k=8, metric="cosine")
                .sort_by("date", "desc")
                .build())
        
        assert query.vector_query is not None
        assert np.array_equal(query.vector_query.vector, vector)
        assert query.vector_query.field_name == "vec"
        assert query.vector_query.k == 8
        assert query.limit_value == 8
        assert len(query.sort_specs) == 1
    
    def test_query_builder_complex_with_vector(self):
        """Test complex boolean query with vector search."""
        vector = np.array([0.5, 0.5])
        from dataknobs_data.query import Filter
        
        builder = QueryBuilder()
        
        # Build complex query with OR logic and vector search
        query = (builder
                .where("type", Operator.EQ, "article")
                .or_(
                    Filter("priority", Operator.GT, 5),
                    Filter("featured", Operator.EQ, True)
                )
                .similar_to(vector=vector, k=20, score_threshold=0.6)
                .build())
        
        assert query.condition is not None  # Complex boolean logic
        assert query.vector_query is not None
        assert query.vector_query.score_threshold == 0.6
        assert query.limit_value == 20


class TestVectorQueryEdgeCases:
    """Test edge cases for vector query functionality."""
    
    def test_empty_vector_query(self):
        """Test handling of empty/null vector scenarios."""
        query = Query()
        assert query.vector_query is None
        
        # Clear non-existent vector
        query.clear_vector()
        assert query.vector_query is None
    
    def test_vector_query_metadata(self):
        """Test vector query metadata handling."""
        vector = np.array([0.1, 0.2])
        
        vq = VectorQuery(
            vector=vector,
            metadata={"model": "bert", "version": "1.0"}
        )
        
        assert vq.metadata["model"] == "bert"
        assert vq.metadata["version"] == "1.0"
        
        # Test serialization preserves metadata
        data = vq.to_dict()
        assert data["metadata"]["model"] == "bert"
        
        restored = VectorQuery.from_dict(data)
        assert restored.metadata["model"] == "bert"
    
    def test_multiple_vector_operations(self):
        """Test chaining multiple vector operations."""
        vector1 = np.array([0.1, 0.2])
        vector2 = np.array([0.3, 0.4])
        
        # Second similar_to should replace the first
        query = (Query()
                .similar_to(vector=vector1, k=5)
                .similar_to(vector=vector2, k=10))
        
        assert np.array_equal(query.vector_query.vector, vector2)
        assert query.vector_query.k == 10
        assert query.limit_value == 10  # Should update to match new k
    
    def test_hybrid_with_only_text(self):
        """Test hybrid query with only text (no vector)."""
        query = Query().hybrid(
            text_query="search term",
            text_field="content"
        )
        
        assert len(query.filters) == 1
        assert query.vector_query is None
    
    def test_hybrid_with_only_vector(self):
        """Test hybrid query with only vector (no text)."""
        vector = np.array([0.1, 0.2])
        
        query = Query().hybrid(
            vector=vector,
            vector_field="embedding"
        )
        
        assert len(query.filters) == 0
        assert query.vector_query is not None
        assert np.array_equal(query.vector_query.vector, vector)