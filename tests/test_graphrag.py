"""Basic tests for GraphRAG components."""

import pytest
from unittest.mock import Mock, patch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphrag.models import QueryIntent, QueryClassification
from graphrag.config import GraphRAGConfig


class TestConfig:
    """Test configuration management."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GraphRAGConfig()

        assert config.retrieval_top_k == 5
        assert config.graph_max_hops == 3
        assert config.max_context_length == 8000

    def test_neo4j_config(self):
        """Test Neo4j configuration."""
        config = GraphRAGConfig()

        assert config.neo4j.uri == "bolt://localhost:7687"
        assert config.neo4j.username == "neo4j"

    def test_postgres_connection_string(self):
        """Test PostgreSQL connection string generation."""
        config = GraphRAGConfig()

        conn_str = config.postgres.connection_string
        assert "postgresql://" in conn_str
        assert config.postgres.user in conn_str
        assert str(config.postgres.port) in conn_str


class TestQueryClassification:
    """Test query classification models."""

    def test_classification_model(self):
        """Test QueryClassification model."""
        classification = QueryClassification(
            primary_intent=QueryIntent.FACTUAL_LOOKUP,
            requires_graph=True,
            requires_sql=False,
            requires_vector=False,
            confidence=0.95,
        )

        assert classification.primary_intent == QueryIntent.FACTUAL_LOOKUP
        assert classification.requires_graph is True
        assert classification.confidence == 0.95

    def test_classification_with_secondary_intent(self):
        """Test classification with secondary intent."""
        classification = QueryClassification(
            primary_intent=QueryIntent.HYBRID,
            secondary_intent=QueryIntent.AGGREGATION,
            requires_graph=True,
            requires_sql=True,
            requires_vector=False,
            confidence=0.85,
        )

        assert classification.secondary_intent == QueryIntent.AGGREGATION


class TestSQLValidation:
    """Test SQL validation logic."""

    def test_forbidden_keywords(self):
        """Test detection of forbidden SQL keywords."""
        from graphrag.sql_engine import SQLEngine

        forbidden_queries = [
            "DELETE FROM users WHERE id=1",
            "DROP TABLE employees",
            "UPDATE users SET password='hacked'",
            "INSERT INTO logs VALUES ('evil')",
        ]

        for query in forbidden_queries:
            is_valid, error = SQLEngine.validate_sql(None, query)
            assert is_valid is False
            assert error is not None

    def test_allowed_select(self):
        """Test that SELECT queries are allowed."""
        from graphrag.sql_engine import SQLEngine

        valid_query = "SELECT * FROM employees WHERE role='Engineer'"
        is_valid, error = SQLEngine.validate_sql(None, valid_query)

        assert is_valid is True
        assert error is None


class TestContextAggregator:
    """Test context aggregation."""

    def test_empty_aggregation(self):
        """Test aggregation with no results."""
        from graphrag.context_aggregator import ContextAggregator

        aggregator = ContextAggregator()
        result = aggregator.aggregate()

        assert result["total_sources"] == 0
        assert result["formatted_context"] == ""

    def test_source_ranking(self):
        """Test source ranking by confidence."""
        from graphrag.context_aggregator import ContextAggregator
        from graphrag.models import Source

        aggregator = ContextAggregator()

        sources = [
            Source(type="document", content="Low confidence", confidence=0.5),
            Source(type="database", content="High confidence", confidence=1.0),
            Source(type="graph", content="Medium confidence", confidence=0.8),
        ]

        ranked = aggregator._rank_sources(sources)

        # Database source (confidence 1.0) should be first
        assert ranked[0].type == "database"
        assert ranked[0].confidence == 1.0


def test_imports():
    """Test that all modules can be imported."""
    from graphrag import GraphRAG, GraphRAGConfig
    from graphrag.graph_store import GraphStore
    from graphrag.vector_store import VectorStore
    from graphrag.sql_engine import SQLEngine
    from graphrag.query_router import QueryRouter
    from graphrag.context_aggregator import ContextAggregator
    from graphrag.answer_generator import AnswerGenerator

    assert GraphRAG is not None
    assert GraphStore is not None
    assert VectorStore is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
