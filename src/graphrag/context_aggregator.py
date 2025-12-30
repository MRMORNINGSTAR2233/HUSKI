"""Context aggregator for merging multi-source retrieval results."""

import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict

from .models import (
    GraphResult,
    SQLResult,
    VectorResult,
    Source,
)

logger = logging.getLogger(__name__)


class ContextAggregator:
    """Aggregates and ranks results from multiple retrieval sources."""

    def __init__(self, max_context_length: int = 8000):
        """Initialize context aggregator.

        Args:
            max_context_length: Maximum context length in characters
        """
        self.max_context_length = max_context_length

    def aggregate(
        self,
        graph_results: Optional[List[GraphResult]] = None,
        sql_results: Optional[List[SQLResult]] = None,
        vector_results: Optional[List[VectorResult]] = None,
    ) -> Dict[str, Any]:
        """Aggregate results from multiple sources.

        Args:
            graph_results: Results from graph queries
            sql_results: Results from SQL queries
            vector_results: Results from vector search

        Returns:
            Aggregated context dictionary
        """
        aggregated = {
            "sources": [],
            "formatted_context": "",
            "total_sources": 0,
            "source_breakdown": {"graph": 0, "sql": 0, "vector": 0},
        }

        # Process graph results
        if graph_results:
            graph_sources = self._process_graph_results(graph_results)
            aggregated["sources"].extend(graph_sources)
            aggregated["source_breakdown"]["graph"] = len(graph_sources)

        # Process SQL results
        if sql_results:
            sql_sources = self._process_sql_results(sql_results)
            aggregated["sources"].extend(sql_sources)
            aggregated["source_breakdown"]["sql"] = len(sql_sources)

        # Process vector results
        if vector_results:
            vector_sources = self._process_vector_results(vector_results)
            aggregated["sources"].extend(vector_sources)
            aggregated["source_breakdown"]["vector"] = len(vector_sources)

        # Rank and deduplicate
        ranked_sources = self._rank_sources(aggregated["sources"])

        # Format context for LLM
        formatted_context = self._format_context(ranked_sources)

        aggregated["sources"] = ranked_sources
        aggregated["formatted_context"] = formatted_context
        aggregated["total_sources"] = len(ranked_sources)

        logger.info(
            f"Aggregated {aggregated['total_sources']} sources: "
            f"Graph={aggregated['source_breakdown']['graph']}, "
            f"SQL={aggregated['source_breakdown']['sql']}, "
            f"Vector={aggregated['source_breakdown']['vector']}"
        )

        return aggregated

    def _process_graph_results(self, results: List[GraphResult]) -> List[Source]:
        """Process graph query results into sources.

        Args:
            results: List of GraphResult objects

        Returns:
            List of Source objects
        """
        sources = []

        for result in results:
            # Format entities
            if result.entities:
                entity_content = self._format_graph_entities(result.entities)
                sources.append(
                    Source(
                        type="graph",
                        content=entity_content,
                        confidence=0.95,  # High confidence for graph facts
                        metadata={
                            "cypher_query": result.cypher_query,
                            "entity_count": len(result.entities),
                            "execution_time_ms": result.execution_time_ms,
                        },
                    )
                )

            # Format relationships
            if result.relationships:
                rel_content = self._format_graph_relationships(result.relationships)
                sources.append(
                    Source(
                        type="graph",
                        content=rel_content,
                        confidence=0.95,
                        metadata={
                            "cypher_query": result.cypher_query,
                            "relationship_count": len(result.relationships),
                        },
                    )
                )

            # Add path description if available
            if result.path_description:
                sources.append(
                    Source(
                        type="graph",
                        content=result.path_description,
                        confidence=0.90,
                        metadata={"type": "path_description"},
                    )
                )

        return sources

    def _process_sql_results(self, results: List[SQLResult]) -> List[Source]:
        """Process SQL query results into sources.

        Args:
            results: List of SQLResult objects

        Returns:
            List of Source objects
        """
        sources = []

        for result in results:
            # Format SQL result data
            content = self._format_sql_data(result)

            sources.append(
                Source(
                    type="database",
                    content=content,
                    confidence=1.0,  # Highest confidence for exact queries
                    metadata={
                        "sql_query": result.sql_query,
                        "row_count": result.row_count,
                        "columns": result.columns,
                        "execution_time_ms": result.execution_time_ms,
                    },
                )
            )

        return sources

    def _process_vector_results(self, results: List[VectorResult]) -> List[Source]:
        """Process vector search results into sources.

        Args:
            results: List of VectorResult objects

        Returns:
            List of Source objects
        """
        sources = []

        for result in results:
            for i, (doc, metadata, distance) in enumerate(
                zip(result.documents, result.metadatas, result.distances)
            ):
                # Convert distance to confidence (lower distance = higher confidence)
                confidence = max(0.0, 1.0 - distance)

                sources.append(
                    Source(
                        type="document",
                        content=doc,
                        confidence=confidence,
                        metadata={
                            **metadata,
                            "rank": i + 1,
                            "distance": distance,
                        },
                    )
                )

        return sources

    def _rank_sources(self, sources: List[Source]) -> List[Source]:
        """Rank sources by confidence and relevance.

        Args:
            sources: List of Source objects

        Returns:
            Ranked list of sources
        """
        # Sort by confidence (descending) and type priority
        type_priority = {"database": 0, "graph": 1, "document": 2}

        ranked = sorted(
            sources,
            key=lambda s: (-s.confidence, type_priority.get(s.type, 3)),
        )

        # Deduplicate similar content
        deduplicated = self._deduplicate_sources(ranked)

        logger.info(f"Ranked {len(sources)} sources, deduplicated to {len(deduplicated)}")

        return deduplicated

    def _deduplicate_sources(self, sources: List[Source]) -> List[Source]:
        """Remove duplicate or highly similar sources.

        Args:
            sources: List of Source objects

        Returns:
            Deduplicated list
        """
        # Simple deduplication by exact content match
        # TODO: Implement semantic similarity deduplication
        seen_content = set()
        deduplicated = []

        for source in sources:
            content_key = source.content.strip().lower()[:100]  # First 100 chars
            if content_key not in seen_content:
                seen_content.add(content_key)
                deduplicated.append(source)

        return deduplicated

    def _format_context(self, sources: List[Source]) -> str:
        """Format sources into context string for LLM.

        Args:
            sources: List of Source objects

        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0

        # Add sources until max context length
        for i, source in enumerate(sources, 1):
            formatted = self._format_source(source, index=i)
            formatted_length = len(formatted)

            if current_length + formatted_length > self.max_context_length:
                logger.warning(
                    f"Context length limit reached. Using {i-1} of {len(sources)} sources."
                )
                break

            context_parts.append(formatted)
            current_length += formatted_length

        return "\n\n".join(context_parts)

    def _format_source(self, source: Source, index: int) -> str:
        """Format a single source for context.

        Args:
            source: Source object
            index: Source index

        Returns:
            Formatted string
        """
        type_label = source.type.upper()
        confidence_pct = int(source.confidence * 100)

        return f"[{type_label} SOURCE {index}] (Confidence: {confidence_pct}%)\n{source.content}"

    def _format_graph_entities(self, entities: List[Dict[str, Any]]) -> str:
        """Format graph entities.

        Args:
            entities: List of entity dictionaries

        Returns:
            Formatted string
        """
        lines = ["Graph Entities:"]
        for entity in entities[:10]:  # Limit to top 10
            entity_str = ", ".join([f"{k}: {v}" for k, v in entity.items()])
            lines.append(f"  - {entity_str}")

        return "\n".join(lines)

    def _format_graph_relationships(self, relationships: List[Dict[str, Any]]) -> str:
        """Format graph relationships.

        Args:
            relationships: List of relationship dictionaries

        Returns:
            Formatted string
        """
        lines = ["Graph Relationships:"]
        for rel in relationships[:10]:  # Limit to top 10
            rel_str = ", ".join([f"{k}: {v}" for k, v in rel.items()])
            lines.append(f"  - {rel_str}")

        return "\n".join(lines)

    def _format_sql_data(self, result: SQLResult) -> str:
        """Format SQL query results.

        Args:
            result: SQLResult object

        Returns:
            Formatted string
        """
        lines = [f"SQL Query Result ({result.row_count} rows):"]
        lines.append(f"Query: {result.sql_query}")
        lines.append("")

        if result.data:
            # For aggregations (single row), format specially
            if result.row_count == 1:
                for key, value in result.data[0].items():
                    lines.append(f"{key}: {value}")
            else:
                # For multiple rows, show sample
                lines.append("Sample Data:")
                for i, row in enumerate(result.data[:5], 1):
                    row_str = ", ".join([f"{k}={v}" for k, v in row.items()])
                    lines.append(f"  {i}. {row_str}")

                if result.row_count > 5:
                    lines.append(f"  ... and {result.row_count - 5} more rows")

        return "\n".join(lines)
