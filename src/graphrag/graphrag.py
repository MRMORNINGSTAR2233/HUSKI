"""Main GraphRAG orchestrator."""

import logging
import time
from typing import Dict, Any, Optional, List
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .config import GraphRAGConfig
from .models import (
    QueryIntent,
    GraphResult,
    SQLResult,
    VectorResult,
    GraphRAGResponse,
    ReasoningStep,
)
from .graph_store import GraphStore
from .vector_store import VectorStore
from .sql_engine import SQLEngine
from .query_router import QueryRouter
from .context_aggregator import ContextAggregator
from .answer_generator import AnswerGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GraphRAG:
    """Main GraphRAG system orchestrator."""

    def __init__(self, config: Optional[GraphRAGConfig] = None):
        """Initialize GraphRAG system.

        Args:
            config: GraphRAG configuration (uses defaults if None)
        """
        self.config = config or GraphRAGConfig()

        # Initialize components
        self.graph_store = GraphStore(self.config.neo4j) if self.config.enable_graph_queries else None
        self.vector_store = VectorStore(self.config.chroma, self.config.llm) if self.config.enable_vector_search else None
        self.sql_engine = SQLEngine(self.config.postgres, self.config.llm) if self.config.enable_sql_queries else None

        self.query_router = QueryRouter(self.config.llm)
        self.context_aggregator = ContextAggregator(self.config.max_context_length)
        self.answer_generator = AnswerGenerator(self.config.llm)

        self._connected = False
        self._reasoning_steps: List[ReasoningStep] = []

        logger.info("GraphRAG initialized")

    def connect(self) -> None:
        """Connect to all enabled data sources."""
        if self._connected:
            logger.warning("Already connected")
            return

        try:
            if self.graph_store:
                self.graph_store.connect()

            if self.vector_store:
                self.vector_store.connect()

            if self.sql_engine:
                self.sql_engine.connect()

            self._connected = True
            logger.info("GraphRAG connected to all data sources")

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            raise

    def close(self) -> None:
        """Close all data source connections."""
        if self.graph_store:
            self.graph_store.close()

        if self.sql_engine:
            self.sql_engine.close()

        self._connected = False
        logger.info("GraphRAG connections closed")

    def query(self, query: str, explain: bool = False) -> GraphRAGResponse:
        """Execute a query through the GraphRAG pipeline.

        Args:
            query: User query
            explain: Whether to include detailed explanation

        Returns:
            GraphRAGResponse with answer and metadata
        """
        if not self._connected:
            raise RuntimeError("Not connected. Call connect() first.")

        start_time = time.time()
        self._reasoning_steps = []

        try:
            # Step 1: Classify query
            classification = self.query_router.classify_query(query)
            self._add_reasoning_step(
                "Classify Query",
                f"Classified as {classification.primary_intent.value} with {classification.confidence:.0%} confidence"
            )

            # Step 2: Generate execution plan
            execution_plan = self.query_router.generate_execution_plan(query, classification)
            self._add_reasoning_step(
                "Generate Execution Plan",
                f"Will use strategies: {', '.join(execution_plan['strategies'])}"
            )

            # Step 3: Execute retrievals
            graph_results = []
            sql_results = []
            vector_results = []

            if execution_plan.get("parallel_execution"):
                # Execute in parallel
                graph_results, sql_results, vector_results = self._parallel_retrieval(
                    query, execution_plan
                )
            else:
                # Execute sequentially
                graph_results, sql_results, vector_results = self._sequential_retrieval(
                    query, execution_plan
                )

            # Step 4: Aggregate context
            aggregated = self.context_aggregator.aggregate(
                graph_results=graph_results if graph_results else None,
                sql_results=sql_results if sql_results else None,
                vector_results=vector_results if vector_results else None,
            )
            self._add_reasoning_step(
                "Aggregate Context",
                f"Merged {aggregated['total_sources']} sources from {sum(aggregated['source_breakdown'].values())} retrievals"
            )

            # Step 5: Generate answer
            answer_result = self.answer_generator.generate_with_provenance(
                query=query,
                context=aggregated["formatted_context"],
                sources=aggregated["sources"],
                reasoning_steps=self._reasoning_steps,
            )
            self._add_reasoning_step(
                "Generate Answer",
                f"Synthesized answer with {len(answer_result['cited_sources'])} citations"
            )

            # Calculate total latency
            latency_ms = (time.time() - start_time) * 1000

            # Build response
            response = GraphRAGResponse(
                answer=answer_result["answer"],
                sources=answer_result["cited_sources"],
                reasoning_steps=self._reasoning_steps,
                confidence=answer_result["confidence"],
                latency_ms=latency_ms,
                query_classification=classification,
            )

            # Add explanation if requested
            if explain:
                explanation = self.answer_generator.explain_answer(
                    query=query,
                    answer=response.answer,
                    reasoning_steps=response.reasoning_steps,
                )
                response.answer = f"{response.answer}\n\n**Explanation:** {explanation}"

            logger.info(f"Query completed in {latency_ms:.2f}ms")

            return response

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

    def _parallel_retrieval(
        self, query: str, execution_plan: Dict[str, Any]
    ) -> tuple[List[GraphResult], List[SQLResult], List[VectorResult]]:
        """Execute retrievals in parallel.

        Args:
            query: User query
            execution_plan: Execution plan

        Returns:
            Tuple of (graph_results, sql_results, vector_results)
        """
        graph_results = []
        sql_results = []
        vector_results = []

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []

            if "graph" in execution_plan["strategies"] and self.graph_store:
                futures.append(("graph", executor.submit(self._execute_graph_query, query)))

            if "sql" in execution_plan["strategies"] and self.sql_engine:
                futures.append(("sql", executor.submit(self._execute_sql_query, query)))

            if "vector" in execution_plan["strategies"] and self.vector_store:
                futures.append(("vector", executor.submit(self._execute_vector_search, query)))

            # Collect results
            for strategy, future in futures:
                try:
                    result = future.result(timeout=self.config.query_timeout)
                    if strategy == "graph":
                        graph_results.append(result)
                    elif strategy == "sql":
                        sql_results.append(result)
                    elif strategy == "vector":
                        vector_results.append(result)
                except Exception as e:
                    logger.error(f"{strategy} retrieval failed: {e}")

        return graph_results, sql_results, vector_results

    def _sequential_retrieval(
        self, query: str, execution_plan: Dict[str, Any]
    ) -> tuple[List[GraphResult], List[SQLResult], List[VectorResult]]:
        """Execute retrievals sequentially for multi-hop queries.

        Args:
            query: User query
            execution_plan: Execution plan

        Returns:
            Tuple of (graph_results, sql_results, vector_results)
        """
        graph_results = []
        sql_results = []
        vector_results = []

        # Handle sub-queries if present
        sub_queries = execution_plan.get("sub_queries", [{"query": query}])

        for sub_query_info in sub_queries:
            sub_query = sub_query_info.get("query", query)

            if "graph" in execution_plan["strategies"] and self.graph_store:
                result = self._execute_graph_query(sub_query)
                graph_results.append(result)

            if "sql" in execution_plan["strategies"] and self.sql_engine:
                result = self._execute_sql_query(sub_query)
                sql_results.append(result)

            if "vector" in execution_plan["strategies"] and self.vector_store:
                result = self._execute_vector_search(sub_query)
                vector_results.append(result)

        return graph_results, sql_results, vector_results

    def _execute_graph_query(self, query: str) -> GraphResult:
        """Execute graph query.

        Args:
            query: Query string

        Returns:
            GraphResult
        """
        self._add_reasoning_step("Graph Query", f"Querying knowledge graph for: {query}")

        # Simple graph query (in production, this would use LLM to generate Cypher)
        # For now, demonstrate with a basic schema query
        try:
            # This is a placeholder - real implementation would generate Cypher from query
            result = self.graph_store.get_schema()
            return GraphResult(
                entities=[result],
                relationships=[],
                cypher_query="CALL db.schema.visualization()",
                execution_time_ms=0.0,
            )
        except Exception as e:
            logger.error(f"Graph query failed: {e}")
            return GraphResult(entities=[], relationships=[], cypher_query="", execution_time_ms=0.0)

    def _execute_sql_query(self, query: str) -> SQLResult:
        """Execute SQL query.

        Args:
            query: Natural language query

        Returns:
            SQLResult
        """
        self._add_reasoning_step("SQL Query", f"Generating and executing SQL for: {query}")

        try:
            return self.sql_engine.query(query)
        except Exception as e:
            logger.error(f"SQL query failed: {e}")
            return SQLResult(data=[], sql_query="", execution_time_ms=0.0)

    def _execute_vector_search(self, query: str) -> VectorResult:
        """Execute vector search.

        Args:
            query: Search query

        Returns:
            VectorResult
        """
        self._add_reasoning_step(
            "Vector Search",
            f"Searching documents for: {query}"
        )

        try:
            return self.vector_store.search(query, top_k=self.config.retrieval_top_k)
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return VectorResult(documents=[], metadatas=[], distances=[], execution_time_ms=0.0)

    def _add_reasoning_step(self, action: str, rationale: str) -> None:
        """Add a reasoning step to the trace.

        Args:
            action: Action taken
            rationale: Rationale for action
        """
        step = ReasoningStep(
            step=len(self._reasoning_steps) + 1,
            action=action,
            rationale=rationale,
        )
        self._reasoning_steps.append(step)

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
