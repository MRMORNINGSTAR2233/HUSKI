"""Query router with LLM-based intent classification."""

import logging
import json
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from .config import LLMConfig
from .models import QueryIntent, QueryClassification

logger = logging.getLogger(__name__)


class QueryRouter:
    """Routes queries to appropriate retrieval strategies based on intent."""

    def __init__(self, llm_config: LLMConfig):
        """Initialize query router.

        Args:
            llm_config: LLM configuration
        """
        self.llm_config = llm_config
        self.llm = ChatOpenAI(
            api_key=llm_config.api_key,
            model=llm_config.model,
            temperature=0.0,  # Deterministic classification
        )

        # Classification prompt
        self.classification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query classification expert. Analyze the user's query and classify it into one or more categories.

Categories:
- FACTUAL_LOOKUP: Simple entity retrieval (e.g., "Who directed Inception?")
- MULTI_HOP_REASONING: Requires traversing relationships (e.g., "Which movie by Inception's director won an Oscar?")
- AGGREGATION: Numerical/statistical computation (e.g., "Average salary of engineers")
- EXPLANATORY: Requires textual context and explanation (e.g., "Explain how photosynthesis works")
- HYBRID: Combination of multiple types

For each query, determine:
1. Primary intent (most important)
2. Secondary intent (if applicable)
3. Whether it requires graph queries (relationship traversal)
4. Whether it requires SQL queries (aggregations, filtering)
5. Whether it requires vector search (textual context)

Respond with ONLY valid JSON matching this schema:
{{
    "primary_intent": "FACTUAL_LOOKUP" | "MULTI_HOP_REASONING" | "AGGREGATION" | "EXPLANATORY" | "HYBRID",
    "secondary_intent": null | "FACTUAL_LOOKUP" | "MULTI_HOP_REASONING" | "AGGREGATION" | "EXPLANATORY",
    "requires_graph": true | false,
    "requires_sql": true | false,
    "requires_vector": true | false,
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of classification"
}}"""),
            ("user", "{query}")
        ])

    def classify_query(self, query: str) -> QueryClassification:
        """Classify query intent.

        Args:
            query: User query

        Returns:
            QueryClassification result
        """
        try:
            # Generate classification
            messages = self.classification_prompt.format_messages(query=query)
            response = self.llm.invoke(messages)

            # Parse JSON response
            result = json.loads(response.content)

            # Convert to QueryClassification model
            classification = QueryClassification(
                primary_intent=QueryIntent(result["primary_intent"].lower()),
                secondary_intent=QueryIntent(result["secondary_intent"].lower())
                if result.get("secondary_intent")
                else None,
                requires_graph=result["requires_graph"],
                requires_sql=result["requires_sql"],
                requires_vector=result["requires_vector"],
                confidence=result["confidence"],
                reasoning=result.get("reasoning"),
            )

            logger.info(
                f"Query classified as {classification.primary_intent} "
                f"(confidence: {classification.confidence:.2f})"
            )

            return classification

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse classification response: {e}")
            # Fallback to hybrid with low confidence
            return QueryClassification(
                primary_intent=QueryIntent.HYBRID,
                requires_graph=True,
                requires_sql=True,
                requires_vector=True,
                confidence=0.3,
                reasoning="Failed to classify, defaulting to hybrid",
            )
        except Exception as e:
            logger.error(f"Query classification failed: {e}")
            raise

    def decompose_query(self, query: str, classification: QueryClassification) -> Dict[str, Any]:
        """Decompose complex query into sub-queries.

        Args:
            query: User query
            classification: Query classification

        Returns:
            Dictionary with sub-queries and execution plan
        """
        # Only decompose multi-hop or hybrid queries
        if classification.primary_intent not in [
            QueryIntent.MULTI_HOP_REASONING,
            QueryIntent.HYBRID,
        ]:
            return {
                "sub_queries": [{"id": 1, "query": query, "type": classification.primary_intent}],
                "execution_order": [1],
            }

        decomposition_prompt = ChatPromptTemplate.from_messages([
            ("system", """Break down complex queries into logical sub-queries that can be executed sequentially.

For multi-hop queries, identify:
1. Initial entity lookup
2. Relationship traversals
3. Final filtering/aggregation

Respond with ONLY valid JSON:
{{
    "sub_queries": [
        {{"id": 1, "query": "First sub-query", "type": "factual_lookup"}},
        {{"id": 2, "query": "Second sub-query (depends on 1)", "type": "multi_hop_reasoning"}}
    ],
    "execution_order": [1, 2],
    "dependencies": {{"2": [1]}}
}}"""),
            ("user", "Query: {query}\nClassification: {classification}")
        ])

        try:
            messages = decomposition_prompt.format_messages(
                query=query,
                classification=classification.primary_intent.value,
            )
            response = self.llm.invoke(messages)

            result = json.loads(response.content)
            logger.info(f"Query decomposed into {len(result['sub_queries'])} sub-queries")

            return result

        except Exception as e:
            logger.error(f"Query decomposition failed: {e}")
            # Fallback to single query
            return {
                "sub_queries": [{"id": 1, "query": query, "type": classification.primary_intent.value}],
                "execution_order": [1],
            }

    def generate_execution_plan(
        self, query: str, classification: QueryClassification
    ) -> Dict[str, Any]:
        """Generate execution plan for query.

        Args:
            query: User query
            classification: Query classification

        Returns:
            Execution plan with strategies and order
        """
        plan = {
            "query": query,
            "classification": classification.dict(),
            "strategies": [],
            "parallel_execution": False,
        }

        # Determine which retrieval strategies to use
        if classification.requires_graph:
            plan["strategies"].append("graph")

        if classification.requires_sql:
            plan["strategies"].append("sql")

        if classification.requires_vector:
            plan["strategies"].append("vector")

        # For simple queries, enable parallel execution
        if classification.primary_intent in [QueryIntent.FACTUAL_LOOKUP, QueryIntent.EXPLANATORY]:
            plan["parallel_execution"] = True

        # Decompose if needed
        if classification.primary_intent in [QueryIntent.MULTI_HOP_REASONING, QueryIntent.HYBRID]:
            decomposition = self.decompose_query(query, classification)
            plan["sub_queries"] = decomposition["sub_queries"]
            plan["execution_order"] = decomposition["execution_order"]

        logger.info(f"Execution plan: {len(plan['strategies'])} strategies, parallel={plan['parallel_execution']}")

        return plan
