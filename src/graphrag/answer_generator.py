"""Answer generator with provenance and source citations."""

import logging
from typing import List, Dict, Any
from langchain.prompts import ChatPromptTemplate

from .config import LLMConfig
from .models import Source, ReasoningStep
from .llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """Generates final answers with source citations and reasoning traces."""

    def __init__(self, llm_config: LLMConfig):
        """Initialize answer generator.

        Args:
            llm_config: LLM configuration
        """
        self.llm_config = llm_config
        self.llm = LLMFactory.create_chat_llm(llm_config)

        # Answer generation prompt
        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant with access to structured and unstructured knowledge sources.

Your task is to synthesize a precise, accurate answer using the provided context. Follow these guidelines:

1. **Prioritize Structured Data**: Database and graph results are more reliable than text documents
2. **Cite Sources**: Reference sources using [SOURCE N] tags where N is the source number
3. **Acknowledge Uncertainty**: If sources conflict or data is incomplete, state this explicitly
4. **Be Precise**: For numerical queries, provide exact values from database results
5. **Show Reasoning**: For multi-hop queries, explain the logical steps
6. **Be Concise**: Answer directly without unnecessary preamble

Context:
{context}

Guidelines for citations:
- Use [SOURCE 1], [SOURCE 2], etc. to cite specific sources
- For graph paths, explain the relationship chain
- For SQL results, mention the query type (average, count, etc.)
- For documents, note if information is contextual vs. factual

If the context doesn't contain enough information to answer the question, say so honestly."""),
            ("user", "{query}")
        ])

    def generate_answer(
        self,
        query: str,
        context: str,
        sources: List[Source],
        reasoning_steps: List[ReasoningStep],
    ) -> Dict[str, Any]:
        """Generate final answer with citations.

        Args:
            query: User's original query
            context: Formatted context from aggregator
            sources: List of sources used
            reasoning_steps: List of reasoning steps taken

        Returns:
            Dictionary with answer and metadata
        """
        try:
            # Generate answer
            messages = self.answer_prompt.format_messages(
                query=query,
                context=context,
            )
            response = self.llm.invoke(messages)

            answer = response.content.strip()

            # Extract cited sources
            cited_source_ids = self._extract_citations(answer)
            cited_sources = [
                sources[i - 1] for i in cited_source_ids if 0 < i <= len(sources)
            ]

            # Calculate confidence based on sources
            confidence = self._calculate_confidence(cited_sources, sources)

            logger.info(
                f"Generated answer with {len(cited_sources)} citations, "
                f"confidence: {confidence:.2f}"
            )

            return {
                "answer": answer,
                "cited_sources": cited_sources,
                "confidence": confidence,
                "all_sources": sources,
                "reasoning_steps": reasoning_steps,
            }

        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            raise

    def generate_with_provenance(
        self,
        query: str,
        context: str,
        sources: List[Source],
        reasoning_steps: List[ReasoningStep],
    ) -> Dict[str, Any]:
        """Generate answer with detailed provenance information.

        Args:
            query: User's original query
            context: Formatted context
            sources: List of sources
            reasoning_steps: Reasoning steps

        Returns:
            Dictionary with answer and provenance
        """
        # Generate base answer
        result = self.generate_answer(query, context, sources, reasoning_steps)

        # Add provenance details
        provenance = self._build_provenance(
            result["cited_sources"],
            reasoning_steps,
        )

        result["provenance"] = provenance

        return result

    def _extract_citations(self, answer: str) -> List[int]:
        """Extract source citation numbers from answer.

        Args:
            answer: Generated answer text

        Returns:
            List of cited source indices
        """
        import re

        # Find all [SOURCE N] citations
        citations = re.findall(r'\[SOURCE (\d+)\]', answer)
        return [int(c) for c in citations]

    def _calculate_confidence(
        self, cited_sources: List[Source], all_sources: List[Source]
    ) -> float:
        """Calculate answer confidence based on sources.

        Args:
            cited_sources: Sources cited in answer
            all_sources: All available sources

        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not cited_sources:
            return 0.5  # Medium confidence if no citations

        # Average confidence of cited sources
        avg_confidence = sum(s.confidence for s in cited_sources) / len(cited_sources)

        # Boost confidence if multiple source types agree
        source_types = set(s.type for s in cited_sources)
        type_bonus = 0.1 if len(source_types) > 1 else 0.0

        # Boost confidence for database/graph sources
        structured_sources = sum(1 for s in cited_sources if s.type in ["database", "graph"])
        structured_bonus = 0.1 if structured_sources > 0 else 0.0

        confidence = min(1.0, avg_confidence + type_bonus + structured_bonus)

        return round(confidence, 2)

    def _build_provenance(
        self,
        cited_sources: List[Source],
        reasoning_steps: List[ReasoningStep],
    ) -> Dict[str, Any]:
        """Build detailed provenance information.

        Args:
            cited_sources: Sources cited in answer
            reasoning_steps: Reasoning steps taken

        Returns:
            Provenance dictionary
        """
        provenance = {
            "source_breakdown": {},
            "reasoning_chain": [],
            "data_lineage": [],
        }

        # Source type breakdown
        for source in cited_sources:
            source_type = source.type
            if source_type not in provenance["source_breakdown"]:
                provenance["source_breakdown"][source_type] = []

            provenance["source_breakdown"][source_type].append({
                "content_preview": source.content[:100] + "...",
                "confidence": source.confidence,
                "metadata": source.metadata,
            })

        # Reasoning chain
        provenance["reasoning_chain"] = [
            {
                "step": step.step,
                "action": step.action,
                "rationale": step.rationale,
            }
            for step in reasoning_steps
        ]

        # Data lineage (where did the data come from?)
        for source in cited_sources:
            lineage_entry = {
                "source_type": source.type,
                "confidence": source.confidence,
            }

            # Add specific lineage info based on source type
            if source.type == "graph":
                lineage_entry["cypher_query"] = source.metadata.get("cypher_query", "N/A")
            elif source.type == "database":
                lineage_entry["sql_query"] = source.metadata.get("sql_query", "N/A")
            elif source.type == "document":
                lineage_entry["document_id"] = source.metadata.get("id", "N/A")

            provenance["data_lineage"].append(lineage_entry)

        return provenance

    def explain_answer(
        self,
        query: str,
        answer: str,
        reasoning_steps: List[ReasoningStep],
    ) -> str:
        """Generate natural language explanation of how the answer was derived.

        Args:
            query: Original query
            answer: Generated answer
            reasoning_steps: Steps taken to derive answer

        Returns:
            Natural language explanation
        """
        explanation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an AI assistant explaining your reasoning process.

Given a query, answer, and the steps taken to derive it, provide a clear, concise explanation of your reasoning process in 2-3 sentences.

Focus on:
1. What sources were consulted
2. What operations were performed (graph traversal, SQL aggregation, text search)
3. How the information was synthesized

Be clear and educational."""),
            ("user", """Query: {query}

Answer: {answer}

Reasoning Steps:
{steps}

Provide a brief explanation of how this answer was derived:""")
        ])

        try:
            steps_text = "\n".join([
                f"{step.step}. {step.action}: {step.rationale}"
                for step in reasoning_steps
            ])

            messages = explanation_prompt.format_messages(
                query=query,
                answer=answer,
                steps=steps_text,
            )
            response = self.llm.invoke(messages)

            return response.content.strip()

        except Exception as e:
            logger.error(f"Explanation generation failed: {e}")
            return "Unable to generate explanation."
