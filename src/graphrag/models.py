"""Data models for GraphRAG."""

from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class QueryIntent(str, Enum):
    """Types of query intents."""

    FACTUAL_LOOKUP = "factual_lookup"
    MULTI_HOP_REASONING = "multi_hop_reasoning"
    AGGREGATION = "aggregation"
    EXPLANATORY = "explanatory"
    HYBRID = "hybrid"


class RetrievalStrategy(str, Enum):
    """Retrieval strategies."""

    GRAPH = "graph"
    SQL = "sql"
    VECTOR = "vector"
    HYBRID = "hybrid"


class QueryClassification(BaseModel):
    """Query classification result."""

    primary_intent: QueryIntent
    secondary_intent: Optional[QueryIntent] = None
    requires_graph: bool = False
    requires_sql: bool = False
    requires_vector: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: Optional[str] = None


class GraphResult(BaseModel):
    """Result from graph query."""

    entities: List[Dict[str, Any]] = Field(default_factory=list)
    relationships: List[Dict[str, Any]] = Field(default_factory=list)
    cypher_query: str
    execution_time_ms: float
    path_description: Optional[str] = None


class SQLResult(BaseModel):
    """Result from SQL query."""

    data: List[Dict[str, Any]] = Field(default_factory=list)
    sql_query: str
    execution_time_ms: float
    row_count: int = 0
    columns: List[str] = Field(default_factory=list)


class VectorResult(BaseModel):
    """Result from vector search."""

    documents: List[str] = Field(default_factory=list)
    metadatas: List[Dict[str, Any]] = Field(default_factory=list)
    distances: List[float] = Field(default_factory=list)
    execution_time_ms: float


class Source(BaseModel):
    """Source citation."""

    type: str  # "graph", "database", "document"
    content: str
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReasoningStep(BaseModel):
    """Single step in reasoning trace."""

    step: int
    action: str
    rationale: str
    result: Optional[str] = None


class GraphRAGResponse(BaseModel):
    """Final response from GraphRAG."""

    answer: str
    sources: List[Source] = Field(default_factory=list)
    reasoning_steps: List[ReasoningStep] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    latency_ms: float
    query_classification: Optional[QueryClassification] = None
