"""Configuration management for GraphRAG."""

import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class Neo4jConfig(BaseModel):
    """Neo4j database configuration."""

    uri: str = Field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    username: str = Field(default_factory=lambda: os.getenv("NEO4J_USERNAME", "neo4j"))
    password: str = Field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", ""))
    database: str = Field(default_factory=lambda: os.getenv("NEO4J_DATABASE", "neo4j"))


class ChromaConfig(BaseModel):
    """ChromaDB configuration."""

    persist_directory: str = Field(
        default_factory=lambda: os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma")
    )
    collection_name: str = Field(
        default_factory=lambda: os.getenv("CHROMA_COLLECTION_NAME", "graphrag_documents")
    )


class PostgresConfig(BaseModel):
    """PostgreSQL database configuration."""

    host: str = Field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    port: int = Field(default_factory=lambda: int(os.getenv("POSTGRES_PORT", "5432")))
    database: str = Field(default_factory=lambda: os.getenv("POSTGRES_DATABASE", "graphrag_db"))
    user: str = Field(default_factory=lambda: os.getenv("POSTGRES_USER", "postgres"))
    password: str = Field(default_factory=lambda: os.getenv("POSTGRES_PASSWORD", ""))

    @property
    def connection_string(self) -> str:
        """Generate SQLAlchemy connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class LLMConfig(BaseModel):
    """LLM configuration."""

    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    model: str = Field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview"))
    embedding_model: str = Field(
        default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    )
    temperature: float = 0.0
    max_tokens: Optional[int] = None


class GraphRAGConfig(BaseModel):
    """Main GraphRAG configuration."""

    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)
    postgres: PostgresConfig = Field(default_factory=PostgresConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    # Application settings
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    max_context_length: int = Field(
        default_factory=lambda: int(os.getenv("MAX_CONTEXT_LENGTH", "8000"))
    )
    retrieval_top_k: int = Field(
        default_factory=lambda: int(os.getenv("RETRIEVAL_TOP_K", "5"))
    )
    graph_max_hops: int = Field(default_factory=lambda: int(os.getenv("GRAPH_MAX_HOPS", "3")))
    query_timeout: int = Field(default_factory=lambda: int(os.getenv("QUERY_TIMEOUT", "30")))

    # Feature flags
    enable_sql_queries: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_SQL_QUERIES", "true").lower() == "true"
    )
    enable_graph_queries: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_GRAPH_QUERIES", "true").lower() == "true"
    )
    enable_vector_search: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_VECTOR_SEARCH", "true").lower() == "true"
    )
    enable_query_caching: bool = Field(
        default_factory=lambda: os.getenv("ENABLE_QUERY_CACHING", "true").lower() == "true"
    )

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
