# GraphRAG - Graph-Enhanced Retrieval-Augmented Generation

A hybrid RAG system that combines vector search, knowledge graphs, and SQL databases for intelligent question answering with provenance.

## Features

- 🔍 **Hybrid Retrieval**: Combines vector search, graph traversal, and SQL queries
- 🧠 **Intelligent Query Routing**: LLM-powered intent classification
- 🔗 **Multi-hop Reasoning**: Graph-based relationship traversal
- 📊 **Structured Data Support**: SQL generation for numerical queries
- 🎯 **Source Citations**: Full provenance tracking for answers
- ⚡ **Parallel Execution**: Concurrent retrieval from multiple sources

## Architecture

```
User Query → Query Router → [Graph DB | Vector Store | SQL DB] → Context Aggregator → Answer Generator → Response
```

### Components

1. **Graph Store**: Neo4j for entity relationships and multi-hop queries
2. **Vector Store**: ChromaDB for semantic document search
3. **SQL Engine**: PostgreSQL with text-to-SQL generation
4. **Query Router**: LLM-based intent classification and execution planning
5. **Context Aggregator**: Multi-source result merging and ranking
6. **Answer Generator**: LLM synthesis with source citations

## Installation

### Prerequisites

- Python 3.10+
- Neo4j (for graph queries)
- PostgreSQL (for SQL queries)
- OpenAI API key

### Setup

1. **Clone the repository**
   ```bash
   cd HUSKI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   # Or with poetry:
   poetry install
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your credentials:
   # - OPENAI_API_KEY
   # - NEO4J_PASSWORD
   # - POSTGRES_PASSWORD
   ```

4. **Start databases** (using Docker)
   ```bash
   # Neo4j
   docker run -d \
     --name neo4j \
     -p 7474:7474 -p 7687:7687 \
     -e NEO4J_AUTH=neo4j/your_password \
     neo4j:latest

   # PostgreSQL
   docker run -d \
     --name postgres \
     -p 5432:5432 \
     -e POSTGRES_PASSWORD=your_password \
     -e POSTGRES_DB=graphrag_db \
     postgres:latest
   ```

5. **Populate sample data**
   ```bash
   python examples/setup_sample_data.py
   ```

## Quick Start

### Basic Usage

```python
from graphrag import GraphRAG

# Initialize with default config
with GraphRAG() as graphrag:
    # Ask a question
    response = graphrag.query("What is the average salary of engineers?")
    
    print(f"Answer: {response.answer}")
    print(f"Confidence: {response.confidence:.0%}")
    print(f"Sources: {len(response.sources)}")
```

### Query Types

#### 1. Factual Lookup (Graph Query)
```python
response = graphrag.query("Who directed Inception?")
# Uses: Graph traversal in Neo4j
```

#### 2. Multi-hop Reasoning
```python
response = graphrag.query(
    "Which movie directed by the person who directed Inception won an Oscar?"
)
# Uses: Multi-hop graph traversal
```

#### 3. Numerical Aggregation (SQL Query)
```python
response = graphrag.query("What is the average salary of engineers excluding contractors?")
# Uses: Text-to-SQL generation and execution
```

#### 4. Explanatory (Vector Search)
```python
response = graphrag.query("Explain Christopher Nolan's filmmaking style")
# Uses: Semantic search over documents
```

### Advanced Usage

#### With Detailed Explanation
```python
response = graphrag.query("Your question here", explain=True)
print(response.answer)  # Includes reasoning explanation
```

#### Custom Configuration
```python
from graphrag import GraphRAG, GraphRAGConfig

config = GraphRAGConfig()
config.retrieval_top_k = 10
config.max_context_length = 12000
config.enable_sql_queries = False  # Disable SQL

graphrag = GraphRAG(config)
```

#### Add Documents to Vector Store
```python
with GraphRAG() as graphrag:
    documents = ["Document 1 text...", "Document 2 text..."]
    metadatas = [{"source": "doc1"}, {"source": "doc2"}]
    
    graphrag.vector_store.add_documents(documents, metadatas)
```

#### Add Entities to Knowledge Graph
```python
with GraphRAG() as graphrag:
    # Add entity
    graphrag.graph_store.add_entity(
        label="Person",
        properties={"name": "John Doe", "age": 30}
    )
    
    # Add relationship
    graphrag.graph_store.add_relationship(
        from_label="Person",
        from_property="name",
        from_value="John Doe",
        to_label="Movie",
        to_property="title",
        to_value="Example Movie",
        relationship_type="ACTED_IN"
    )
```

## Project Structure

```
HUSKI/
├── src/graphrag/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── models.py              # Data models
│   ├── graphrag.py            # Main orchestrator
│   ├── graph_store.py         # Neo4j integration
│   ├── vector_store.py        # ChromaDB integration
│   ├── sql_engine.py          # PostgreSQL + text-to-SQL
│   ├── query_router.py        # Intent classification
│   ├── context_aggregator.py  # Result merging
│   └── answer_generator.py    # Answer synthesis
├── examples/
│   ├── basic_usage.py         # Usage examples
│   └── setup_sample_data.py   # Sample data loader
├── tests/
│   └── test_graphrag.py       # Unit tests
├── requirements.txt           # Dependencies
├── pyproject.toml            # Poetry config
└── README.md
```

## Configuration

All configuration is managed through environment variables or the `GraphRAGConfig` class:

### Environment Variables (.env)

```bash
# LLM
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4-turbo-preview

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DATABASE=graphrag_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password

# Application
RETRIEVAL_TOP_K=5
GRAPH_MAX_HOPS=3
MAX_CONTEXT_LENGTH=8000
```

## Testing

Run tests with pytest:

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_graphrag.py::TestConfig -v

# With coverage
pytest tests/ --cov=graphrag --cov-report=html
```

## Examples

### Example 1: Movie Database Query
```python
# Multi-hop query
response = graphrag.query(
    "What movies did the director of Interstellar also direct?"
)

# Output:
# Answer: Christopher Nolan, who directed Interstellar, also directed 
# Inception (2010), Dunkirk (2017), and The Dark Knight (2008). [SOURCE 1]
# Confidence: 95%
```

### Example 2: Employee Database Query
```python
# Aggregation query
response = graphrag.query(
    "What is the average salary of engineers in the Engineering department?"
)

# Output:
# Answer: The average salary for engineers in the Engineering department 
# is $125,000. [SOURCE 1]
# Confidence: 100%
```

### Example 3: Hybrid Query
```python
# Combines graph + vector search
response = graphrag.query(
    "Tell me about Christopher Nolan's directing style and his most successful films"
)

# Uses both graph (for film data) and vector search (for style description)
```

## Performance

Typical query latencies:
- Simple factual: 200-500ms
- Multi-hop graph: 500-1500ms
- SQL aggregation: 100-300ms
- Vector search: 300-800ms
- Hybrid: 1000-2000ms

Optimizations:
- Parallel execution for independent queries
- Result caching (configurable)
- Context compression
- Top-K limiting

## Security

### SQL Injection Prevention
- Whitelist: Only SELECT queries allowed
- Forbidden keywords: DELETE, DROP, UPDATE, INSERT, etc.
- Query validation before execution
- Parameterized queries

### Data Access
- Read-only database connections
- Environment variable isolation
- No credential logging

## Limitations

1. **Graph Construction**: Requires manual or semi-automated entity/relation extraction
2. **LLM Dependency**: Requires OpenAI API (or compatible endpoint)
3. **Database Setup**: Neo4j and PostgreSQL must be running
4. **Context Window**: Limited by LLM context length (8K-32K tokens)

## Roadmap

- [ ] Support for more embedding models (local/open-source)
- [ ] Automated entity extraction pipeline
- [ ] Query caching with Redis
- [ ] GraphQL API interface
- [ ] Web UI for visualization
- [ ] Support for more graph databases (TigerGraph, FalkorDB)
- [ ] Temporal reasoning (time-aware queries)
- [ ] Multi-modal support (images, tables)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

See [LICENSE](LICENSE) file for details.

## References

- [GraphRAG Research Paper](https://arxiv.org/abs/2404.16130)
- [Neo4j Documentation](https://neo4j.com/docs/)
- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)

## Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues and discussions
- See [GRAPHRAG_IMPLEMENTATION_PLAN.md](GRAPHRAG_IMPLEMENTATION_PLAN.md) for detailed architecture

---

**Built with:** Python, LangChain, Neo4j, ChromaDB, PostgreSQL, OpenAI
Hybrid Unstructured-Structured Knowledge Integration
