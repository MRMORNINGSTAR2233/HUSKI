# GraphRAG Quick Reference

## Installation & Setup

```bash
# 1. Run setup script
./setup.sh

# 2. Edit .env with your credentials
nano .env  # or use your preferred editor

# 3. Load sample data
python examples/setup_sample_data.py

# 4. Run examples
python examples/basic_usage.py
```

## Quick Start Code

```python
from graphrag import GraphRAG

# Simple usage
with GraphRAG() as graphrag:
    response = graphrag.query("Your question here")
    print(response.answer)
```

## Query Examples

### Factual Queries (Graph)
```python
response = graphrag.query("Who directed Inception?")
```

### Multi-hop Queries (Graph)
```python
response = graphrag.query(
    "Which movie by Inception's director won an Oscar?"
)
```

### Aggregation Queries (SQL)
```python
response = graphrag.query(
    "What is the average salary of engineers?"
)
```

### Explanatory Queries (Vector)
```python
response = graphrag.query(
    "Explain Christopher Nolan's filmmaking style"
)
```

## Adding Data

### Add Documents
```python
with GraphRAG() as graphrag:
    graphrag.vector_store.add_documents(
        documents=["Text 1", "Text 2"],
        metadatas=[{"source": "doc1"}, {"source": "doc2"}]
    )
```

### Add Graph Entities
```python
with GraphRAG() as graphrag:
    graphrag.graph_store.add_entity(
        label="Person",
        properties={"name": "Jane Doe", "age": 28}
    )
```

### Add Graph Relationships
```python
with GraphRAG() as graphrag:
    graphrag.graph_store.add_relationship(
        from_label="Person",
        from_property="name",
        from_value="Jane Doe",
        to_label="Company",
        to_property="name",
        to_value="TechCorp",
        relationship_type="WORKS_FOR"
    )
```

## Response Structure

```python
response = graphrag.query("Question")

# Access answer
print(response.answer)

# Access confidence
print(response.confidence)  # 0.0 to 1.0

# Access sources
for source in response.sources:
    print(f"{source.type}: {source.content}")

# Access reasoning steps
for step in response.reasoning_steps:
    print(f"{step.step}. {step.action}: {step.rationale}")

# Get latency
print(f"{response.latency_ms}ms")
```

## Configuration

```python
from graphrag import GraphRAG, GraphRAGConfig

config = GraphRAGConfig()

# Customize settings
config.retrieval_top_k = 10
config.max_context_length = 12000
config.graph_max_hops = 5

# Disable specific features
config.enable_sql_queries = False
config.enable_graph_queries = True
config.enable_vector_search = True

graphrag = GraphRAG(config)
```

## Common Issues

### Connection Errors
```python
# Make sure databases are running:
docker ps  # Should show neo4j and postgres

# Restart if needed:
docker restart neo4j postgres-graphrag
```

### API Key Errors
```python
# Check .env file has:
OPENAI_API_KEY=sk-your-key-here
```

### Empty Results
```python
# Make sure data is loaded:
python examples/setup_sample_data.py
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=graphrag

# Run specific test
pytest tests/test_graphrag.py::TestConfig -v
```

## Docker Commands

```bash
# Start Neo4j
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest

# Start PostgreSQL
docker run -d --name postgres-graphrag \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=graphrag_db \
  postgres:latest

# Stop containers
docker stop neo4j postgres-graphrag

# Remove containers
docker rm neo4j postgres-graphrag
```

## Project Structure

```
src/graphrag/
├── graphrag.py           # Main orchestrator
├── graph_store.py        # Neo4j operations
├── vector_store.py       # ChromaDB operations
├── sql_engine.py         # SQL generation & execution
├── query_router.py       # Query classification
├── context_aggregator.py # Result merging
├── answer_generator.py   # Answer synthesis
├── config.py            # Configuration
└── models.py            # Data models
```

## Key Files

- `pyproject.toml` - Poetry configuration
- `requirements.txt` - Pip dependencies
- `.env` - Environment variables
- `setup.sh` - Quick setup script
- `examples/basic_usage.py` - Usage examples
- `examples/setup_sample_data.py` - Data loader
- `tests/test_graphrag.py` - Unit tests

## Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...

# Neo4j (if not using defaults)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# PostgreSQL (if not using defaults)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DATABASE=graphrag_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password

# Optional tuning
RETRIEVAL_TOP_K=5
GRAPH_MAX_HOPS=3
MAX_CONTEXT_LENGTH=8000
```

## Performance Tips

1. Use parallel execution for independent queries
2. Enable caching for repeated queries
3. Limit `retrieval_top_k` to avoid context overflow
4. Use specific queries instead of broad ones
5. Populate graph with well-structured data

## Next Steps

1. ✓ Setup complete - you're ready to use GraphRAG!
2. Try the examples: `python examples/basic_usage.py`
3. Load your own data
4. Build your application
5. See `GRAPHRAG_IMPLEMENTATION_PLAN.md` for architecture details
