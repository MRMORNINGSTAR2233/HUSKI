# Graph-Enhanced RAG (GraphRAG) Implementation Plan

## Executive Summary

This document outlines the ideation and architectural design for building a hybrid GraphRAG system that addresses the limitations of traditional RAG systems when handling structured data, numerical queries, and multi-hop reasoning chains.

---

## 1. Problem Analysis

### Current RAG Limitations
- **Structured Data Blindness**: Cannot perform SQL-like aggregations (averages, sums, filters)
- **Multi-hop Reasoning Failures**: Struggles with relationship traversals (e.g., "Who directed the movie that won Best Picture in 2010?")
- **Numerical Precision**: Produces approximations instead of exact calculations
- **Provenance Gaps**: Cannot show clear reasoning paths
- **Context Window Waste**: Retrieves verbose text when precise data points suffice

### Target Capabilities
- Hybrid retrieval: text chunks + structured queries
- Intelligent query routing (text vs. graph vs. SQL)
- Multi-hop graph traversals
- Exact numerical computations
- Explainable reasoning traces

---

## 2. System Architecture Overview

### High-Level Components

```
┌─────────────────────────────────────────────────────────────┐
│                      User Query                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           Query Analyzer & Router (LLM-based)               │
│  - Intent Classification                                     │
│  - Query Decomposition                                       │
│  - Execution Plan Generation                                 │
└──────┬──────────────┬──────────────┬────────────────────────┘
       │              │              │
       ▼              ▼              ▼
┌──────────┐   ┌──────────┐   ┌──────────────┐
│  Vector  │   │  Graph   │   │   Database   │
│ Retriever│   │ Traverser│   │ Query Engine │
│ (Text)   │   │  (Neo4j) │   │  (SQL/API)   │
└─────┬────┘   └─────┬────┘   └──────┬───────┘
      │              │               │
      └──────────────┼───────────────┘
                     ▼
        ┌────────────────────────┐
        │  Context Aggregator    │
        │  - Merge Results       │
        │  - Rank & Filter       │
        │  - Format for LLM      │
        └───────────┬────────────┘
                    ▼
        ┌────────────────────────┐
        │   Answer Generator     │
        │   (LLM with Context)   │
        └───────────┬────────────┘
                    ▼
        ┌────────────────────────┐
        │  Response + Provenance │
        └────────────────────────┘
```

---

## 3. Core Components Design

### 3.1 Query Analyzer & Router

**Purpose**: Intelligent query understanding and execution planning

**Mechanisms**:
1. **Intent Classification**
   - Use few-shot LLM prompting to categorize queries:
     - `FACTUAL_LOOKUP`: Simple entity retrieval
     - `MULTI_HOP_REASONING`: Relationship traversal needed
     - `AGGREGATION`: Numerical/statistical computation
     - `EXPLANATORY`: Requires textual context
     - `HYBRID`: Combination of above

2. **Query Decomposition**
   - Break complex queries into sub-queries
   - Example: "Average salary of engineers in SF who joined after 2020"
     - Sub-query 1: Filter engineers in SF
     - Sub-query 2: Filter join date > 2020
     - Sub-query 3: Compute average salary

3. **Execution Plan Generation**
   - For each sub-query, determine:
     - Retrieval strategy (vector/graph/SQL)
     - Execution order (DAG for dependencies)
     - Result aggregation method

**Implementation Approach**:
- Use structured output from LLM (JSON schema)
- Maintain a taxonomy of query patterns
- Fine-tune a smaller model for classification (optional optimization)

---

### 3.2 Knowledge Graph Construction

**Graph Schema Design**:

```
Entities:
- Person (attributes: name, birth_date, nationality)
- Movie (attributes: title, release_year, budget, revenue)
- Company (attributes: name, founded_year, industry)
- Department (attributes: name, location)
- Employee (attributes: name, salary, hire_date, role)

Relations:
- DIRECTED (Person → Movie)
- ACTED_IN (Person → Movie)
- WON_AWARD (Movie → Award, Person → Award)
- WORKS_IN (Employee → Department)
- MANAGES (Employee → Department)
- COLLABORATES_WITH (Person → Person)
```

**Graph Construction Pipeline**:

1. **Entity Extraction**
   - Use NER models (spaCy, Stanza) on corpus
   - LLM-based extraction for complex entities
   - Entity resolution & deduplication (fuzzy matching)

2. **Relation Extraction**
   - Dependency parsing for syntactic relations
   - LLM prompting for semantic relations
   - Relation classification models (BERT-based)

3. **Graph Population**
   - Batch insert into Neo4j/TigerGraph
   - Create indexes on frequently queried properties
   - Implement entity linking to external KGs (Wikidata, DBpedia)

**Storage Considerations**:
- **Neo4j**: Better for complex traversals, Cypher query language
- **TigerGraph**: Handles larger graphs, better scalability
- **FalkorDB**: Redis-based, faster in-memory operations

---

### 3.3 Vector Store for Unstructured Text

**Purpose**: Traditional RAG retrieval for explanatory content

**Implementation**:
- **Embedding Model**: `text-embedding-3-large` or `sentence-transformers/all-mpnet-base-v2`
- **Vector DB**: Pinecone, Weaviate, or ChromaDB
- **Chunking Strategy**:
  - Semantic chunking (split by paragraphs/topics)
  - Chunk size: 512-1024 tokens with 128-token overlap
  - Store metadata: source, entity mentions, chunk type

**Hybrid Search**:
- Combine dense (vector) + sparse (BM25) retrieval
- Re-ranking with cross-encoder models

---

### 3.4 Database Query Engine

**Purpose**: Execute SQL for tabular/numerical data

**Mechanisms**:
1. **Text-to-SQL Generation**
   - Use LLM with schema context to generate SQL
   - Validate SQL syntax before execution
   - Implement query sanitization (prevent injection)

2. **API Wrapper for External DBs**
   - Connect to PostgreSQL, MySQL, BigQuery
   - Abstract queries through ORM (SQLAlchemy)

3. **Result Formatting**
   - Convert SQL results to natural language
   - Generate summary statistics
   - Create visualizations (optional)

---

### 3.5 Context Aggregator

**Purpose**: Merge results from multiple retrieval sources

**Strategies**:

1. **Result Ranking**
   - Score each result by relevance, recency, source authority
   - Use learned-to-rank models (LambdaMART)

2. **Deduplication**
   - Detect overlapping information from different sources
   - Keep highest-quality version

3. **Context Compression**
   - Summarize verbose chunks
   - Use LLMLingua or similar techniques
   - Keep only query-relevant portions

4. **Formatting for LLM**
   - Structure as: [Graph Facts] + [SQL Results] + [Text Context]
   - Add source citations
   - Include confidence scores

---

### 3.6 Answer Generator

**Purpose**: Synthesize final response from aggregated context

**Prompt Structure**:
```
System: You are an AI assistant with access to structured and unstructured knowledge.

Context:
[Graph Entities & Relations]
- Christopher Nolan DIRECTED Inception (2010)
- Inception WON_AWARD Best Picture (False), Best Cinematography (True)

[Database Results]
- Query: SELECT AVG(salary) FROM employees WHERE role='Engineer' AND contractor=False
- Result: $125,000

[Text Chunks]
1. [Relevance: 0.92] "Engineers at the company typically earn..."
2. [Relevance: 0.87] "Contractor salaries are excluded from..."

Question: {user_query}

Instructions:
- Synthesize a precise answer using the structured data
- Cite sources using [Graph], [DB], or [Doc] tags
- If data conflicts, acknowledge uncertainty
- Explain reasoning steps for complex queries
```

---

## 4. Data Flow Example

### Query: "What's the average salary of engineers in SF excluding contractors?"

**Step 1: Query Analysis**
- Intent: `AGGREGATION` + `FACTUAL_LOOKUP`
- Decomposition:
  - Filter: role = 'Engineer'
  - Filter: location = 'SF'
  - Filter: contractor = False
  - Compute: AVG(salary)

**Step 2: Execution**
- Router → Database Query Engine
- Generated SQL:
  ```sql
  SELECT AVG(salary) 
  FROM employees 
  WHERE role = 'Engineer' 
    AND location = 'SF' 
    AND contractor = False
  ```

**Step 3: Parallel Text Retrieval** (optional)
- Vector search: "engineer salaries san francisco"
- Retrieve context about pay scales, cost of living

**Step 4: Context Aggregation**
- Primary: SQL result = $135,000
- Supporting: Text chunks about SF tech salaries

**Step 5: Answer Generation**
- "The average salary for engineers in San Francisco, excluding contractors, is $135,000. [DB Query]"
- "This aligns with industry reports showing SF engineers earn 20% above national average. [Doc:salary_report_2024.pdf]"

---

## 5. Multi-Hop Reasoning Example

### Query: "Which movie directed by the person who directed Inception won an Oscar?"

**Step 1: Query Analysis**
- Intent: `MULTI_HOP_REASONING`
- Decomposition:
  - Sub-query 1: Who directed Inception?
  - Sub-query 2: What movies did that person direct?
  - Sub-query 3: Which of those movies won an Oscar?

**Step 2: Graph Traversal**
Cypher Query:
```cypher
MATCH (p:Person)-[:DIRECTED]->(m1:Movie {title: 'Inception'})
MATCH (p)-[:DIRECTED]->(m2:Movie)
MATCH (m2)-[:WON_AWARD]->(a:Award {category: 'Oscar'})
RETURN p.name, m2.title, a.name
```

**Step 3: Result**
- Christopher Nolan directed Inception
- He also directed: Dunkirk, Interstellar, The Dark Knight, etc.
- Dunkirk won 3 Oscars

**Step 4: Answer with Provenance**
- "Christopher Nolan, who directed Inception, also directed Dunkirk, which won 3 Academy Awards. [Graph Path: Inception → directed_by → Christopher Nolan → directed → Dunkirk → won_award → Oscar]"

---

## 6. Technical Stack Recommendations

### Core Infrastructure
- **Language**: Python 3.10+
- **LLM Framework**: LangChain or LlamaIndex
- **Graph Database**: Neo4j (Community Edition to start)
- **Vector Database**: ChromaDB (local) → Pinecone (production)
- **Relational DB**: PostgreSQL

### Key Libraries
```
Knowledge Graph:
- neo4j-driver
- py2neo
- rdflib (for RDF/OWL ontologies)

Entity/Relation Extraction:
- spaCy (NER)
- transformers (BERT for relation extraction)
- rebel-relation-extraction (pre-trained)

Vector Embeddings:
- sentence-transformers
- openai (for embeddings)

LLM Integration:
- langchain
- llama-index
- guidance (for structured generation)

Text-to-SQL:
- sqlalchemy
- langchain SQL agents
- vanna.ai (specialized text-to-SQL)

Orchestration:
- prefect or airflow (for pipelines)
- celery (for async tasks)
```

### Development Tools
- **Testing**: pytest with graph fixtures
- **Monitoring**: LangSmith or Phoenix for LLM tracing
- **Visualization**: Graphviz, Neo4j Browser
- **Containerization**: Docker for reproducibility

---

## 7. Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
**Goal**: Basic hybrid retrieval working

- [ ] Set up Neo4j instance
- [ ] Create sample knowledge graph (movies/directors dataset)
- [ ] Implement vector store with sample documents
- [ ] Build simple query router (rule-based)
- [ ] Test single-hop graph queries

**Deliverable**: Demo answering "Who directed Inception?" using graph + text

---

### Phase 2: Query Intelligence (Weeks 3-4)
**Goal**: LLM-powered query understanding

- [ ] Implement LLM-based query classifier
- [ ] Build query decomposition system
- [ ] Create execution planner (DAG generation)
- [ ] Test on multi-hop questions

**Deliverable**: Successfully answer 2-hop queries like "What movies did Inception's director also direct?"

---

### Phase 3: Database Integration (Weeks 5-6)
**Goal**: Handle numerical/tabular queries

- [ ] Set up PostgreSQL with sample employee data
- [ ] Implement text-to-SQL generator
- [ ] Add SQL result formatter
- [ ] Build safety validators

**Deliverable**: Answer "Average salary of engineers" queries

---

### Phase 4: Context Aggregation (Weeks 7-8)
**Goal**: Intelligent result merging

- [ ] Implement ranking algorithms
- [ ] Build deduplication system
- [ ] Add context compression
- [ ] Create citation/provenance system

**Deliverable**: Answers include sources from all 3 retrieval types

---

### Phase 5: Graph Construction Pipeline (Weeks 9-11)
**Goal**: Automate knowledge graph building

- [ ] Build entity extraction pipeline
- [ ] Implement relation extraction
- [ ] Create entity resolution system
- [ ] Add incremental graph updates

**Deliverable**: Ingest a corpus and auto-generate graph

---

### Phase 6: Optimization & Evaluation (Weeks 12-14)
**Goal**: Production-ready system

- [ ] Benchmark query latency
- [ ] Implement caching strategies
- [ ] Add batch processing
- [ ] Create evaluation suite (accuracy, latency)
- [ ] Fine-tune retrieval thresholds

**Deliverable**: System handles 100+ QPS with <2s latency

---

### Phase 7: Advanced Features (Weeks 15-16)
**Goal**: Enhanced capabilities

- [ ] Temporal reasoning (time-aware queries)
- [ ] Uncertainty quantification
- [ ] Interactive clarification
- [ ] Multi-modal support (images, tables)

---

## 8. Datasets for Development & Testing

### Knowledge Graph Datasets
1. **Freebase** (via Wikidata dumps)
   - 50M+ entities, 3B+ facts
   - Good for general knowledge

2. **DBpedia**
   - Structured Wikipedia data
   - Strong entity coverage

3. **YAGO**
   - High-precision knowledge graph
   - Good for factual QA

### QA Benchmarks
1. **HotpotQA**
   - Multi-hop reasoning questions
   - 113k Wikipedia-based Q&A pairs

2. **ComplexWebQuestions**
   - Questions requiring multi-hop over Freebase
   - 34k examples

3. **WikiTableQuestions**
   - Questions over Wikipedia tables
   - 22k examples for tabular reasoning

4. **Spider**
   - Text-to-SQL benchmark
   - 10k questions over 200 databases

### Custom Domain Data
- Create synthetic datasets for specific domains (HR, finance, etc.)
- Use GPT-4 to generate graph-compatible QA pairs

---

## 9. Evaluation Metrics

### Accuracy Metrics
- **Exact Match (EM)**: For factual queries
- **F1 Score**: For partial answers
- **Graph Path Accuracy**: % of correct reasoning traces
- **SQL Query Correctness**: Validated against ground truth

### Efficiency Metrics
- **Query Latency**: End-to-end response time
- **Token Efficiency**: Context tokens used vs. baseline RAG
- **Retrieval Precision@K**: Relevance of top K results

### Explainability Metrics
- **Provenance Coverage**: % of answers with sources
- **Reasoning Trace Validity**: Human-evaluated coherence
- **User Trust Score**: Survey-based metric

---

## 10. Key Challenges & Mitigation Strategies

### Challenge 1: Graph Construction at Scale
**Problem**: Extracting accurate entities/relations from large corpora is expensive and error-prone

**Mitigations**:
- Start with high-precision, low-recall extraction
- Use active learning to improve over time
- Leverage pre-trained entity linkers (SpaCy + Wikipedia)
- Implement human-in-the-loop validation for critical domains

---

### Challenge 2: Query Intent Ambiguity
**Problem**: Same query could be interpreted multiple ways

**Example**: "Show me Python developers"
- Graph query (people with Python skill)?
- Text search (documents about Python development)?
- Database query (employees table filtering)?

**Mitigations**:
- Use clarification dialogs when confidence < threshold
- Maintain user interaction history for context
- Multi-strategy execution (run all, rank results)

---

### Challenge 3: Graph Incompleteness
**Problem**: Not all facts are in the graph

**Mitigations**:
- Fallback to text retrieval when graph returns null
- Confidence scoring on graph completeness
- Continuous graph enrichment from new documents

---

### Challenge 4: SQL Generation Errors
**Problem**: LLMs generate incorrect or unsafe SQL

**Mitigations**:
- Whitelist allowed operations (SELECT only, no DELETE/DROP)
- Validate queries with SQL parser before execution
- Use read-only database replicas
- Maintain query cache for common patterns

---

### Challenge 5: Context Window Limits
**Problem**: Large graph subgraphs + text chunks exceed token limits

**Mitigations**:
- Summarize graph paths (e.g., "3-hop path via 5 intermediate nodes")
- Use iterative refinement (broad search → narrow focused search)
- Implement dynamic context allocation based on query complexity

---

## 11. Advanced Optimizations

### 11.1 Caching Strategy
- **Query-Level**: Cache common questions → answers
- **Retrieval-Level**: Cache graph traversals (TTL: 1 hour)
- **Embedding-Level**: Cache document embeddings (persistent)

### 11.2 Parallelization
- Simultaneously query graph + vector store + database
- Use async/await for non-blocking I/O
- Batch graph queries when possible

### 11.3 Adaptive Retrieval
- Learn which retrieval strategy works best for query types
- Use reinforcement learning to optimize router decisions
- A/B test different hybrid combinations

### 11.4 Graph Compression
- Store only high-confidence edges
- Prune low-utility nodes periodically
- Use graph summarization techniques

---

## 12. Explainability & Provenance

### Reasoning Trace Format
```json
{
  "query": "Average salary of SF engineers?",
  "reasoning_steps": [
    {
      "step": 1,
      "action": "Route to SQL engine",
      "rationale": "Detected aggregation intent"
    },
    {
      "step": 2,
      "action": "Generate SQL query",
      "sql": "SELECT AVG(salary) FROM employees WHERE...",
      "result": 135000
    },
    {
      "step": 3,
      "action": "Retrieve supporting text",
      "chunks": ["doc1.pdf:para3", "doc2.html:section2"]
    }
  ],
  "sources": [
    {"type": "database", "table": "employees", "confidence": 1.0},
    {"type": "document", "id": "doc1.pdf", "confidence": 0.89}
  ]
}
```

### Visualization
- Graph paths: Render as interactive diagrams
- Query execution: Flowchart of routing decisions
- Source attribution: Inline citations in answer

---

## 13. Security & Privacy Considerations

### Data Access Control
- Role-based access to graph nodes/edges
- Query filtering based on user permissions
- Encrypt sensitive entity attributes

### SQL Injection Prevention
- Parameterized queries only
- Whitelist table/column names
- Rate limiting on query generation

### PII Protection
- Anonymize entities in training data
- Redact sensitive information in responses
- Audit logging for compliance

---

## 14. Integration Patterns

### REST API Design
```
POST /query
{
  "question": "Who directed Inception?",
  "mode": "auto",  // or "graph_only", "text_only", "hybrid"
  "include_provenance": true,
  "max_hops": 3
}

Response:
{
  "answer": "Christopher Nolan directed Inception.",
  "sources": [...],
  "reasoning_trace": {...},
  "confidence": 0.98,
  "latency_ms": 450
}
```

### Streaming Responses
- Use Server-Sent Events (SSE) for real-time updates
- Stream partial results as they arrive (graph → DB → text)

### Webhook Support
- Async processing for complex queries
- Callback URL for completion notification

---

## 15. Future Enhancements

### 15.1 Temporal Knowledge Graphs
- Track entity/relation changes over time
- Answer "What was X's salary in 2020?" queries

### 15.2 Multi-Modal Knowledge
- Incorporate images, videos into graph
- Cross-reference text ↔ visual entities

### 15.3 Federated Graphs
- Query across multiple organization silos
- Privacy-preserving graph federation

### 15.4 Active Learning Loop
- User feedback improves query routing
- Correct entity extraction errors via interaction

### 15.5 Neural Graph Reasoning
- Graph Neural Networks for implicit relation inference
- Link prediction to fill knowledge gaps

---

## 16. Success Criteria

### Minimum Viable Product (MVP)
- [ ] Answer 80% of single-hop factual questions correctly
- [ ] Handle 50% of multi-hop queries (2-3 hops)
- [ ] Execute basic SQL aggregations with 90% accuracy
- [ ] Response latency < 3 seconds for 95th percentile
- [ ] Provide provenance for all answers

### Production Readiness
- [ ] 95% accuracy on domain-specific benchmark
- [ ] Sub-second latency for cached queries
- [ ] Handle 1000 QPS with horizontal scaling
- [ ] Explainable reasoning for 100% of responses
- [ ] Zero SQL injection incidents

---

## 17. Cost Analysis

### Infrastructure Costs (Monthly Estimate)
- **Neo4j AuraDB Professional**: $200-500 (depends on graph size)
- **Pinecone**: $70-200 (1M-5M vectors)
- **PostgreSQL (managed)**: $50-150
- **Compute (API servers)**: $200-400
- **LLM API calls**: $500-2000 (depends on query volume)

**Total**: ~$1,020-$3,250/month for moderate scale

### Optimization Opportunities
- Self-host Neo4j Community Edition: Save $200/mo
- Use ChromaDB locally: Save $70/mo
- Cache aggressively: Reduce LLM calls by 50%
- Batch processing: Reduce API costs

---

## 18. Risk Register

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Graph extraction accuracy < 70% | High | Medium | Human validation loop, start with high-precision sources |
| LLM hallucination in SQL generation | High | Medium | Validation layer, query whitelisting |
| Query latency > 5s | Medium | High | Caching, parallel retrieval, index optimization |
| Graph incompleteness breaks queries | Medium | High | Fallback to text retrieval, null-handling |
| Privacy leak via provenance | High | Low | Access controls, PII redaction |

---

## 19. Development Workflow

### Local Development
1. Docker Compose for Neo4j + PostgreSQL + Vector DB
2. Sample datasets (1000 entities, 5000 relations)
3. Jupyter notebooks for experimentation
4. Unit tests for each component

### CI/CD Pipeline
1. **Testing**: 
   - Unit tests (80% coverage)
   - Integration tests (end-to-end queries)
   - Regression tests (benchmark accuracy)

2. **Deployment**:
   - Staging: Auto-deploy on merge to `develop`
   - Production: Manual approval for `main`
   - Blue-green deployment for zero downtime

3. **Monitoring**:
   - Query latency dashboards
   - Accuracy metrics tracking
   - Error rate alerts

---

## 20. Team Structure (Recommended)

### Phase 1-2 (Prototype)
- 1 ML Engineer (LLM integration, retrieval)
- 1 Backend Engineer (APIs, database)
- 1 Data Scientist (evaluation, benchmarks)

### Phase 3-7 (Production)
- +1 Graph Database Specialist
- +1 DevOps Engineer
- +1 QA Engineer
- +0.5 UX Researcher (for explainability studies)

---

## 21. Conclusion

This GraphRAG implementation addresses fundamental limitations of text-only retrieval by:

1. **Precision**: Structured queries for exact answers
2. **Reasoning**: Graph traversals for multi-hop logic
3. **Scalability**: Modular architecture for growth
4. **Explainability**: Provenance for every claim
5. **Flexibility**: Hybrid retrieval adapts to query type

The phased approach allows incremental value delivery while managing technical risk. Starting with a constrained domain (e.g., movie database) provides fast validation before scaling to enterprise knowledge graphs.

**Next Step**: Choose a pilot use case and begin Phase 1 implementation with sample dataset.

---

## Appendix A: Sample Cypher Queries

### Find All Movies by Director
```cypher
MATCH (p:Person {name: 'Christopher Nolan'})-[:DIRECTED]->(m:Movie)
RETURN m.title, m.year
ORDER BY m.year DESC
```

### Multi-Hop: Co-actors in Award-Winning Films
```cypher
MATCH (actor:Person)-[:ACTED_IN]->(movie:Movie)<-[:ACTED_IN]-(coactor:Person)
WHERE (movie)-[:WON_AWARD]->(:Award {category: 'Best Picture'})
AND actor.name = 'Leonardo DiCaprio'
RETURN DISTINCT coactor.name, movie.title
```

### Aggregate: Average Budget by Genre
```cypher
MATCH (m:Movie)-[:HAS_GENRE]->(g:Genre)
RETURN g.name, AVG(m.budget) as avg_budget
ORDER BY avg_budget DESC
```

---

## Appendix B: Example Prompts

### Query Classification Prompt
```
Classify the following query into one or more categories:
[FACTUAL_LOOKUP, MULTI_HOP, AGGREGATION, EXPLANATORY]

Query: "What's the average salary of engineers hired after 2020?"

Think step by step:
1. Does it require a calculation? (AGGREGATION - yes)
2. Does it require graph traversal? (no)
3. Does it need textual explanation? (optional)

Output JSON:
{
  "primary_intent": "AGGREGATION",
  "secondary_intent": null,
  "requires_graph": false,
  "requires_sql": true,
  "requires_text": false
}
```

### Text-to-SQL Prompt
```
Database Schema:
Table: employees
- id (INT)
- name (VARCHAR)
- role (VARCHAR)
- salary (DECIMAL)
- hire_date (DATE)
- is_contractor (BOOLEAN)

Generate a SQL query for: "Average salary of engineers hired after 2020"

Constraints:
- Read-only (SELECT only)
- Filter out contractors
- Return a single aggregate value

SQL:
```

---

## Appendix C: References

### Academic Papers
1. "Graph-based RAG: Enhancing LLMs with Structured Knowledge" (2024)
2. "Neural Graph Reasoning over Knowledge Bases" (2023)
3. "Text-to-SQL in the Wild: A Naturally-Occurring Dataset" (2023)

### Technical Resources
- Neo4j GraphAcademy: graph.academy
- LlamaIndex GraphRAG Guide: docs.llamaindex.ai
- LangChain Graph QA: python.langchain.com/docs/use_cases/graph
- FalkorDB Documentation: docs.falkordb.com

### Tools & Libraries
- Rebel (Relation Extraction): huggingface.co/Babelscape/rebel-large
- Vanna.ai (Text-to-SQL): vanna.ai
- LangSmith (LLM Monitoring): smith.langchain.com

---

**Document Version**: 1.0  
**Last Updated**: December 30, 2025  
**Author**: GraphRAG Implementation Team  
**Status**: Ideation Complete - Ready for Phase 1
