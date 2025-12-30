# Multi-Provider LLM Support

GraphRAG now supports **three LLM providers**: OpenAI, Google Gemini, and Groq.

## Overview

| Provider | Best For | Context Window | Speed | Cost |
|----------|----------|----------------|-------|------|
| **OpenAI** | Production reliability, advanced reasoning | 128K tokens | Normal | $$$ |
| **Google Gemini** | Long documents, multimodal | 2M tokens | Normal | $$ |
| **Groq** | Low latency, high throughput | 131K tokens | ⚡ 280+ tps | $ |

## Available Models

### OpenAI
- `gpt-4-turbo-preview` (128K context)
- `gpt-4` (8K context)
- `gpt-3.5-turbo` (16K context)
- Embeddings: `text-embedding-3-large`, `text-embedding-3-small`

### Google Gemini
- `gemini-3-pro` (2M context, latest)
- `gemini-2.5-pro` (2M context, advanced reasoning)
- `gemini-2.5-flash` (1M context, fast)
- `gemini-3-flash` (1M context, fastest)
- Embeddings: `models/text-embedding-004`

### Groq
- `llama-3.3-70b-versatile` (131K context, 280 tps)
- `llama-3.1-8b-instant` (131K context, 1000+ tps)
- `openai/gpt-oss-120b` (131K context, 500 tps)
- Embeddings: Falls back to OpenAI (Groq doesn't provide embeddings)

## Setup

### 1. Install Dependencies

All providers are included in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 2. Get API Keys

- **OpenAI**: https://platform.openai.com/api-keys
- **Gemini**: https://ai.google.dev/
- **Groq**: https://console.groq.com/

### 3. Configure Environment

Edit `.env`:
```bash
# Choose provider (openai, gemini, or groq)
LLM_PROVIDER=openai

# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4-turbo-preview
OPENAI_EMBEDDING_MODEL=text-embedding-3-large

# Google Gemini
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-pro
GEMINI_EMBEDDING_MODEL=models/text-embedding-004

# Groq
GROQ_API_KEY=...
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_EMBEDDING_MODEL=text-embedding-3-large  # Falls back to OpenAI
```

## Usage

### Method 1: Environment Variable (Recommended)

```python
from graphrag import GraphRAG, GraphRAGConfig

# Uses LLM_PROVIDER from .env
config = GraphRAGConfig()
graphrag = GraphRAG(config)
graphrag.connect()

result = graphrag.query("Your question")
print(result.answer)

graphrag.close()
```

### Method 2: Programmatic Selection

```python
from graphrag import GraphRAG, GraphRAGConfig

# OpenAI
config = GraphRAGConfig()
config.llm.provider = "openai"
config.llm.openai_model = "gpt-4-turbo-preview"

# Gemini
config = GraphRAGConfig()
config.llm.provider = "gemini"
config.llm.gemini_model = "gemini-2.5-pro"

# Groq
config = GraphRAGConfig()
config.llm.provider = "groq"
config.llm.groq_model = "llama-3.3-70b-versatile"

graphrag = GraphRAG(config)
```

### Method 3: Context Manager

```python
from graphrag import GraphRAG, GraphRAGConfig

config = GraphRAGConfig()
config.llm.provider = "groq"  # Fast inference

with GraphRAG(config) as graphrag:
    result = graphrag.query("Who directed Inception?")
    print(result.answer)
```

## Comparison Demo

Run the multi-provider comparison tool:

```bash
python examples/multi_provider_demo.py
```

This will:
1. Show provider information (models, context windows, features)
2. List all available models for each provider
3. Test GraphRAG with each provider
4. Display a latency comparison table

## Performance Comparison

Based on typical query latencies:

### Simple Factual Queries
- **Groq**: 200-400ms ⚡ (fastest)
- **Gemini**: 400-900ms
- **OpenAI**: 500-1000ms

### Multi-hop Graph Queries
- **Groq**: 400-1000ms
- **Gemini**: 800-1800ms
- **OpenAI**: 1000-2000ms

### SQL Aggregation Queries
- **Groq**: 150-400ms ⚡
- **OpenAI**: 300-800ms
- **Gemini**: 400-900ms

### Long Context Queries (100K+ tokens)
- **Gemini**: Best (2M context window)
- **Groq**: Good (131K context)
- **OpenAI**: Good (128K context)

## Recommendations

### Use OpenAI When:
- ✅ You need maximum reliability
- ✅ Production environment with SLA requirements
- ✅ Advanced reasoning tasks
- ✅ Function calling is critical

### Use Gemini When:
- ✅ Processing very long documents (>128K tokens)
- ✅ Need multimodal understanding (images + text)
- ✅ Cost optimization (cheaper than GPT-4)
- ✅ Experimental features (code execution)

### Use Groq When:
- ✅ Speed is the top priority
- ✅ High-throughput batch processing
- ✅ Real-time applications (<500ms latency)
- ✅ Cost optimization (open-source models)
- ⚠️ Note: Requires OpenAI key for embeddings

## Implementation Details

### Factory Pattern

The `LLMFactory` class handles provider selection:

```python
from graphrag.llm_factory import LLMFactory
from graphrag.config import GraphRAGConfig

config = GraphRAGConfig()
config.llm.provider = "gemini"

# Create chat LLM
llm = LLMFactory.create_chat_llm(config)

# Create embeddings
embeddings = LLMFactory.create_embeddings(config)

# Get provider info
info = LLMFactory.get_provider_info(config)
print(info)
```

### Auto-Configuration

The `LLMConfig` class automatically selects the right settings:

```python
config = GraphRAGConfig()
config.llm.provider = "gemini"

# Auto-selects Gemini API key
print(config.llm.api_key)  # Returns GEMINI_API_KEY

# Auto-selects Gemini model
print(config.llm.model)  # Returns gemini-2.5-pro

# Auto-selects Gemini embeddings
print(config.llm.embedding_model)  # Returns models/text-embedding-004
```

## Troubleshooting

### Groq Embeddings

Groq doesn't provide embeddings, so GraphRAG falls back to OpenAI:
```python
# When using Groq, also set OpenAI key for embeddings
GROQ_API_KEY=...
OPENAI_API_KEY=sk-...  # Required for embeddings
```

### Model Not Found

If you get a model error:
1. Check the model name is correct (see "Available Models" above)
2. Ensure your API key has access to that model
3. Try a different model version

### Rate Limits

Each provider has different rate limits:
- **OpenAI**: 500 RPM (GPT-4), 3,500 RPM (GPT-3.5)
- **Gemini**: 60 RPM (free tier), higher for paid
- **Groq**: Very high (500+ tokens/sec)

## Migration Guide

### From OpenAI-only to Multi-provider

**Before:**
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4-turbo-preview")
```

**After:**
```python
from graphrag.llm_factory import LLMFactory
from graphrag.config import GraphRAGConfig

config = GraphRAGConfig()
config.llm.provider = "gemini"  # or "openai" or "groq"

llm = LLMFactory.create_chat_llm(config)
```

## Examples

### Compare All Providers

```python
from graphrag import GraphRAG, GraphRAGConfig
import time

providers = ["openai", "gemini", "groq"]
query = "What is the capital of France?"

for provider in providers:
    config = GraphRAGConfig()
    config.llm.provider = provider
    
    with GraphRAG(config) as graphrag:
        start = time.time()
        result = graphrag.query(query)
        latency = time.time() - start
        
        print(f"{provider}: {result.answer} ({latency:.2f}s)")
```

### Use Different Providers for Different Tasks

```python
from graphrag import GraphRAG, GraphRAGConfig

# Fast provider for simple queries
fast_config = GraphRAGConfig()
fast_config.llm.provider = "groq"
fast_graphrag = GraphRAG(fast_config)

# Smart provider for complex reasoning
smart_config = GraphRAGConfig()
smart_config.llm.provider = "openai"
smart_config.llm.openai_model = "gpt-4-turbo-preview"
smart_graphrag = GraphRAG(smart_config)

# Long context provider for documents
long_config = GraphRAGConfig()
long_config.llm.provider = "gemini"
long_config.llm.gemini_model = "gemini-2.5-pro"
long_graphrag = GraphRAG(long_config)
```

## Cost Comparison

Approximate costs per 1M tokens (as of 2024):

| Provider | Model | Input | Output | Total (1M in + 1M out) |
|----------|-------|-------|--------|------------------------|
| OpenAI | GPT-4 Turbo | $10 | $30 | $40 |
| OpenAI | GPT-3.5 Turbo | $0.50 | $1.50 | $2 |
| Gemini | Gemini 2.5 Pro | $3.50 | $10.50 | $14 |
| Gemini | Gemini 3 Flash | $0.075 | $0.30 | $0.375 |
| Groq | Llama 3.3 70B | $0.59 | $0.79 | $1.38 |
| Groq | Llama 3.1 8B | $0.05 | $0.08 | $0.13 |

**Recommendation**: Use Groq for 90% of queries, fall back to OpenAI/Gemini for complex reasoning.

## Future Plans

- [ ] Support for Anthropic Claude
- [ ] Support for local models (Ollama)
- [ ] Support for Azure OpenAI
- [ ] Support for Cohere
- [ ] Dynamic provider selection based on query complexity
- [ ] Cost tracking per provider
- [ ] Automatic fallback if primary provider fails

## References

- [OpenAI Platform](https://platform.openai.com/)
- [Google Gemini API](https://ai.google.dev/)
- [Groq Documentation](https://console.groq.com/docs)
- [LangChain Integrations](https://python.langchain.com/docs/integrations/llms/)
