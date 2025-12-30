"""Example demonstrating multiple LLM provider support (OpenAI, Gemini, Groq)."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphrag import GraphRAG, GraphRAGConfig, LLMFactory


def print_provider_info(config):
    """Print information about the configured provider."""
    info = LLMFactory.get_provider_info(config.llm)
    print(f"\n{'='*60}")
    print(f"Provider: {info['name']}")
    print(f"{'='*60}")
    print(f"Model: {info['model']}")
    print(f"Embedding Model: {info['embedding_model']}")
    print(f"Context Window: {info['context_window']}")
    print(f"\nFeatures:")
    for feature in info.get('features', []):
        print(f"  • {feature}")
    print(f"{'='*60}\n")


def test_with_provider(provider_name: str):
    """Test GraphRAG with a specific provider.

    Args:
        provider_name: Name of the provider (openai, gemini, or groq)
    """
    print(f"\n\n{'#'*60}")
    print(f"# Testing with {provider_name.upper()} Provider")
    print(f"{'#'*60}")

    # Create config for specific provider
    config = GraphRAGConfig()
    config.llm.provider = provider_name

    # Print provider information
    print_provider_info(config)

    # Test query
    query = "What is machine learning?"

    try:
        with GraphRAG(config) as graphrag:
            print(f"Query: {query}")
            print("\nExecuting query...")

            response = graphrag.query(query)

            print(f"\n{'='*60}")
            print("RESULTS")
            print(f"{'='*60}")
            print(f"\nAnswer:\n{response.answer}\n")
            print(f"Confidence: {response.confidence:.0%}")
            print(f"Latency: {response.latency_ms:.2f}ms")
            print(f"Sources: {len(response.sources)}")

            print(f"\nReasoning Steps:")
            for step in response.reasoning_steps:
                print(f"  {step.step}. {step.action}")

    except Exception as e:
        print(f"\n✗ Error with {provider_name}: {e}")
        print(f"  Make sure {provider_name.upper()}_API_KEY is set in .env file")


def compare_providers():
    """Compare all three providers with the same query."""
    print("\n" + "="*60)
    print("MULTI-PROVIDER COMPARISON")
    print("="*60)

    providers = ["openai", "gemini", "groq"]
    query = "Explain the concept of neural networks in simple terms"

    results = {}

    for provider in providers:
        try:
            config = GraphRAGConfig()
            config.llm.provider = provider

            # Quick test to see if API key is configured
            if provider == "openai" and not config.llm.openai_api_key:
                print(f"\n⚠️  Skipping {provider.upper()}: API key not configured")
                continue
            elif provider == "gemini" and not config.llm.gemini_api_key:
                print(f"\n⚠️  Skipping {provider.upper()}: API key not configured")
                continue
            elif provider == "groq" and not config.llm.groq_api_key:
                print(f"\n⚠️  Skipping {provider.upper()}: API key not configured")
                continue

            print(f"\n📊 Testing {provider.upper()}...")

            with GraphRAG(config) as graphrag:
                response = graphrag.query(query)
                results[provider] = {
                    "answer_length": len(response.answer),
                    "confidence": response.confidence,
                    "latency_ms": response.latency_ms,
                    "sources": len(response.sources),
                }
                print(f"  ✓ Completed in {response.latency_ms:.2f}ms")

        except Exception as e:
            print(f"  ✗ Error: {e}")
            results[provider] = {"error": str(e)}

    # Print comparison table
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"\n{'Provider':<12} {'Latency':<12} {'Confidence':<12} {'Sources':<10}")
    print("-" * 60)

    for provider, data in results.items():
        if "error" not in data:
            print(
                f"{provider.upper():<12} "
                f"{data['latency_ms']:.0f}ms{'':<7} "
                f"{data['confidence']:.0%}{'':<7} "
                f"{data['sources']}"
            )
        else:
            print(f"{provider.upper():<12} ERROR: {data['error'][:40]}")

    print("-" * 60)


def show_available_models():
    """Display available models for each provider."""
    print("\n" + "="*60)
    print("AVAILABLE MODELS")
    print("="*60)

    models_info = {
        "OpenAI": [
            "gpt-4-turbo-preview (Latest GPT-4 Turbo)",
            "gpt-4 (Standard GPT-4)",
            "gpt-3.5-turbo (Fast and cost-effective)",
        ],
        "Google Gemini": [
            "gemini-2.5-pro (Advanced reasoning, 2M context)",
            "gemini-2.5-flash (Fast, balanced)",
            "gemini-2.5-flash-lite (Ultra fast)",
            "gemini-3-pro (Most intelligent)",
            "gemini-3-flash (Most balanced)",
        ],
        "Groq": [
            "llama-3.3-70b-versatile (280 tps, 131K context)",
            "llama-3.1-8b-instant (560 tps, fast)",
            "openai/gpt-oss-120b (500 tps, 131K context)",
            "openai/gpt-oss-20b (1000 tps, ultra fast)",
        ],
    }

    for provider, models in models_info.items():
        print(f"\n{provider}:")
        for model in models:
            print(f"  • {model}")

    print("\n" + "="*60)
    print("To change model, set these in .env:")
    print("  OPENAI_MODEL=gpt-4-turbo-preview")
    print("  GEMINI_MODEL=gemini-2.5-pro")
    print("  GROQ_MODEL=llama-3.3-70b-versatile")
    print("="*60)


def main():
    """Run provider examples."""
    print("\n" + "="*60)
    print("GRAPHRAG MULTI-PROVIDER EXAMPLES")
    print("="*60)
    print("\nThis example demonstrates GraphRAG with multiple LLM providers:")
    print("  • OpenAI (GPT-4, GPT-3.5)")
    print("  • Google Gemini (Gemini 2.5/3.0)")
    print("  • Groq (Llama 3.3, GPT-OSS)")

    print("\nMake sure you have set the appropriate API keys in .env:")
    print("  LLM_PROVIDER=openai|gemini|groq")
    print("  OPENAI_API_KEY=sk-...")
    print("  GEMINI_API_KEY=...")
    print("  GROQ_API_KEY=...")

    # Show available models
    show_available_models()

    print("\nPress Enter to continue or Ctrl+C to exit...")
    try:
        input()
    except KeyboardInterrupt:
        print("\nExiting...")
        return

    # Test with configured provider
    config = GraphRAGConfig()
    current_provider = config.llm.provider

    print(f"\n✓ Current provider: {current_provider.upper()}")
    test_with_provider(current_provider)

    # Optionally test all providers
    print("\n\nWould you like to compare all providers? (y/n): ", end="")
    try:
        choice = input().strip().lower()
        if choice == 'y':
            compare_providers()
    except KeyboardInterrupt:
        print("\n\nExamples completed!")

    # Example: Testing each provider individually
    print("\n\nWant to test specific providers? (openai/gemini/groq/skip): ", end="")
    try:
        choice = input().strip().lower()
        if choice in ['openai', 'gemini', 'groq']:
            test_with_provider(choice)
    except KeyboardInterrupt:
        pass

    print("\n\n" + "="*60)
    print("Examples completed!")
    print("="*60)
    print("\nTo use a specific provider in your code:")
    print("""
from graphrag import GraphRAG, GraphRAGConfig

# Method 1: Use environment variable LLM_PROVIDER
config = GraphRAGConfig()
graphrag = GraphRAG(config)

# Method 2: Set provider programmatically
config = GraphRAGConfig()
config.llm.provider = "gemini"  # or "openai" or "groq"
graphrag = GraphRAG(config)
    """)


if __name__ == "__main__":
    main()
