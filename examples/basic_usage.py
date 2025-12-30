"""Example usage of GraphRAG system."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphrag import GraphRAG, GraphRAGConfig


def basic_example():
    """Basic GraphRAG usage example."""
    print("=" * 60)
    print("GraphRAG Basic Example")
    print("=" * 60)

    # Initialize GraphRAG with default config
    config = GraphRAGConfig()
    graphrag = GraphRAG(config)

    try:
        # Connect to data sources
        print("\n[1] Connecting to data sources...")
        graphrag.connect()
        print("✓ Connected successfully")

        # Example query
        query = "What is the average salary of engineers?"
        print(f"\n[2] Query: {query}")

        # Execute query
        print("\n[3] Executing query...")
        response = graphrag.query(query, explain=True)

        # Display results
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"\nAnswer:\n{response.answer}")
        print(f"\nConfidence: {response.confidence:.0%}")
        print(f"Latency: {response.latency_ms:.2f}ms")

        print(f"\n\nSources ({len(response.sources)}):")
        for i, source in enumerate(response.sources, 1):
            print(f"\n{i}. {source.type.upper()} (confidence: {source.confidence:.0%})")
            print(f"   {source.content[:150]}...")

        print(f"\n\nReasoning Steps ({len(response.reasoning_steps)}):")
        for step in response.reasoning_steps:
            print(f"{step.step}. {step.action}: {step.rationale}")

    except Exception as e:
        print(f"\n✗ Error: {e}")

    finally:
        # Clean up
        print("\n[4] Closing connections...")
        graphrag.close()
        print("✓ Done")


def context_manager_example():
    """Example using context manager."""
    print("\n" + "=" * 60)
    print("GraphRAG Context Manager Example")
    print("=" * 60)

    with GraphRAG() as graphrag:
        queries = [
            "Who directed Inception?",
            "What movies did Christopher Nolan direct?",
            "Average budget of sci-fi movies",
        ]

        for i, query in enumerate(queries, 1):
            print(f"\n[{i}] Query: {query}")
            try:
                response = graphrag.query(query)
                print(f"Answer: {response.answer}")
                print(f"Confidence: {response.confidence:.0%}")
            except Exception as e:
                print(f"Error: {e}")


def vector_search_example():
    """Example focusing on vector search."""
    print("\n" + "=" * 60)
    print("Vector Search Example")
    print("=" * 60)

    with GraphRAG() as graphrag:
        # Add some sample documents
        print("\n[1] Adding sample documents to vector store...")

        if graphrag.vector_store:
            documents = [
                "Python is a high-level programming language known for its simplicity.",
                "Machine learning is a subset of artificial intelligence.",
                "Neural networks are inspired by biological neural networks.",
                "Data science combines statistics, programming, and domain knowledge.",
            ]

            metadatas = [
                {"topic": "programming", "source": "doc1"},
                {"topic": "ai", "source": "doc2"},
                {"topic": "ai", "source": "doc3"},
                {"topic": "data", "source": "doc4"},
            ]

            count = graphrag.vector_store.add_documents(
                documents=documents,
                metadatas=metadatas,
            )
            print(f"✓ Added {count} chunks")

            # Query the vector store
            query = "Tell me about artificial intelligence"
            print(f"\n[2] Query: {query}")

            response = graphrag.query(query)
            print(f"\nAnswer: {response.answer}")
            print(f"Sources: {len(response.sources)}")


def multi_hop_example():
    """Example of multi-hop reasoning query."""
    print("\n" + "=" * 60)
    print("Multi-Hop Reasoning Example")
    print("=" * 60)

    with GraphRAG() as graphrag:
        # This would require a populated graph database
        query = "Which movie directed by the person who directed Inception won an Oscar?"

        print(f"\nQuery: {query}")
        print("\n[INFO] This requires a populated Neo4j graph database.")
        print("[INFO] See setup instructions in README.md")

        try:
            response = graphrag.query(query, explain=True)
            print(f"\nAnswer: {response.answer}")
            print(f"\nReasoning Steps:")
            for step in response.reasoning_steps:
                print(f"  {step.step}. {step.action}")
        except Exception as e:
            print(f"\nNote: {e}")
            print("This example requires Neo4j to be running with sample data.")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("GRAPHRAG EXAMPLES")
    print("=" * 60)
    print("\nNOTE: Make sure you have:")
    print("1. Created a .env file (copy from .env.example)")
    print("2. Set your OPENAI_API_KEY")
    print("3. Started Neo4j (if using graph queries)")
    print("4. Started PostgreSQL (if using SQL queries)")
    print("\nPress Enter to continue or Ctrl+C to exit...")

    try:
        input()
    except KeyboardInterrupt:
        print("\nExiting...")
        return

    # Run examples
    try:
        basic_example()
        # Uncomment to run other examples:
        # context_manager_example()
        # vector_search_example()
        # multi_hop_example()

    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user")
    except Exception as e:
        print(f"\n\nExample failed with error: {e}")


if __name__ == "__main__":
    main()
