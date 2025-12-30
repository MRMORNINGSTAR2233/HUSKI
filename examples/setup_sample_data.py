"""Setup script to populate sample data for GraphRAG."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graphrag.graph_store import GraphStore
from graphrag.vector_store import VectorStore
from graphrag.config import GraphRAGConfig


def setup_sample_graph():
    """Populate Neo4j with sample movie data."""
    print("Setting up sample graph data...")

    config = GraphRAGConfig()

    with GraphStore(config.neo4j) as graph:
        # Add sample people
        people = [
            {"name": "Christopher Nolan", "birth_year": 1970},
            {"name": "Leonardo DiCaprio", "birth_year": 1974},
            {"name": "Matthew McConaughey", "birth_year": 1969},
            {"name": "Tom Hardy", "birth_year": 1977},
        ]

        for person in people:
            try:
                graph.add_entity("Person", person)
                print(f"  ✓ Added person: {person['name']}")
            except Exception as e:
                print(f"  ✗ Error adding {person['name']}: {e}")

        # Add sample movies
        movies = [
            {"title": "Inception", "year": 2010, "budget": 160000000, "genre": "Sci-Fi"},
            {"title": "Interstellar", "year": 2014, "budget": 165000000, "genre": "Sci-Fi"},
            {"title": "Dunkirk", "year": 2017, "budget": 100000000, "genre": "War"},
            {"title": "The Dark Knight", "year": 2008, "budget": 185000000, "genre": "Action"},
        ]

        for movie in movies:
            try:
                graph.add_entity("Movie", movie)
                print(f"  ✓ Added movie: {movie['title']}")
            except Exception as e:
                print(f"  ✗ Error adding {movie['title']}: {e}")

        # Add relationships
        relationships = [
            ("Person", "name", "Christopher Nolan", "Movie", "title", "Inception", "DIRECTED"),
            ("Person", "name", "Christopher Nolan", "Movie", "title", "Interstellar", "DIRECTED"),
            ("Person", "name", "Christopher Nolan", "Movie", "title", "Dunkirk", "DIRECTED"),
            ("Person", "name", "Christopher Nolan", "Movie", "title", "The Dark Knight", "DIRECTED"),
            ("Person", "name", "Leonardo DiCaprio", "Movie", "title", "Inception", "ACTED_IN"),
            ("Person", "name", "Matthew McConaughey", "Movie", "title", "Interstellar", "ACTED_IN"),
            ("Person", "name", "Tom Hardy", "Movie", "title", "Inception", "ACTED_IN"),
            ("Person", "name", "Tom Hardy", "Movie", "title", "Dunkirk", "ACTED_IN"),
        ]

        for rel in relationships:
            try:
                graph.add_relationship(*rel)
                print(f"  ✓ Added relationship: {rel[1]} {rel[6]} {rel[4]}")
            except Exception as e:
                print(f"  ✗ Error adding relationship: {e}")

    print("\n✓ Sample graph data setup complete!")


def setup_sample_documents():
    """Populate ChromaDB with sample documents."""
    print("\nSetting up sample vector documents...")

    config = GraphRAGConfig()

    with VectorStore(config.chroma, config.llm) as vector_store:
        documents = [
            "Christopher Nolan is a British-American film director known for his cerebral approach to filmmaking. He has directed several critically acclaimed films including Inception, Interstellar, and The Dark Knight trilogy.",
            "Inception (2010) is a science fiction action film about a thief who steals corporate secrets through dream-sharing technology. The film grossed over $829 million worldwide and won four Academy Awards.",
            "Interstellar (2014) is a science fiction film about a team of astronauts who travel through a wormhole in search of a new habitable planet. It won the Academy Award for Best Visual Effects.",
            "Dunkirk (2017) is a war film depicting the Dunkirk evacuation of World War II. It was nominated for eight Academy Awards and won three, including Best Sound Editing and Best Sound Mixing.",
            "The Dark Knight (2008) is a superhero film and the second installment in The Dark Knight Trilogy. It is considered one of the greatest superhero films ever made and grossed over $1 billion worldwide.",
        ]

        metadatas = [
            {"type": "biography", "subject": "Christopher Nolan"},
            {"type": "movie_info", "title": "Inception", "year": 2010},
            {"type": "movie_info", "title": "Interstellar", "year": 2014},
            {"type": "movie_info", "title": "Dunkirk", "year": 2017},
            {"type": "movie_info", "title": "The Dark Knight", "year": 2008},
        ]

        count = vector_store.add_documents(
            documents=documents,
            metadatas=metadatas,
            chunk=False,
        )

        print(f"  ✓ Added {count} documents")

    print("\n✓ Sample vector documents setup complete!")


def setup_sample_sql_data():
    """Create and populate sample SQL database."""
    print("\nSetting up sample SQL data...")
    print("[INFO] This requires PostgreSQL to be running.")
    print("[INFO] You'll need to manually create tables and insert data.")
    print("\nExample SQL:")
    print("""
    CREATE TABLE employees (
        id SERIAL PRIMARY KEY,
        name VARCHAR(100),
        role VARCHAR(50),
        department VARCHAR(50),
        salary DECIMAL(10, 2),
        hire_date DATE,
        is_contractor BOOLEAN DEFAULT FALSE
    );

    INSERT INTO employees (name, role, department, salary, hire_date, is_contractor) VALUES
        ('Alice Smith', 'Engineer', 'Engineering', 125000, '2020-01-15', FALSE),
        ('Bob Johnson', 'Engineer', 'Engineering', 135000, '2019-06-20', FALSE),
        ('Carol White', 'Designer', 'Design', 95000, '2021-03-10', FALSE),
        ('David Brown', 'Engineer', 'Engineering', 115000, '2022-02-01', TRUE),
        ('Eve Davis', 'Manager', 'Engineering', 155000, '2018-11-05', FALSE);
    """)


def main():
    """Run all setup scripts."""
    print("=" * 60)
    print("GRAPHRAG SAMPLE DATA SETUP")
    print("=" * 60)

    print("\nThis script will populate sample data for GraphRAG.")
    print("\nPrerequisites:")
    print("1. Neo4j running at bolt://localhost:7687")
    print("2. PostgreSQL running at localhost:5432")
    print("3. .env file configured with API keys and credentials")

    print("\nPress Enter to continue or Ctrl+C to exit...")
    try:
        input()
    except KeyboardInterrupt:
        print("\nExiting...")
        return

    # Setup graph data
    try:
        setup_sample_graph()
    except Exception as e:
        print(f"\n✗ Graph setup failed: {e}")
        print("Make sure Neo4j is running and credentials are correct.")

    # Setup vector data
    try:
        setup_sample_documents()
    except Exception as e:
        print(f"\n✗ Vector setup failed: {e}")
        print("Make sure OPENAI_API_KEY is set in .env file.")

    # Setup SQL data
    setup_sample_sql_data()

    print("\n" + "=" * 60)
    print("Setup complete! You can now run examples/basic_usage.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
