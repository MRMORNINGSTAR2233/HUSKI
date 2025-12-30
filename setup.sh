#!/bin/bash

# GraphRAG Quick Setup Script
# This script helps you set up the GraphRAG environment

set -e

echo "=========================================="
echo "GraphRAG Setup Script"
echo "=========================================="
echo ""

# Check Python version
echo "[1/6] Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
required_version="3.10"

if (( $(echo "$python_version < $required_version" | bc -l) )); then
    echo "❌ Python 3.10+ required (found $python_version)"
    exit 1
else
    echo "✓ Python $python_version detected"
fi

# Create .env file if it doesn't exist
echo ""
echo "[2/6] Setting up environment file..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✓ Created .env file from template"
    echo "⚠️  Please edit .env and add your API keys and credentials"
else
    echo "✓ .env file already exists"
fi

# Install dependencies
echo ""
echo "[3/6] Installing Python dependencies..."
if command -v poetry &> /dev/null; then
    echo "Using Poetry..."
    poetry install
else
    echo "Using pip..."
    pip install -r requirements.txt
fi
echo "✓ Dependencies installed"

# Check if Docker is running
echo ""
echo "[4/6] Checking Docker..."
if command -v docker &> /dev/null; then
    if docker ps &> /dev/null; then
        echo "✓ Docker is running"
        
        # Start Neo4j
        echo ""
        echo "[5/6] Starting Neo4j (if not already running)..."
        if [ ! "$(docker ps -q -f name=neo4j)" ]; then
            docker run -d \
                --name neo4j \
                -p 7474:7474 -p 7687:7687 \
                -e NEO4J_AUTH=neo4j/graphrag123 \
                neo4j:latest
            echo "✓ Neo4j started on ports 7474 (HTTP) and 7687 (Bolt)"
            echo "  Access at: http://localhost:7474"
            echo "  Default credentials: neo4j/graphrag123"
        else
            echo "✓ Neo4j already running"
        fi
        
        # Start PostgreSQL
        echo ""
        echo "[6/6] Starting PostgreSQL (if not already running)..."
        if [ ! "$(docker ps -q -f name=postgres-graphrag)" ]; then
            docker run -d \
                --name postgres-graphrag \
                -p 5432:5432 \
                -e POSTGRES_PASSWORD=graphrag123 \
                -e POSTGRES_DB=graphrag_db \
                postgres:latest
            echo "✓ PostgreSQL started on port 5432"
            echo "  Database: graphrag_db"
            echo "  Default credentials: postgres/graphrag123"
        else
            echo "✓ PostgreSQL already running"
        fi
    else
        echo "❌ Docker daemon not running. Please start Docker."
        exit 1
    fi
else
    echo "⚠️  Docker not found. Please install Docker to run Neo4j and PostgreSQL."
    echo "   You can manually install and configure these databases."
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file with your OPENAI_API_KEY"
echo "2. If using default Docker setup, update .env with:"
echo "   NEO4J_PASSWORD=graphrag123"
echo "   POSTGRES_PASSWORD=graphrag123"
echo ""
echo "3. Load sample data:"
echo "   python examples/setup_sample_data.py"
echo ""
echo "4. Run examples:"
echo "   python examples/basic_usage.py"
echo ""
echo "5. Or use in your code:"
echo "   from graphrag import GraphRAG"
echo "   with GraphRAG() as graphrag:"
echo "       response = graphrag.query('Your question here')"
echo ""
echo "For more info, see README.md"
echo ""
