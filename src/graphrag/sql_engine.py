"""Text-to-SQL engine with safety validators."""

import logging
import time
import re
from typing import Dict, Any, List, Optional
import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Where
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from .config import PostgresConfig, LLMConfig
from .models import SQLResult

logger = logging.getLogger(__name__)


class SQLEngine:
    """Generates and executes SQL queries with safety validation."""

    # SQL operations whitelist
    ALLOWED_OPERATIONS = {"SELECT"}
    FORBIDDEN_KEYWORDS = {
        "DELETE",
        "DROP",
        "TRUNCATE",
        "ALTER",
        "CREATE",
        "INSERT",
        "UPDATE",
        "GRANT",
        "REVOKE",
        "EXEC",
        "EXECUTE",
    }

    def __init__(self, postgres_config: PostgresConfig, llm_config: LLMConfig):
        """Initialize SQL engine.

        Args:
            postgres_config: PostgreSQL configuration
            llm_config: LLM configuration
        """
        self.postgres_config = postgres_config
        self.llm_config = llm_config
        self._engine: Optional[Engine] = None

        self.llm = ChatOpenAI(
            api_key=llm_config.api_key,
            model=llm_config.model,
            temperature=0.0,
        )

    def connect(self) -> None:
        """Establish database connection."""
        try:
            self._engine = create_engine(
                self.postgres_config.connection_string,
                pool_pre_ping=True,
                echo=False,
            )
            # Test connection
            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            logger.info("Connected to PostgreSQL database")

        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    def close(self) -> None:
        """Close database connection."""
        if self._engine:
            self._engine.dispose()
            logger.info("PostgreSQL connection closed")

    def get_schema(self) -> Dict[str, Any]:
        """Get database schema information.

        Returns:
            Dictionary with tables and columns
        """
        if not self._engine:
            raise RuntimeError("Not connected to database. Call connect() first.")

        inspector = inspect(self._engine)
        schema = {}

        for table_name in inspector.get_table_names():
            columns = []
            for column in inspector.get_columns(table_name):
                columns.append({
                    "name": column["name"],
                    "type": str(column["type"]),
                    "nullable": column["nullable"],
                })
            schema[table_name] = columns

        logger.info(f"Retrieved schema for {len(schema)} tables")
        return schema

    def validate_sql(self, sql: str) -> tuple[bool, Optional[str]]:
        """Validate SQL query for safety.

        Args:
            sql: SQL query string

        Returns:
            Tuple of (is_valid, error_message)
        """
        sql_upper = sql.upper()

        # Check for forbidden keywords
        for keyword in self.FORBIDDEN_KEYWORDS:
            if re.search(rf'\b{keyword}\b', sql_upper):
                return False, f"Forbidden keyword detected: {keyword}"

        # Parse SQL
        try:
            parsed = sqlparse.parse(sql)
            if not parsed:
                return False, "Empty or invalid SQL"

            # Check operation type
            first_token = parsed[0].get_type()
            if first_token not in ["SELECT", "UNKNOWN"]:  # UNKNOWN for some valid SELECTs
                # Double-check it's actually a SELECT
                if not sql_upper.strip().startswith("SELECT"):
                    return False, f"Only SELECT queries allowed, got: {first_token}"

            # Additional safety checks
            if "--" in sql or "/*" in sql or "*/" in sql:
                return False, "SQL comments not allowed"

            if ";" in sql[:-1]:  # Allow trailing semicolon
                return False, "Multiple statements not allowed"

            return True, None

        except Exception as e:
            return False, f"SQL parsing error: {e}"

    def generate_sql(self, query: str, schema: Optional[Dict[str, Any]] = None) -> str:
        """Generate SQL from natural language query.

        Args:
            query: Natural language query
            schema: Database schema (if None, will fetch automatically)

        Returns:
            Generated SQL query
        """
        if schema is None:
            schema = self.get_schema()

        # Format schema for prompt
        schema_str = self._format_schema_for_prompt(schema)

        sql_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a SQL expert. Generate ONLY valid PostgreSQL SELECT queries.

Rules:
1. ONLY generate SELECT queries (no INSERT, UPDATE, DELETE, DROP, etc.)
2. Use proper table and column names from the schema
3. Include appropriate WHERE clauses, JOINs, and GROUP BY as needed
4. For aggregations, use functions like AVG, SUM, COUNT, etc.
5. Respond with ONLY the SQL query, no explanations or markdown

Database Schema:
{schema}

Example:
Query: "Average salary of engineers"
SQL: SELECT AVG(salary) FROM employees WHERE role = 'Engineer';"""),
            ("user", "{query}")
        ])

        try:
            messages = sql_prompt.format_messages(query=query, schema=schema_str)
            response = self.llm.invoke(messages)

            # Extract SQL from response
            sql = response.content.strip()

            # Remove markdown code blocks if present
            if sql.startswith("```"):
                sql = re.sub(r'```(?:sql)?\n?', '', sql).strip()

            # Remove trailing semicolon for consistency
            sql = sql.rstrip(";")

            logger.info(f"Generated SQL: {sql}")

            # Validate generated SQL
            is_valid, error = self.validate_sql(sql)
            if not is_valid:
                raise ValueError(f"Generated SQL failed validation: {error}")

            return sql

        except Exception as e:
            logger.error(f"SQL generation failed: {e}")
            raise

    def execute_sql(self, sql: str) -> SQLResult:
        """Execute SQL query.

        Args:
            sql: SQL query to execute

        Returns:
            SQLResult with query results
        """
        if not self._engine:
            raise RuntimeError("Not connected to database. Call connect() first.")

        # Validate before execution
        is_valid, error = self.validate_sql(sql)
        if not is_valid:
            raise ValueError(f"SQL validation failed: {error}")

        start_time = time.time()

        try:
            with self._engine.connect() as conn:
                result = conn.execute(text(sql))
                rows = result.fetchall()

                # Convert to list of dicts
                columns = list(result.keys())
                data = [dict(zip(columns, row)) for row in rows]

            execution_time = (time.time() - start_time) * 1000

            logger.info(
                f"SQL executed in {execution_time:.2f}ms, "
                f"returned {len(data)} rows"
            )

            return SQLResult(
                data=data,
                sql_query=sql,
                execution_time_ms=execution_time,
                row_count=len(data),
                columns=columns,
            )

        except Exception as e:
            logger.error(f"SQL execution failed: {e}")
            raise

    def query(self, natural_language_query: str) -> SQLResult:
        """Generate and execute SQL from natural language.

        Args:
            natural_language_query: Natural language query

        Returns:
            SQLResult with query results
        """
        # Generate SQL
        sql = self.generate_sql(natural_language_query)

        # Execute SQL
        return self.execute_sql(sql)

    def _format_schema_for_prompt(self, schema: Dict[str, Any]) -> str:
        """Format schema for LLM prompt.

        Args:
            schema: Database schema

        Returns:
            Formatted schema string
        """
        lines = []
        for table_name, columns in schema.items():
            lines.append(f"\nTable: {table_name}")
            for col in columns:
                nullable = "NULL" if col["nullable"] else "NOT NULL"
                lines.append(f"  - {col['name']} ({col['type']}) {nullable}")

        return "\n".join(lines)

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
