"""Neo4j graph database manager for GraphRAG."""

import logging
import time
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, AuthError

from .config import Neo4jConfig
from .models import GraphResult

logger = logging.getLogger(__name__)


class GraphStore:
    """Manages interactions with Neo4j knowledge graph."""

    def __init__(self, config: Neo4jConfig):
        """Initialize Neo4j connection.

        Args:
            config: Neo4j configuration
        """
        self.config = config
        self._driver: Optional[Driver] = None

    def connect(self) -> None:
        """Establish connection to Neo4j database."""
        try:
            self._driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
            )
            # Verify connectivity
            self._driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self.config.uri}")
        except AuthError as e:
            logger.error(f"Authentication failed: {e}")
            raise
        except ServiceUnavailable as e:
            logger.error(f"Neo4j service unavailable: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self) -> None:
        """Close Neo4j connection."""
        if self._driver:
            self._driver.close()
            logger.info("Neo4j connection closed")

    def execute_cypher(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> GraphResult:
        """Execute a Cypher query.

        Args:
            query: Cypher query string
            parameters: Query parameters

        Returns:
            GraphResult with query results
        """
        if not self._driver:
            raise RuntimeError("Not connected to Neo4j. Call connect() first.")

        start_time = time.time()
        try:
            with self._driver.session(database=self.config.database) as session:
                result = session.run(query, parameters or {})
                records = [record.data() for record in result]

            execution_time = (time.time() - start_time) * 1000

            # Parse entities and relationships from results
            entities = []
            relationships = []

            for record in records:
                for key, value in record.items():
                    if isinstance(value, dict):
                        # Likely a node
                        entities.append(value)
                    elif isinstance(value, list):
                        # Potentially relationships or paths
                        for item in value:
                            if isinstance(item, dict):
                                relationships.append(item)

            logger.info(f"Cypher query executed in {execution_time:.2f}ms")

            return GraphResult(
                entities=entities,
                relationships=relationships,
                cypher_query=query,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            logger.error(f"Cypher query failed: {e}")
            raise

    def add_entity(
        self, label: str, properties: Dict[str, Any], return_entity: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Add an entity (node) to the graph.

        Args:
            label: Node label (e.g., "Person", "Movie")
            properties: Node properties
            return_entity: Whether to return the created entity

        Returns:
            Created entity if return_entity=True
        """
        # Create property string for Cypher
        prop_keys = ", ".join([f"{k}: ${k}" for k in properties.keys()])

        query = f"""
        CREATE (n:{label} {{{prop_keys}}})
        {"RETURN n" if return_entity else ""}
        """

        result = self.execute_cypher(query, properties)

        if return_entity and result.entities:
            return result.entities[0]
        return None

    def add_relationship(
        self,
        from_label: str,
        from_property: str,
        from_value: Any,
        to_label: str,
        to_property: str,
        to_value: Any,
        relationship_type: str,
        relationship_properties: Optional[Dict[str, Any]] = None,
    ) -> GraphResult:
        """Add a relationship between two entities.

        Args:
            from_label: Source node label
            from_property: Property to match source node
            from_value: Value of source property
            to_label: Target node label
            to_property: Property to match target node
            to_value: Value of target property
            relationship_type: Type of relationship
            relationship_properties: Additional relationship properties

        Returns:
            GraphResult with created relationship
        """
        rel_props = relationship_properties or {}
        prop_string = ""
        if rel_props:
            prop_keys = ", ".join([f"{k}: ${k}" for k in rel_props.keys()])
            prop_string = f" {{{prop_keys}}}"

        query = f"""
        MATCH (a:{from_label} {{{from_property}: $from_value}})
        MATCH (b:{to_label} {{{to_property}: $to_value}})
        CREATE (a)-[r:{relationship_type}{prop_string}]->(b)
        RETURN a, r, b
        """

        parameters = {
            "from_value": from_value,
            "to_value": to_value,
            **rel_props,
        }

        return self.execute_cypher(query, parameters)

    def find_entity(self, label: str, properties: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find entities matching given properties.

        Args:
            label: Node label
            properties: Properties to match

        Returns:
            List of matching entities
        """
        prop_keys = " AND ".join([f"n.{k} = ${k}" for k in properties.keys()])

        query = f"""
        MATCH (n:{label})
        WHERE {prop_keys}
        RETURN n
        """

        result = self.execute_cypher(query, properties)
        return result.entities

    def traverse_path(
        self,
        start_label: str,
        start_property: str,
        start_value: Any,
        relationship_pattern: str,
        max_hops: int = 3,
    ) -> GraphResult:
        """Traverse graph paths from a starting node.

        Args:
            start_label: Starting node label
            start_property: Property to identify start node
            start_value: Value of start property
            relationship_pattern: Pattern like "[:DIRECTED*1..3]" or "[:ACTED_IN]->()"
            max_hops: Maximum traversal depth

        Returns:
            GraphResult with path information
        """
        query = f"""
        MATCH path = (start:{start_label} {{{start_property}: $start_value}}){relationship_pattern}
        RETURN path, nodes(path) as nodes, relationships(path) as rels
        LIMIT 100
        """

        parameters = {"start_value": start_value}

        result = self.execute_cypher(query, parameters)

        # Generate path description
        if result.entities:
            path_desc = f"Found {len(result.entities)} paths from {start_label}"
            result.path_description = path_desc

        return result

    def multi_hop_query(
        self, start_entity: Dict[str, Any], hops: List[str], end_condition: Optional[str] = None
    ) -> GraphResult:
        """Execute multi-hop graph traversal query.

        Args:
            start_entity: Starting entity with label and properties
            hops: List of relationship types to traverse
            end_condition: Optional condition for end nodes

        Returns:
            GraphResult with traversal results
        """
        # Build dynamic query for multi-hop
        start_label = start_entity.get("label")
        start_props = start_entity.get("properties", {})

        # Create match pattern for each hop
        match_parts = [f"(n0:{start_label})"]
        where_parts = []

        # Add start node conditions
        for key, value in start_props.items():
            where_parts.append(f"n0.{key} = ${key}")

        # Add relationship hops
        for i, rel_type in enumerate(hops):
            match_parts.append(f"-[:{rel_type}]->(n{i + 1})")

        if end_condition:
            where_parts.append(end_condition)

        match_clause = "".join(match_parts)
        where_clause = " AND ".join(where_parts) if where_parts else ""

        query = f"""
        MATCH {match_clause}
        {"WHERE " + where_clause if where_clause else ""}
        RETURN *
        LIMIT 50
        """

        return self.execute_cypher(query, start_props)

    def get_schema(self) -> Dict[str, Any]:
        """Get graph schema information.

        Returns:
            Dictionary with node labels and relationship types
        """
        # Get node labels
        labels_query = "CALL db.labels() YIELD label RETURN collect(label) as labels"
        labels_result = self.execute_cypher(labels_query)

        # Get relationship types
        rels_query = "CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types"
        rels_result = self.execute_cypher(rels_query)

        return {
            "node_labels": labels_result.entities[0].get("labels", []) if labels_result.entities else [],
            "relationship_types": rels_result.entities[0].get("types", []) if rels_result.entities else [],
        }

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
