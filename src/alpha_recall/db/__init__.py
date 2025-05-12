"""
Database layer for alpha_recall.
"""

from alpha_recall.db.base import GraphDatabase
from alpha_recall.db.neo4j_db import Neo4jDatabase
from alpha_recall.db.semantic_search import SemanticSearch
from alpha_recall.db.vector_store import VectorStore
from alpha_recall.db.composite_db import CompositeDatabase
from alpha_recall.db.factory import create_db_instance, create_graph_db, create_vector_store

__all__ = [
    # Base classes
    "GraphDatabase", 
    "SemanticSearch",
    
    # Implementations
    "Neo4jDatabase", 
    "VectorStore",
    "CompositeDatabase",
    
    # Factory functions
    "create_db_instance",
    "create_graph_db",
    "create_vector_store"
]

