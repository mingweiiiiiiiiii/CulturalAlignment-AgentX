# db.py
# This module handles interactions with the Milvus database for storing and retrieving cultural embeddings.

from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType
import numpy as np
from typing import Dict
from graph_types import GraphState

# Milvus connection parameters
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "cultural_expert_embeddings"

def connect_milvus():
    """Establishes a connection to the Milvus database."""
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)

def create_collection(dim: int = 768):
    """Creates a collection in Milvus for storing cultural embeddings.

    Args:
        dim (int): The dimensionality of the embeddings. Default is 768.
    """
    connect_milvus()

    if COLLECTION_NAME in Collection.list_collections():
        print(f"Collection '{COLLECTION_NAME}' already exists.")
        return

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]

    schema = CollectionSchema(fields, description="Expert cultural embeddings")
    collection = Collection(name=COLLECTION_NAME, schema=schema)
    collection.create_index(field_name="embedding", index_params={
        "metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}
    })
    collection.load()

def insert_embedding(embedding: np.ndarray):
    """Inserts an embedding into the Milvus collection.

    Args:
        embedding (np.ndarray): The embedding to be inserted.

    Raises:
        TypeError: If the embedding is not a numpy ndarray.
        ValueError: If the embedding does not have the correct shape.
    """
    connect_milvus()
    collection = Collection(COLLECTION_NAME)
    # Validate embedding shape and type
    if not isinstance(embedding, np.ndarray):
        raise TypeError("Embedding must be a numpy ndarray.")
    if embedding.ndim != 1:
        raise ValueError("Embedding must be a 1D numpy array.")
    if embedding.shape[0] != 768:
        raise ValueError(f"Embedding must have dimension 768, got {embedding.shape[0]}")
    try:
        # Milvus expects a list of fields, each field is a list of values
        collection.insert([embedding.tolist()])
        collection.flush()
    except Exception as e:
        print(f"Error inserting embedding into Milvus: {e}")
        raise

def search_similar(embedding: np.ndarray, top_k: int = 3):
    """Searches for similar embeddings in the Milvus collection.

    Args:
        embedding (np.ndarray): The embedding to search for similar entries.
        top_k (int): The number of similar embeddings to return.

    Returns:
        List: A list of similar embeddings.

    Raises:
        TypeError: If the embedding is not a numpy ndarray.
        ValueError: If the embedding does not have the correct shape.
    """
    connect_milvus()
    collection = Collection(COLLECTION_NAME)
    collection.load()
    # Validate embedding shape and type
    if not isinstance(embedding, np.ndarray):
        raise TypeError("Embedding must be a numpy ndarray.")
    if embedding.ndim != 1:
        raise ValueError("Embedding must be a 1D numpy array.")
    if embedding.shape[0] != 768:
        raise ValueError(f"Embedding must have dimension 768, got {embedding.shape[0]}")
    try:
        results = collection.search(
            data=[embedding.tolist()],
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=top_k,
        )
        return results[0]
    except Exception as e:
        print(f"Error searching for similar embeddings in Milvus: {e}")
        raise

# Simple in-memory database for storing key-value pairs
simple_db = {}

def database_node(state: GraphState) -> Dict:
    """A simple database node that handles read and write operations to an in-memory store.
    
    Args:
        state (GraphState): The current state containing db_action, db_key, and db_value
        
    Returns:
        Dict: A dictionary containing the result of the operation and current state
    """
    action = state.get("db_action")
    key = state.get("db_key")
    value = state.get("db_value")
    result = None

    if action == "write" and key:
        simple_db[key] = value
    elif action == "read" and key:
        result = simple_db.get(key)

    return {
        "db_result": result,
        "current_state": "database"
    }
