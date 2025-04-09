import os
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
import numpy as np
import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny

class QdrantVectorStore:
    """Vector store implementation using Qdrant"""
    
    def __init__(self, collection_name: str = "shl_assessments", vector_size: int = 1536):
        """
        Initialize the Qdrant vector store
        
        Args:
            collection_name: Name of the collection to use
            vector_size: Size of the embedding vectors
        """
        # Try to get Qdrant Cloud credentials from environment or secrets
        qdrant_url = None
        qdrant_api_key = None
        
        # Check environment variables
        if "QDRANT_URL" in os.environ and "QDRANT_API_KEY" in os.environ:
            qdrant_url = os.environ["QDRANT_URL"]
            qdrant_api_key = os.environ["QDRANT_API_KEY"]
        # Check Streamlit secrets
        elif hasattr(st, 'secrets') and "QDRANT_URL" in st.secrets and "QDRANT_API_KEY" in st.secrets:
            qdrant_url = st.secrets["QDRANT_URL"]
            qdrant_api_key = st.secrets["QDRANT_API_KEY"]
        
        # Initialize Qdrant client - cloud if credentials exist, local otherwise
        if qdrant_url and qdrant_api_key:
            # Connect to Qdrant Cloud
            self.client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key
            )
            self.using_cloud = True
            print("Using Qdrant Cloud for vector storage")
        else:
            # Fallback to local mode
            data_dir = Path(os.path.join(os.path.dirname(os.path.dirname(__file__)), "qdrant_data"))
            data_dir.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(data_dir))
            self.using_cloud = False
            print("Using local storage for vectors (Qdrant Cloud credentials not found)")
        
        self.collection_name = collection_name
        self.vector_size = vector_size
        
        # Create collection if it doesn't exist
        try:
            self.client.get_collection(collection_name=collection_name)
        except Exception:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
    
    def add_vectors(self, vectors: List[List[float]], payloads: List[Dict[str, Any]]) -> None:
        """
        Add vectors and their payloads to the collection
        
        Args:
            vectors: List of embedding vectors
            payloads: List of payloads (metadata) for each vector
        """
        # Convert vectors to numpy array for validation
        vectors_np = np.array(vectors)
        
        # Validate the vector dimensions
        if vectors_np.shape[1] != self.vector_size:
            raise ValueError(
                f"Vector size mismatch. Expected {self.vector_size}, got {vectors_np.shape[1]}"
            )
        
        # Create points to add
        points = [
            models.PointStruct(
                id=i,
                vector=vector if isinstance(vector, list) else vector.tolist(),
                payload=payload
            )
            for i, (vector, payload) in enumerate(zip(vectors, payloads))
        ]
        
        # Add points to the collection
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def search(self, query_vector: List[float], limit: int = 10, 
               filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar vectors
        
        Args:
            query_vector: The query embedding vector
            limit: Maximum number of results to return
            filters: Dictionary of filters to apply
            
        Returns:
            List of assessment dictionaries
        """
        # Prepare search conditions
        search_params = {
            "collection_name": self.collection_name,
            "query_vector": query_vector,
            "limit": limit,
            "with_payload": True
        }
        
        # Add filters if provided
        if filters and len(filters) > 0:
            qdrant_filter = self._build_filter(filters)
            search_params["query_filter"] = qdrant_filter
        
        # Perform search
        results = self.client.search(**search_params)
        
        # Extract and format the results
        formatted_results = []
        for res in results:
            payload = res.payload.copy()
            payload["relevance_score"] = res.score
            formatted_results.append(payload)
        
        return formatted_results
    
    def _build_filter(self, filters: Dict[str, Any]) -> Filter:
        """
        Build Qdrant filter from filter dictionary
        
        Args:
            filters: Dictionary of filters
            
        Returns:
            Qdrant Filter object
        """
        must_conditions = []
        
        for key, value in filters.items():
            if key == "test_type" and isinstance(value, list):
                # For test types, we need to check if any of the specified types
                # matches any in the assessment's test types
                if len(value) > 0:
                    must_conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchAny(any=value)
                        )
                    )
            else:
                # For other filters, we need an exact match
                must_conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
        
        return Filter(must=must_conditions) 