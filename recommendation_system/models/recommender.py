import os
import json
from typing import Dict, List, Optional, Any, Union
import openai
from openai import OpenAI
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from utils.vectorize import get_embedding
from utils.vector_store import QdrantVectorStore

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class BaseRecommender:
    """Base recommender class for SHL assessments"""
    
    def __init__(self, vector_store):
        """
        Initialize the recommender
        
        Args:
            vector_store: Vector store instance for retrieving assessments
        """
        self.vector_store = vector_store
    
    def process_query(self, query: str, remote_testing: Optional[str] = None,
                     adaptive_irt: Optional[str] = None, test_types: Optional[List[str]] = None,
                     limit: int = 10) -> List[Dict]:
        """
        Process a query and return relevant assessments
        
        Args:
            query: The query string
            remote_testing: Filter for remote testing support ("Yes" or "No")
            adaptive_irt: Filter for adaptive/IRT support ("Yes" or "No")
            test_types: List of test types to include
            limit: Maximum number of results to return
            
        Returns:
            List of assessment dictionaries
        """
        # Get the embedding for the query
        query_embedding = get_embedding(query)
        
        # Prepare filters for the vector search
        filters = self._prepare_filters(remote_testing, adaptive_irt, test_types)
        
        # Search for similar assessments
        results = self.vector_store.search(
            query_embedding, 
            limit=limit,
            filters=filters
        )
        
        return results
    
    def enhanced_recommendations(self, query: str, job_description_url: Optional[str] = None,
                               remote_testing: Optional[str] = None, adaptive_irt: Optional[str] = None,
                               test_types: Optional[List[str]] = None, limit: int = 10) -> List[Dict]:
        """
        Use GPT to enhance the query processing
        
        Args:
            query: The query string
            job_description_url: Optional URL to a job description
            remote_testing: Filter for remote testing support ("Yes" or "No")
            adaptive_irt: Filter for adaptive/IRT support ("Yes" or "No")
            test_types: List of test types to include
            limit: Maximum number of results to return
            
        Returns:
            List of assessment dictionaries
        """
        # Generate an enhanced query using GPT
        enhanced_query = self._enhance_query_with_gpt(query, job_description_url)
        
        # Process the enhanced query
        return self.process_query(
            enhanced_query,
            remote_testing=remote_testing,
            adaptive_irt=adaptive_irt,
            test_types=test_types,
            limit=limit
        )
    
    def _enhance_query_with_gpt(self, query: str, job_description_url: Optional[str] = None) -> str:
        """
        Use GPT to extract key skills and requirements from the query
        
        Args:
            query: The original query
            job_description_url: Optional URL to a job description
            
        Returns:
            Enhanced query string
        """
        # Prepare the prompt for GPT
        if job_description_url:
            prompt = f"""
            I have a job description available at {job_description_url}.
            Based on this job description, extract the key skills, competencies, and requirements.
            Format them as a detailed list that can be used to search for relevant assessments.
            Focus on technical skills, personality traits, competencies, and cognitive abilities.
            """
        else:
            prompt = f"""
            Extract the key skills, competencies, and requirements from this job description or query:
            
            "{query}"
            
            Format them as a detailed list that can be used to search for relevant assessments.
            Focus on technical skills, personality traits, competencies, and cognitive abilities.
            """
        
        # Call the OpenAI API to enhance the query
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts key skills and requirements from job descriptions."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            # Extract the enhanced query from the response
            enhanced_query = response.choices[0].message.content.strip()
            
            # Combine with original query for better results
            return f"{query} {enhanced_query}"
            
        except Exception as e:
            print(f"Error enhancing query with GPT: {e}")
            # Fall back to original query if there's an error
            return query
    
    def _prepare_filters(self, remote_testing: Optional[str] = None,
                        adaptive_irt: Optional[str] = None,
                        test_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Prepare filters for the vector search
        
        Args:
            remote_testing: Filter for remote testing support ("Yes" or "No")
            adaptive_irt: Filter for adaptive/IRT support ("Yes" or "No")
            test_types: List of test types to include
            
        Returns:
            Dictionary of filters
        """
        filters = {}
        
        if remote_testing is not None:
            filters["remote_testing"] = remote_testing
        
        if adaptive_irt is not None:
            filters["adaptive_irt"] = adaptive_irt
        
        if test_types is not None and len(test_types) > 0:
            filters["test_type"] = test_types
        
        return filters 