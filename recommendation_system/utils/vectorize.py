import os
from typing import List, Union
import numpy as np
import openai
from openai import OpenAI
from tqdm import tqdm
import time

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text: str, model: str = "text-embedding-ada-002") -> List[float]:
    """
    Get embedding for a single text string
    
    Args:
        text: Text to embed
        model: OpenAI embedding model to use
        
    Returns:
        List of embedding values
    """
    # Ensure the text is a string
    if not isinstance(text, str):
        text = str(text)
    
    # Truncate long texts to the model's context limit
    # text-embedding-ada-002 has an 8191 token limit
    max_tokens = 8000  # Setting a bit below the limit to be safe
    if len(text.split()) > max_tokens:
        text = " ".join(text.split()[:max_tokens])
    
    try:
        response = client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        # Return a zero vector of the expected size in case of error
        return [0.0] * 1536  # text-embedding-ada-002 produces 1536-dimensional vectors

def batch_get_embeddings(texts: List[str], model: str = "text-embedding-ada-002", 
                         batch_size: int = 100) -> List[List[float]]:
    """
    Get embeddings for a batch of texts
    
    Args:
        texts: List of texts to embed
        model: OpenAI embedding model to use
        batch_size: Number of texts to process in each batch
        
    Returns:
        List of embedding vectors
    """
    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        # Get the current batch
        batch = texts[i:i+batch_size]
        
        try:
            # Get embeddings for the batch
            response = client.embeddings.create(
                model=model,
                input=batch
            )
            
            # Extract embeddings from the response
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
            
            # Add a small delay to respect API rate limits
            if i + batch_size < len(texts):
                time.sleep(0.5)
                
        except Exception as e:
            print(f"Error in batch {i//batch_size + 1}: {e}")
            # Add zero vectors for this batch in case of error
            zero_vector = [0.0] * 1536  # text-embedding-ada-002 produces 1536-dimensional vectors
            all_embeddings.extend([zero_vector] * len(batch))
    
    return all_embeddings 