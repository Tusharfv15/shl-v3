from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import uvicorn
import sys
from pathlib import Path

# Add parent directory to path so we can import our modules
sys.path.append(str(Path(__file__).resolve().parent))

# Import the recommender
from main import SHLRecommender

# Initialize the app
app = FastAPI(
    title="SHL Assessment Recommender API",
    description="API for recommending SHL assessments based on job descriptions",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Initialize recommender
recommender = SHLRecommender()

class QueryModel(BaseModel):
    query: str
    top_k: int = 5
    enhanced: bool = False
    filters: Optional[Dict[str, Any]] = None

@app.post("/recommend", response_model=List[Dict[str, Any]])
async def recommend(request: QueryModel):
    """
    Get SHL assessment recommendations based on a text query
    
    Args:
        request: QueryModel containing the query and options
    
    Returns:
        List of assessment recommendations
    """
    try:
        results = recommender.recommend(
            query=request.query,
            top_k=request.top_k,
            enhanced=request.enhanced,
            filters=request.filters
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/recommend", response_model=List[Dict[str, Any]])
async def recommend_get(
    query: str = Query(..., description="Job description or requirements"),
    top_k: int = Query(5, description="Number of recommendations to return"),
    enhanced: bool = Query(False, description="Whether to use enhanced query processing with GPT")
):
    """
    Get SHL assessment recommendations based on a text query (GET method)
    
    Args:
        query: The job description or requirements
        top_k: Number of recommendations to return
        enhanced: Whether to use enhanced query processing with GPT
    
    Returns:
        List of assessment recommendations
    """
    try:
        results = recommender.recommend(
            query=query,
            top_k=top_k,
            enhanced=enhanced
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend-from-url", response_model=List[Dict[str, Any]])
async def recommend_from_url(request: dict):
    """
    Get SHL assessment recommendations based on a job description URL
    
    Args:
        request: Dictionary containing the URL and options
    
    Returns:
        List of assessment recommendations
    """
    try:
        url = request.get("url")
        if not url:
            raise HTTPException(status_code=400, detail="URL is required")
        
        top_k = request.get("top_k", 5)
        enhanced = request.get("enhanced", False)
        filters = request.get("filters")
        
        results = recommender.recommend_from_url(
            url=url,
            top_k=top_k,
            enhanced=enhanced,
            filters=filters
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "Welcome to the SHL Assessment Recommender API",
        "endpoints": {
            "/recommend": "POST or GET to get recommendations based on text",
            "/recommend-from-url": "POST to get recommendations based on a URL"
        }
    }

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 