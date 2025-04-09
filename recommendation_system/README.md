# SHL Assessment Recommender System

A semantic search and recommendation system that helps match job requirements with appropriate SHL assessment tools.

## Overview

This system uses OpenAI's embedding models to vectorize SHL assessment descriptions and then uses vector similarity search to find the most relevant assessments for a given job description or requirement query.

## Key Features

- **Semantic Search**: Find assessments based on meaning, not just keywords
- **Filtering**: Filter assessments by remote testing support, adaptive/IRT capabilities, and test types
- **Enhanced Recommendations**: Option to use GPT to refine recommendations (when enabled)
- **Admin Functions**: Build embeddings directly from the Streamlit interface
- **Qdrant Cloud Integration**: Store vectors in a managed cloud service for better deployment

## Setting Up Qdrant Cloud

### 1. Create a Qdrant Cloud Account

- Go to [https://cloud.qdrant.io/](https://cloud.qdrant.io/)
- Sign up for an account
- Create a new cluster (the free tier is sufficient for small to medium datasets)
- Get your cluster URL and API key

### 2. Configure Environment Variables

Add your Qdrant Cloud credentials to the `.env` file or Streamlit secrets:

```
OPENAI_API_KEY=your_openai_api_key
QDRANT_URL=https://your-cluster-id.region.gcp.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
```

### 3. Create a Collection

The system will automatically create a collection named "shl_assessments" when you build embeddings. If you want to manually create a collection:

```python
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

client = QdrantClient(
    url="https://your-cluster-id.region.gcp.cloud.qdrant.io",
    api_key="your_qdrant_api_key",
)

client.recreate_collection(
    collection_name="shl_assessments",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)
```

## Running Locally

1. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

2. Build embeddings:

   ```
   python build_embeddings.py --data-file data/your_assessments.csv
   ```

3. Run the Streamlit app:
   ```
   cd streamlit_app
   streamlit run app.py
   ```

## Deploying to Streamlit Cloud

1. Fork/clone this repository
2. Add the following secrets in the Streamlit Cloud dashboard:
   - `OPENAI_API_KEY`
   - `QDRANT_URL`
   - `QDRANT_API_KEY`
3. Set the main file path to: `recommendation_system/streamlit_app/app.py`
4. Deploy the app
5. Use the Admin Functions in the sidebar to upload your assessment data and build embeddings

## Using the Admin Interface

1. In the Streamlit app sidebar, expand the "Admin Functions" section
2. Upload your SHL assessments CSV file
3. Click "Build Embeddings" to process the data and store it in Qdrant Cloud
4. Once complete, you can use the search functionality to find assessments

## Files and Components

- `build_embeddings.py` - Script for creating and storing assessment embeddings
- `main.py` - Core recommendation logic and SHLRecommender class
- `streamlit_app/app.py` - Streamlit web interface
- `utils/` - Helper functions for data processing, vectorization, and vector storage
- `models/` - Recommendation model implementations
- `data/` - Sample assessment data

## Why Qdrant Cloud?

Qdrant Cloud offers several advantages for this application:

1. **Persistent Storage**: Data remains accessible between Streamlit Cloud deployments
2. **Faster Cold Starts**: The app doesn't need to build a local database at startup
3. **Advanced Filtering**: Combine vector similarity with metadata filters
4. **Scalability**: Easily handle larger assessment catalogs without code changes
5. **Managed Service**: No need to maintain your own vector database infrastructure

## Features

- **Natural Language Querying**: Submit job descriptions or requirements as plain text
- **URL Support**: Process job descriptions directly from URLs
- **Filtering**: Filter assessments by remote testing, adaptive/IRT support, and test types
- **Semantic Search**: Uses OpenAI embeddings for accurate semantic matching
- **Enhanced Mode**: Utilizes GPT for better query understanding and matching
- **Streamlit Web Interface**: Easy-to-use UI for interacting with the system
- **Cloud Vector Storage**: Option to use Qdrant Cloud for vector storage and retrieval (faster deployments)

## System Architecture

The system uses a Retrieval-Augmented Generation (RAG) approach with the following components:

1. **Data Processing**: Cleans and combines assessment attributes into a comprehensive text field
2. **Embedding & Storage**: Generates embeddings using OpenAI's text-embedding-ada-002 model
3. **Vector Storage**: Uses Qdrant for efficient similarity search (local or cloud)
4. **Retrieval System**:
   - Basic Mode: Converts the query to an embedding and performs similarity search
   - Enhanced Mode: Uses GPT to better understand the query before similarity search
5. **Streamlit Interface**: Provides a user-friendly web interface for querying and filtering

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your API keys:

   ```
   OPENAI_API_KEY=your_openai_api_key

   # Optional: Qdrant Cloud credentials (if using cloud storage)
   QDRANT_URL=https://your-cluster-url.qdrant.io
   QDRANT_API_KEY=your_qdrant_api_key
   ```

## Usage

### Building Embeddings

Before using the system, you need to build the embeddings for your assessment data:

```
python main.py build --data-file data/shl_assessments.csv
```

### Using the Web Interface

Launch the Streamlit web interface:

```
python run_app.py
```

This will start a web server at http://localhost:8501 where you can:

1. Enter a job description or URL
2. Set filtering options
3. Get assessment recommendations

### Cloud Deployment

When deploying to Streamlit Cloud:

1. Add your API keys as secrets in the Streamlit Cloud dashboard:

   ```
   OPENAI_API_KEY = "your-openai-api-key"
   QDRANT_URL = "https://your-cluster-url.qdrant.io"
   QDRANT_API_KEY = "your-qdrant-api-key"
   ```

2. Use Qdrant Cloud for vector storage to speed up deployment and avoid local storage issues.

### Command Line Interface

You can also use the system from the command line:

```
# Basic recommendation
python main.py recommend "We need an account manager who can manage client relationships"

# With filters
python main.py recommend "Looking for a sales director" --remote-testing Yes --test-types "Competencies,Personality & Behavior"

# Enhanced mode with GPT
python main.py recommend "Need someone who can lead a team of account managers" --enhanced

# Save results to file
python main.py recommend "Technical project manager position" --output results.json
```

## Evaluation

The system includes evaluation metrics to measure performance:

```
python main.py evaluate --queries-file data/sample_test_queries.json
```

This calculates:

- Mean Recall@K: Measures how many relevant assessments are found
- MAP@K: Evaluates both relevance and ranking order of assessments

## Project Structure

```
recommendation_system/
├── data/                      # Assessment data files
├── models/                    # Recommender models
├── utils/                     # Utility modules for data processing, embeddings, etc.
├── streamlit_app/             # Streamlit web interface
│   └── app.py                 # Streamlit application
├── main.py                    # Main module with CLI interface
├── build_embeddings.py        # Script to build embeddings
├── evaluate_recommender.py    # Evaluation module
├── run_app.py                 # Script to run the Streamlit app
├── test_recommender.py        # Test script
└── requirements.txt           # Python dependencies
```
