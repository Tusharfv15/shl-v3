# SHL Assessment Recommender API

This document provides instructions on how to use the SHL Assessment Recommender API.

## Running the API Server

1. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Start the API server:

   ```bash
   python run_api.py
   ```

   This will start the server at `http://localhost:8000`. The API documentation will be available at `http://localhost:8000/docs`.

## API Endpoints

### 1. Get Recommendations from Text

**Endpoint:** `/recommend`

#### Using GET Request

```
GET /recommend?query=Looking for Java developers with business collaboration skills&top_k=5&enhanced=true
```

Parameters:

- `query` (required): The job description or requirements text
- `top_k` (optional, default=5): Number of recommendations to return
- `enhanced` (optional, default=false): Whether to use enhanced query processing with GPT

#### Using POST Request

```
POST /recommend
```

Request Body:

```json
{
  "query": "Looking for Java developers with business collaboration skills",
  "top_k": 5,
  "enhanced": true,
  "filters": {
    "remote_testing": "Yes",
    "adaptive_irt": "No",
    "test_type": ["Knowledge & Skills"]
  }
}
```

### 2. Get Recommendations from URL

**Endpoint:** `/recommend-from-url`

```
POST /recommend-from-url
```

Request Body:

```json
{
  "url": "https://example.com/job-posting",
  "top_k": 5,
  "enhanced": true,
  "filters": {
    "remote_testing": "Yes"
  }
}
```

## Example Usage with Python

```python
import requests
import json

# Base URL of the API
API_URL = "http://localhost:8000"

# Example 1: Get recommendations using GET request
query_text = "Looking for Java developers with business collaboration skills"
response = requests.get(f"{API_URL}/recommend", params={
    "query": query_text,
    "top_k": 5,
    "enhanced": True
})
results = response.json()
print(json.dumps(results, indent=2))

# Example 2: Get recommendations using POST request with filters
payload = {
    "query": query_text,
    "top_k": 5,
    "enhanced": True,
    "filters": {
        "remote_testing": "Yes",
        "test_type": ["Knowledge & Skills"]
    }
}
response = requests.post(f"{API_URL}/recommend", json=payload)
results = response.json()
print(json.dumps(results, indent=2))

# Example 3: Get recommendations from a URL
url_payload = {
    "url": "https://example.com/job-posting",
    "top_k": 5,
    "enhanced": True
}
response = requests.post(f"{API_URL}/recommend-from-url", json=url_payload)
results = response.json()
print(json.dumps(results, indent=2))
```

## Using with JavaScript/Fetch

```javascript
// Example: Get recommendations using fetch
const query = "Looking for Java developers with business collaboration skills";
const apiUrl = "http://localhost:8000/recommend";

// Using query parameters (GET)
fetch(`${apiUrl}?query=${encodeURIComponent(query)}&top_k=5&enhanced=true`)
  .then((response) => response.json())
  .then((data) => console.log(data))
  .catch((error) => console.error("Error:", error));

// Using POST with JSON body
const payload = {
  query: query,
  top_k: 5,
  enhanced: true,
  filters: {
    remote_testing: "Yes",
  },
};

fetch(apiUrl, {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify(payload),
})
  .then((response) => response.json())
  .then((data) => console.log(data))
  .catch((error) => console.error("Error:", error));
```

## Using with cURL

```bash
# GET request
curl -X GET "http://localhost:8000/recommend?query=Looking%20for%20Java%20developers&top_k=5&enhanced=false"

# POST request
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"query":"Looking for Java developers","top_k":5,"enhanced":true}'
```

## Response Format

The API returns a JSON array of assessment recommendations:

```json
[
  {
    "name": "Core Java (Entry Level) (New)",
    "url": "https://example.com/assessment/1",
    "remote_testing": "Yes",
    "adaptive_irt": "No",
    "assessment_length": "13 min",
    "test_type": "Knowledge & Skills",
    "description": "Multi-choice test that measures the knowledge of basic Java constructs...",
    "relevance_score": 0.8765
  },
  {
    "name": "Core Java (Advanced Level) (New)",
    "url": "https://example.com/assessment/2",
    "remote_testing": "Yes",
    "adaptive_irt": "No",
    "assessment_length": "13 min",
    "test_type": "Knowledge & Skills",
    "description": "Multi-choice test that measures the knowledge of advanced Java concepts...",
    "relevance_score": 0.7654
  },
  ...
]
```

## Integration with Streamlit

You can call this API from your Streamlit app using the `requests` library:

```python
import requests
import streamlit as st

# API base URL
api_url = "http://localhost:8000"

# Get query from user input
query = st.text_area("Enter job description")

if st.button("Get Recommendations"):
    response = requests.get(f"{api_url}/recommend", params={
        "query": query,
        "top_k": 5,
        "enhanced": True
    })

    if response.status_code == 200:
        results = response.json()
        for i, result in enumerate(results, 1):
            st.subheader(f"{i}. {result['name']}")
            st.write(f"Score: {result['relevance_score']:.4f}")
            st.write(f"Type: {result['test_type']}")
            st.write(f"Duration: {result['assessment_length']}")
            st.write(f"Remote Testing: {result['remote_testing']}")
    else:
        st.error(f"Error: {response.text}")
```
