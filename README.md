# SHL Assessment Recommender

A semantic search and recommendation system for SHL assessments that helps match job requirements with appropriate assessment tools.

## Tech Stack

### Core Technologies

- **Python**: Primary programming language
- **OpenAI Embeddings**: For generating vector representations of assessments
- **Qdrant Vector Database**: For storing and retrieving vectors with semantic search
- **Streamlit**: For the web application interface

### Why Qdrant Vector Store?

Qdrant was selected for several key reasons:

1. **Advanced Filtering**: Supports complex filtering operations combined with vector similarity search
2. **High Performance**: Optimized for fast retrieval with minimal latency
3. **Cloud Hosting**: Qdrant Cloud provides persistent storage between app deployments
4. **Scalability**: Can handle growing assessment catalogs efficiently
5. **Simple API**: Clean Python client that integrates well with our stack

### GPT Refinement (Enhanced Mode)

The system offers an "Enhanced Mode" that uses GPT to refine recommendations:

1. **Query Understanding**: GPT analyzes job descriptions to extract key skills, competencies, and requirements
2. **Context-Aware Search**: Improves the relevance of search results by focusing on job-critical attributes
3. **Handling Ambiguity**: Better manages unclear or broadly-defined job requirements
4. **Domain Knowledge**: Leverages GPT's understanding of HR terminology and assessment purposes
5. **Explanation Generation**: Can provide reasoning for why specific assessments were recommended

This approach combines the strengths of both vector search (fast, scalable retrieval) and large language models (contextual understanding, domain knowledge) to deliver more accurate recommendations, especially for complex or specialized job roles.

### URL to Job Description Feature

The system provides two input methods - direct text entry or URL to a job posting:

1. **Web Scraping**: Automatically extracts job descriptions from provided URLs
2. **HTML Parsing**: Intelligently identifies and extracts the relevant job information from various site structures
3. **Text Cleaning**: Removes irrelevant elements like navigation, footers, and advertisements
4. **Content Extraction**: Focuses on job requirements, responsibilities, and qualifications
5. **Error Handling**: Gracefully manages connectivity issues or inaccessible websites

This feature allows users to simply paste a job posting URL rather than manually copying and pasting content, making the workflow more efficient. The system handles extraction from popular job posting sites like LinkedIn, Indeed, and company career pages.

### SHL Catalog Scraper

The project includes a custom scraper (`shl_scraper.py`) that extracts assessment data directly from the SHL product catalog:

1. **Automated Data Collection**: Collects comprehensive information about SHL assessments
2. **Metadata Extraction**: Extracts key details like assessment types, remote testing support, and adaptive capabilities
3. **Description Processing**: Captures detailed descriptions and specifications for each assessment
4. **Data Formatting**: Organizes the scraped data into a structured CSV format
5. **Complete Coverage**: Includes both Individual Test Solutions and Pre-packaged Job Solutions

The resulting dataset (`shl_assessments.csv`) is stored in the `data` folder inside the `recommendation system folder` and can also be found in the root directory of this project and serves as the foundation for the recommendation system. This approach ensures the system has accurate and up-to-date information about SHL's assessment offerings without manual data entry.

### Additional Libraries

- **pandas**: For data processing and manipulation
- **numpy**: For numerical operations
- **python-dotenv**: For environment variable management
- **tqdm**: For progress tracking during embedding generation
- **requests**: For fetching web content from URLs
- **BeautifulSoup**: For parsing HTML and extracting job descriptions

## API-Endpoint result
- git clone the app
- pip install -r requirements.txt
- python run_api.py
- server running on `localhost:8000`
- Result on Postman
  ![image](https://github.com/user-attachments/assets/bc6715c9-7258-4fe3-947f-bd7fd20d2e7b)


## Final Evaluation Results
1) **Test Queries**: Refer `recommendation_system/data/sample_test_queries.json`
2) **Mean Recall@5**: `0.6765`
3) **Mean MAP@5**: `8.8809`
![Screenshot 2025-04-06 215212](https://github.com/user-attachments/assets/fbb99c70-ba66-4065-9c00-4d2c8459d3d1)

## Workflow

### Data Processing Workflow

1. **Data Collection**: Scrape assessment data from SHL catalog using `shl_scraper.py`
2. **Data Ingestion**: Load assessment data from the generated CSV file
3. **Text Preparation**: Clean and combine assessment data into searchable text
4. **Embedding Generation**: Create vector embeddings using OpenAI's API
5. **Vector Storage**: Store vectors and metadata in Qdrant Cloud

### Recommendation Workflow

1. **Query Processing**: Vectorize the user's job description query (from text or URL)
2. **Vector Search**: Find semantically similar assessments in Qdrant
3. **Filter Application**: Apply user-specified filters (remote testing, adaptive testing, test types)
4. **Result Ranking**: Return the most relevant assessments
5. **Enhanced Results** (optional): Use GPT to further refine recommendations




### Deployment Workflow

1. **Local Development**: Run the app locally with `streamlit run app.py`
2. **Cloud Deployment**: Deploy to Streamlit Cloud with environment secrets
3. **Vector Database**: Qdrant Cloud hosts the vector database separately from the app
4. **Admin Functions**: Build or update embeddings directly from the Streamlit interface

## Folder Structure

```
recommendation_system/
├── data/                     # Assessment data files
│   └── shl_assessments.csv   # Scraped assessment data
├── models/                   # Recommender model implementations
├── streamlit_app/            # Streamlit web application
│   ├── app.py                # Main Streamlit application
│   └── .env                  # Environment variables for local development
├── utils/                    # Utility functions
│   ├── data_processor.py     # Data processing utilities
│   ├── vector_store.py       # Vector database interface
│   └── vectorize.py          # Embedding generation functions
├── build_embeddings.py       # Script for building assessment embeddings
├── evaluate_recommender.py   # Evaluation scripts
├── main.py                   # Core recommendation system
├── requirements.txt          # Project dependencies
├── test_recommender.py       # Test scripts
└── shl_scraper.py            # Script for scraping SHL assessment catalog
```

## Getting Started

### Prerequisites

- Python 3.8+
- OpenAI API key
- Qdrant Cloud account (or use local Qdrant instance)

### Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Create a `.env` file in the `streamlit_app` directory with your API keys:

```
OPENAI_API_KEY=your_openai_api_key
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
```

4. Build embeddings: `python build_embeddings.py --data-file data/your_assessments.csv`
5. Run the app: `cd streamlit_app && streamlit run app.py`

### Deployment to Streamlit Cloud

1. Add secrets in the Streamlit Cloud dashboard:
   - `OPENAI_API_KEY`
   - `QDRANT_URL`
   - `QDRANT_API_KEY`
2. Set the main file path to: `recommendation_system/streamlit_app/app.py`
3. Use the Admin Functions in the deployed app to build embeddings
