import streamlit as st
import os
import sys
import pandas as pd
import json
import numpy as np
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import openai
from openai import OpenAI

# Setup page configuration
st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize OpenAI client - first check Streamlit secrets, then environment
openai_api_key = None
if 'OPENAI_API_KEY' in st.secrets:
    openai_api_key = st.secrets['OPENAI_API_KEY']
elif 'OPENAI_API_KEY' in os.environ:
    openai_api_key = os.environ['OPENAI_API_KEY']

if not openai_api_key:
    st.error("OpenAI API key not found. Please add it to your Streamlit secrets or environment variables.")
    st.stop()

# Initialize OpenAI client
client = OpenAI(api_key=openai_api_key)

#--------------------
# Utility Functions
#--------------------

def get_embedding(text: str, model: str = "text-embedding-ada-002") -> List[float]:
    """Get embedding for a text string"""
    if not isinstance(text, str):
        text = str(text)
    
    # Truncate long texts to the model's context limit
    max_tokens = 8000
    if len(text.split()) > max_tokens:
        text = " ".join(text.split()[:max_tokens])
    
    try:
        response = client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error getting embedding: {e}")
        return [0.0] * 1536  # Return a zero vector in case of error

#--------------------
# Sample Data
#--------------------

# Sample assessment data - in a real app, this would come from a database
SAMPLE_ASSESSMENTS = [
    {
        "name": "Account Manager Solution",
        "category": "Individual Test Solutions",
        "description": "The Account Manager solution is an assessment used for job candidates applying to mid-level leadership positions that tend to manage the day-to-day operations and activities of client accounts. Sample tasks for these jobs include, but are not limited to: communicating with clients about project status, developing and maintaining project plans, coordinating internally with appropriate project personnel, and ensuring client expectations are being met.",
        "job_levels": "Mid-Professional",
        "languages": "English (USA)",
        "assessment_length": "49",
        "remote_testing": "Yes",
        "adaptive_irt": "Yes",
        "test_type": "Competencies, Personality & Behavior, Ability & Aptitude, Biodata & Situational Judgement",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/account-manager-solution/"
    },
    {
        "name": "Sales Manager Solution",
        "category": "Individual Test Solutions",
        "description": "The Sales Manager solution is designed to assess candidates for sales management roles that focus on leading and developing a team of sales professionals. Key responsibilities include setting sales targets, training sales staff, analyzing performance data, and developing strategies to increase sales and customer satisfaction.",
        "job_levels": "Mid-Professional",
        "languages": "English (USA)",
        "assessment_length": "55",
        "remote_testing": "Yes",
        "adaptive_irt": "Yes",
        "test_type": "Competencies, Personality & Behavior, Ability & Aptitude",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/sales-manager-solution/"
    },
    {
        "name": "Customer Service Solution",
        "category": "Individual Test Solutions",
        "description": "The Customer Service solution is designed to identify candidates who will excel in customer-facing roles. This assessment evaluates skills such as active listening, problem-solving, communication, and the ability to handle difficult customers with patience and professionalism.",
        "job_levels": "Entry-Level",
        "languages": "Multiple",
        "assessment_length": "35",
        "remote_testing": "Yes",
        "adaptive_irt": "No",
        "test_type": "Competencies, Personality & Behavior, Situational Judgement",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/customer-service-solution/"
    },
    {
        "name": "Project Management Solution",
        "category": "Individual Test Solutions",
        "description": "The Project Management solution helps identify candidates with strong organizational and leadership skills necessary for managing complex projects. It assesses abilities in planning, coordination, resource allocation, risk management, and team leadership.",
        "job_levels": "Mid-Professional",
        "languages": "English (USA)",
        "assessment_length": "60",
        "remote_testing": "Yes",
        "adaptive_irt": "Yes",
        "test_type": "Competencies, Ability & Aptitude, Situational Judgement",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/project-management-solution/"
    },
    {
        "name": "Technical Professional Solution",
        "category": "Individual Test Solutions",
        "description": "The Technical Professional solution is designed for roles that require specialized technical knowledge and analytical thinking. This assessment measures technical aptitude, problem-solving abilities, and the capacity to learn and adapt to new technologies.",
        "job_levels": "Professional",
        "languages": "English (USA)",
        "assessment_length": "45",
        "remote_testing": "Yes",
        "adaptive_irt": "No",
        "test_type": "Technical Skills, Cognitive Ability, Problem Solving",
        "url": "https://www.shl.com/solutions/products/product-catalog/view/technical-professional-solution/"
    }
]

# Pre-compute embeddings for sample data
SAMPLE_EMBEDDINGS = []
for assessment in SAMPLE_ASSESSMENTS:
    combined_text = (
        assessment['name'] + '. ' + 
        assessment['category'] + '. ' + 
        assessment['description'] + '. ' + 
        'Job levels: ' + assessment['job_levels'] + '. ' +
        'Test type: ' + assessment['test_type'] + '.'
    )
    SAMPLE_EMBEDDINGS.append(combined_text)

#--------------------
# Recommender Logic
#--------------------

def recommend(query: str, top_k: int = 5, enhanced: bool = False, 
              filters: Optional[Dict[str, Any]] = None) -> List[Dict]:
    """Recommend assessments based on a text query"""
    # Get the embedding for the query
    query_embedding = get_embedding(query)
    
    # Calculate similarity scores with all sample assessments
    results = []
    for i, assessment in enumerate(SAMPLE_ASSESSMENTS):
        # Skip if it doesn't match filters
        if filters:
            skip = False
            for key, value in filters.items():
                if key == "test_type" and isinstance(value, list):
                    # For test types, check if any match
                    assessment_types = assessment["test_type"].split(", ")
                    if not any(t in assessment_types for t in value):
                        skip = True
                        break
                elif key in assessment and assessment[key] != value:
                    skip = True
                    break
            if skip:
                continue
                
        # Get the embedding for the assessment
        assessment_embedding = get_embedding(SAMPLE_EMBEDDINGS[i])
        
        # Calculate cosine similarity
        similarity = np.dot(query_embedding, assessment_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(assessment_embedding)
        )
        
        # Add to results
        assessment_copy = assessment.copy()
        assessment_copy["relevance_score"] = float(similarity)
        results.append(assessment_copy)
    
    # Sort by similarity score and take top_k
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return results[:top_k]

def recommend_from_url(url: str, top_k: int = 5, enhanced: bool = False,
                      filters: Optional[Dict[str, Any]] = None) -> List[Dict]:
    """Recommend assessments based on a job description URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        text_content = response.text
        
        # Basic HTML tag removal and cleanup
        from html import unescape
        import re
        text_content = re.sub(r'<[^>]+>', ' ', text_content)
        text_content = unescape(text_content)
        text_content = re.sub(r'\s+', ' ', text_content).strip()
        
        # Use the extracted text for recommendations
        return recommend(text_content, top_k, enhanced, filters)
        
    except Exception as e:
        st.error(f"Error fetching job description from URL: {e}")
        return []

#--------------------
# Streamlit App
#--------------------

def main():
    # Custom CSS
    st.markdown("""
    <style>
    .assessment-card {
        background-color: #f5f5f5;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 15px;
    }
    .assessment-title {
        color: #1e3a8a;
        font-weight: bold;
    }
    .assessment-metadata {
        color: #666;
        font-size: 0.9em;
    }
    .assessment-description {
        margin-top: 10px;
        color: #333333;
        font-size: 0.95em;
    }
    .badge {
        display: inline-block;
        padding: 3px 7px;
        border-radius: 10px;
        font-size: 12px;
        font-weight: bold;
        margin-right: 5px;
        background-color: #e2e8f0;
    }
    .yes-badge {
        background-color: #d1fae5;
        color: #065f46;
    }
    .no-badge {
        background-color: #fee2e2;
        color: #991b1b;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")
        
        st.subheader("Filter Options")
        remote_testing = st.selectbox(
            "Remote Testing Support", 
            options=["Any", "Yes", "No"],
            index=0
        )
        
        adaptive_testing = st.selectbox(
            "Adaptive/IRT Support", 
            options=["Any", "Yes", "No"],
            index=0
        )
        
        test_types = st.multiselect(
            "Test Types",
            options=["Competencies", "Personality & Behavior", "Ability & Aptitude", 
                    "Biodata & Situational Judgement", "Technical Skills", 
                    "Cognitive Ability", "Problem Solving"],
            default=[]
        )
        
        advanced_options = st.expander("Advanced Options")
        with advanced_options:
            enhanced_mode = st.checkbox("Use Enhanced Mode (GPT augmented)", value=True)
            top_k = st.slider("Number of recommendations", 1, 5, 3)
        
        st.divider()
        st.markdown("### About")
        st.markdown("""
        This tool helps you find the most suitable SHL assessments based on your requirements.
        
        Simply enter a job description or your requirements in the search box.
        
        Note: This is a demo with sample data.
        """)
    
    # Main content
    st.title("SHL Assessment Recommender 📋")
    
    # Input area
    input_method = st.radio(
        "Input method",
        options=["Text Description", "URL to Job Description"],
        horizontal=True
    )
    
    if input_method == "Text Description":
        query = st.text_area(
            "Enter job description or requirements",
            height=150,
            placeholder="Example: We are looking for a mid-level account manager who can manage client relationships and coordinate with internal teams..."
        )
        url = None
    else:
        url = st.text_input(
            "Enter URL to job description",
            placeholder="https://example.com/job-description"
        )
        query = None
    
    # Get filter settings
    filters = {}
    if remote_testing != "Any":
        filters["remote_testing"] = remote_testing
    if adaptive_testing != "Any":
        filters["adaptive_irt"] = adaptive_testing
    if test_types:
        filters["test_type"] = test_types
    
    # Process the query when button is clicked
    if st.button("Get Recommendations", type="primary", use_container_width=True):
        if not query and not url:
            st.warning("Please enter a job description or URL")
        else:
            with st.spinner("Finding the best assessments for you..."):
                try:
                    # Get recommendations
                    if url:
                        results = recommend_from_url(
                            url=url,
                            top_k=top_k,
                            enhanced=enhanced_mode,
                            filters=filters if filters else None
                        )
                    else:
                        results = recommend(
                            query=query,
                            top_k=top_k,
                            enhanced=enhanced_mode,
                            filters=filters if filters else None
                        )
                    
                    # Display results
                    st.subheader(f"Top {len(results)} Recommended Assessments")
                    
                    if not results:
                        st.info("No assessments match your criteria. Try adjusting your filters.")
                    
                    for i, assessment in enumerate(results, 1):
                        with st.container():
                            st.markdown(f"""
                            <div class="assessment-card">
                                <div class="assessment-title">{i}. {assessment['name']}</div>
                                <div class="assessment-metadata">
                                    <span class="badge">Length: {assessment['assessment_length']} min</span>
                                    <span class="badge {'yes-badge' if assessment['remote_testing'] == 'Yes' else 'no-badge'}">
                                        Remote Testing: {assessment['remote_testing']}
                                    </span>
                                    <span class="badge {'yes-badge' if assessment['adaptive_irt'] == 'Yes' else 'no-badge'}">
                                        Adaptive: {assessment['adaptive_irt']}
                                    </span>
                                    <span class="badge">Type: {assessment['test_type']}</span>
                                </div>
                                <div class="assessment-description">
                                    {assessment['description'][:300]}{"..." if len(assessment['description']) > 300 else ""}
                                </div>
                                <div style="margin-top: 10px;">
                                    <a href="{assessment['url']}" target="_blank">View in SHL Catalog →</a>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Error getting recommendations: {e}")
    
    # Show sample data
    with st.expander("Sample Assessment Data"):
        df = pd.DataFrame(SAMPLE_ASSESSMENTS)
        st.dataframe(df)

if __name__ == "__main__":
    main() 