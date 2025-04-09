import streamlit as st

# IMPORTANT: set_page_config must be the first Streamlit command
st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

import sys
import os
import pandas as pd
import json
from pathlib import Path

# Add parent directory to path so we can import our modules
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# This helps find modules in Streamlit Cloud
sys.path.append("/mount/src/shl-assignment")

# Setup environment before other imports
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load environment variables from .env file
except ImportError:
    # If python-dotenv isn't available, set up OpenAI API key directly
    if "OPENAI_API_KEY" not in os.environ and hasattr(st, 'secrets') and "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    elif "OPENAI_API_KEY" not in os.environ:
        st.error("OpenAI API key not found. Please add it to your environment variables or .streamlit/secrets.toml")
        st.stop()

# Debug information - moved after set_page_config
debug_expander = st.sidebar.expander("Debug Info", expanded=False)
with debug_expander:
    st.write(f"Current working directory: {os.getcwd()}")
    st.write(f"Python path: {sys.path}")

# Import our recommender after env setup
try:
    from main import SHLRecommender
except ImportError as e:
    st.error(f"Error importing SHLRecommender: {e}")
    st.error(f"Trying absolute import...")
    try:
        from recommendation_system.main import SHLRecommender
    except ImportError as e2:
        st.error(f"Still failed: {e2}")
        try:
            # Try one more approach by dynamically finding the main module
            import importlib.util
            import glob
            
            # Look for main.py in nearby directories
            main_files = glob.glob(os.path.join(os.getcwd(), "**/main.py"), recursive=True)
            if main_files:
                # Use the first main.py found
                debug_expander.write(f"Found main.py at: {main_files[0]}")
                
                # Load it as a module
                spec = importlib.util.spec_from_file_location("main_module", main_files[0])
                main_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(main_module)
                
                # Get the SHLRecommender class
                SHLRecommender = main_module.SHLRecommender
            else:
                st.error("Could not find main.py in any nearby directories")
                st.stop()
        except Exception as e3:
            st.error(f"All import attempts failed: {e3}")
            st.error("Falling back to standalone mode...")
            
            # Fall back to standalone implementation 
            from streamlit_app.standalone_app import recommend, recommend_from_url
            
            # Create a minimal SHLRecommender that uses the standalone functions
            class FallbackSHLRecommender:
                def __init__(self):
                    pass
                
                def recommend(self, query, top_k=5, enhanced=False, filters=None):
                    return recommend(query, top_k, enhanced, filters)
                
                def recommend_from_url(self, url, top_k=5, enhanced=False, filters=None):
                    return recommend_from_url(url, top_k, enhanced, filters)
            
            SHLRecommender = FallbackSHLRecommender

# Load sample data to show in the interface
@st.cache_data
def load_sample_data():
    try:
        data_path = os.path.join(parent_dir, "data", "shl_assessments.csv")
        sample_data = pd.read_csv(data_path)
        return sample_data
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        return pd.DataFrame()

# Initialize the recommender
@st.cache_resource
def get_recommender():
    try:
        return SHLRecommender()
    except Exception as e:
        st.error(f"Error initializing recommender: {e}")
        return None

# Main function to run the app
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
            options=["Competencies", "Personality & Behavior", "Ability & Aptitude", "Biodata & Situational Judgement"],
            default=[]
        )
        
        advanced_options = st.expander("Advanced Options")
        with advanced_options:
            enhanced_mode = st.checkbox("Use Enhanced Mode (GPT augmented)", value=True)
            top_k = st.slider("Number of recommendations", 1, 20, 5)
        
        # Admin Functions
        admin_section = st.expander("Admin Functions", expanded=False)
        with admin_section:
            st.markdown("### Build Embeddings")
            st.write("Upload assessment data and build embeddings in the vector store.")
            
            data_file = st.file_uploader("Upload SHL Assessments CSV", type="csv")
            
            if st.button("Build Embeddings", use_container_width=True):
                if data_file is not None:
                    with st.spinner("Building embeddings... This may take a while"):
                        try:
                            # Save the uploaded file temporarily
                            temp_file_path = os.path.join(os.getcwd(), "temp_data.csv")
                            with open(temp_file_path, "wb") as f:
                                f.write(data_file.getbuffer())
                            
                            # Import the build_embeddings function
                            from build_embeddings import build_embeddings
                            
                            # Build the embeddings
                            build_embeddings(temp_file_path, "shl_assessments")
                            
                            # Remove the temp file
                            os.remove(temp_file_path)
                            
                            st.success("Embeddings built successfully!")
                        except Exception as e:
                            st.error(f"Error building embeddings: {e}")
                else:
                    st.warning("Please upload a CSV file first.")
        
        st.divider()
        st.markdown("### About")
        st.markdown("""
        This tool helps you find the most suitable SHL assessments based on your requirements.
        
        Simply enter a job description or your requirements in the search box.
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
            recommender = get_recommender()
            if recommender is None:
                st.error("Failed to initialize the recommender.")
                return
            
            with st.spinner("Finding the best assessments for you..."):
                try:
                    # Get recommendations
                    if url:
                        results = recommender.recommend_from_url(
                            url=url,
                            top_k=top_k,
                            enhanced=enhanced_mode,
                            filters=filters
                        )
                    else:
                        results = recommender.recommend(
                            query=query,
                            top_k=top_k,
                            enhanced=enhanced_mode,
                            filters=filters
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
        sample_data = load_sample_data()
        if not sample_data.empty:
            st.dataframe(sample_data)

if __name__ == "__main__":
    main() 