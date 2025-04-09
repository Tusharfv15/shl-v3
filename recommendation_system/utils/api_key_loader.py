import os
import streamlit as st

def load_openai_api_key():
    """
    Load OpenAI API key from environment variables or Streamlit secrets
    
    Returns:
        The API key if found, None otherwise
    """
    api_key = None
    
    # First try to get from environment
    if "OPENAI_API_KEY" in os.environ:
        api_key = os.environ["OPENAI_API_KEY"]
    
    # Then try Streamlit secrets
    if api_key is None and hasattr(st, 'secrets') and "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
        # Also set in environment for libraries that expect it there
        os.environ["OPENAI_API_KEY"] = api_key
    
    return api_key 