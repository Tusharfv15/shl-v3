import streamlit as st
import sys
import os
from pathlib import Path

# IMPORTANT: set_page_config must be the first Streamlit command
st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set up paths
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
repo_root = parent_dir

# Add necessary paths to sys.path
if str(parent_dir) not in sys.path:
    sys.path.append(str(parent_dir))
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

# Get path to the app.py file
app_path = current_dir / "app.py"

# Display information for debugging
debug_expander = st.sidebar.expander("Debug Info", expanded=False)
with debug_expander:
    st.write(f"Current file: {__file__}")
    st.write(f"App path: {app_path}")
    st.write(f"Current working directory: {os.getcwd()}")
    st.write(f"sys.path: {sys.path}")

# Import and run the app
try:
    sys.path.append(str(current_dir))
    import app
    app.main()
except Exception as e:
    st.error(f"Error running app: {e}")
    
    # Fall back to standalone app if there's an error
    st.warning("Falling back to standalone app...")
    try:
        from standalone_app import main as standalone_main
        standalone_main()
    except Exception as e2:
        st.error(f"Error running standalone app: {e2}")
        st.error("All attempts to run the app have failed. Please check the logs for more information.") 