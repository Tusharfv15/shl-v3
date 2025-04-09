# Streamlit app package initialization
import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import from the parent modules
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from . import app

__all__ = ['app'] 