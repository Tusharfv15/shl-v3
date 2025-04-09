import subprocess
import os
import sys
from pathlib import Path

def main():
    """Run the Streamlit app"""
    current_dir = Path(__file__).resolve().parent
    app_path = current_dir / "streamlit_app" / "app.py"
    
    # Ensure the app.py exists
    if not app_path.exists():
        print(f"Error: Could not find app at {app_path}")
        return 1
    
    print(f"Starting Streamlit app from: {app_path}")
    
    # Run the streamlit app
    try:
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(app_path),
            "--server.port", "8501",
            "--server.address", "localhost"
        ]
        subprocess.run(cmd)
        return 0
    except Exception as e:
        print(f"Error running Streamlit app: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 