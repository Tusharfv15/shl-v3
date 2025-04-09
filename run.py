import sys
from pathlib import Path

# Add recommendation_system to path
sys.path.append(str(Path(__file__).resolve().parent))

# Import the app
from recommendation_system.api import app

# This file is used by gunicorn or uvicorn in production
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("run:app", host="0.0.0.0", port=8000, reload=True) 