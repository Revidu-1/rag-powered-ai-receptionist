"""
Simple script to start the FastAPI backend server
"""

import uvicorn
import os
import sys
from pathlib import Path

if __name__ == "__main__":
    # Change to project root directory (parent of scripts/)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    os.chdir(project_root)
    
    # Add project root to Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Ensure UTF-8 output on Windows to avoid UnicodeEncodeError in logs
    if sys.platform == "win32":
        os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    
    print("Starting Appointment Booking Backend...")
    print("API will be available at: http://localhost:8000")
    print("Dashboard will be available at: http://localhost:8000")
    print("API docs will be available at: http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop the server\n")
    
    uvicorn.run("app.backend:app", host="0.0.0.0", port=8000, reload=False)


