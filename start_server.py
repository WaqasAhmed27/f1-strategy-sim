#!/usr/bin/env python3
"""
Startup script for F1 Prediction System on Railway
Handles PORT environment variable correctly
"""

import os
import uvicorn
import sys
from pathlib import Path

def main():
    """Start the FastAPI server with proper port handling."""
    # Get port from environment variable
    port = int(os.environ.get("PORT", 8000))
    
    print(f"🏎️ Starting F1 Prediction System on port {port}")
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"🐍 Python path: {sys.path}")
    
    # Start the server
    uvicorn.run(
        "web.backend.main:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main()
