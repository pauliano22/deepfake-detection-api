#!/usr/bin/env python3
"""
AI Voice Detection API Server
Main server file - consolidates API functionality.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import the existing API
from api.endpoints import app

def main():
    """Main server entry point with command line arguments"""
    parser = argparse.ArgumentParser(description='AI Voice Detection API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload for development')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes')
    
    args = parser.parse_args()
    
    # Import uvicorn here to avoid import issues
    import uvicorn
    
    print(f"🚀 Starting AI Voice Detection API")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Reload: {args.reload}")
    print(f"   Workers: {args.workers}")
    print(f"   Access: http://{args.host}:{args.port}")
    
    # Run the server
    uvicorn.run(
        "serve:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1
    )

if __name__ == "__main__":
    main()