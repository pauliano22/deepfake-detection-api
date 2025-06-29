"""
API Endpoints for AI Voice Detection
Consolidates API functionality from multiple files.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import joblib
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Any

# Add parent directories to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

try:
    from features.audio_features import extract_audio_features
    from config.settings import *
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")

# Initialize FastAPI app
app = FastAPI(
    title="AI Voice Detection API",
    description="Detect AI-generated voices vs human voices using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for web applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and scaler
model = None
scaler = None
model_info = {
    "loaded": False,
    "model_file": None,
    "scaler_file": None,
    "error": None
}

@app.on_event("startup")
async def load_model():
    """Load the trained model and scaler on startup"""
    global model, scaler, model_info
    
    model_path = MODELS_DIR / MODEL_FILENAME
    scaler_path = MODELS_DIR / "scaler.pkl"
    
    try:
        if model_path.exists() and scaler_path.exists():
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            
            model_info = {
                "loaded": True,
                "model_file": str(model_path),
                "scaler_file": str(scaler_path),
                "model_type": type(model).__name__,
                "error": None
            }
            
            print("✅ Model and scaler loaded successfully")
        else:
            model_info = {
                "loaded": False,
                "model_file": str(model_path) if model_path.exists() else None,
                "scaler_file": str(scaler_path) if scaler_path.exists() else None,
                "error": "Model files not found. Train the model first with: python train.py"
            }
            print("❌ Model files not found. Train the model first.")
            
    except Exception as e:
        model_info = {
            "loaded": False,
            "model_file": None,
            "scaler_file": None,
            "error": f"Error loading model: {str(e)}"
        }
        print(f"❌ Error loading model: {e}")

@app.get("/")
async def root():
    """Root endpoint with basic API information"""
    return {
        "message": "AI Voice Detection API",
        "version": "1.0.0",
        "status": "healthy",
        "model_loaded": model_info["loaded"],
        "endpoints": {
            "health": "/health",
            "detect": "/detect (POST)",
            "batch_detect": "/batch_detect (POST)",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    return {
        "status": "healthy",
        "model_info": model_info,
        "api_version": "1.0.0"
    }

@app.post("/detect")
async def detect_ai_voice(audio: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Detect if an audio file contains AI-generated or human voice.
    
    Args:
        audio: Audio file (WAV, MP3, M4A, FLAC)
    
    Returns:
        Detection results with confidence scores
    """
    
    if not model_info["loaded"]:
        raise HTTPException(
            status_code=503, 
            detail=f"Model not loaded: {model_info.get('error', 'Unknown error')}"
        )
    
    # Validate file
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    allowed_extensions = {'.wav', '.mp3', '.m4a', '.flac'}
    file_extension = Path(audio.filename).suffix.lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format '{file_extension}'. Allowed: {', '.join(allowed_extensions)}"
        )
    
    if audio.size and audio.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB."
        )
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        # Extract features
        features = extract_audio_features(tmp_path, SAMPLE_RATE)
        
        # Convert to numpy array (maintaining feature order)
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        
        # Scale features
        feature_vector_scaled = scaler.transform(feature_vector)
        
        # Make prediction
        prediction_proba = model.predict_proba(feature_vector_scaled)[0]
        prediction = model.predict(feature_vector_scaled)[0]
        
        # Prepare response
        human_prob = float(prediction_proba[0])
        ai_prob = float(prediction_proba[1])
        
        return {
            "filename": audio.filename,
            "is_ai_generated": bool(prediction == 1),
            "confidence": float(max(prediction_proba)),
            "ai_probability": ai_prob,
            "human_probability": human_prob,
            "model_type": model_info.get("model_type", "unknown"),
            "features_extracted": len(features)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio file: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

@app.post("/batch_detect")
async def batch_detect_ai_voices(files: list[UploadFile] = File(...)):
    """
    Detect AI voices in multiple files at once.
    
    Args:
        files: List of audio files (max 10)
    
    Returns:
        List of detection results
    """
    if not model_info["loaded"]:
        raise HTTPException(
            status_code=503, 
            detail=f"Model not loaded: {model_info.get('error', 'Unknown error')}"
        )
    
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 files allowed per batch request."
        )
    
    results = []
    
    for i, file in enumerate(files):
        try:
            # Process each file (similar to single detect)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name
            
            # Extract and process features
            features = extract_audio_features(tmp_path, SAMPLE_RATE)
            feature_vector = np.array(list(features.values())).reshape(1, -1)
            feature_vector_scaled = scaler.transform(feature_vector)
            
            prediction_proba = model.predict_proba(feature_vector_scaled)[0]
            prediction = model.predict(feature_vector_scaled)[0]
            
            results.append({
                "index": i,
                "filename": file.filename,
                "is_ai_generated": bool(prediction == 1),
                "confidence": float(max(prediction_proba)),
                "ai_probability": float(prediction_proba[1]),
                "human_probability": float(prediction_proba[0])
            })
            
            # Clean up
            os.unlink(tmp_path)
            
        except Exception as e:
            results.append({
                "index": i,
                "filename": file.filename,
                "error": str(e)
            })
    
    return {"results": results, "total_files": len(files)}

@app.get("/model/info")
async def get_model_info():
    """Get detailed information about the loaded model"""
    if not model_info["loaded"]:
        return {"error": "Model not loaded", "details": model_info}
    
    try:
        # Get model parameters if available
        model_params = {}
        if hasattr(model, 'get_params'):
            model_params = model.get_params()
        
        # Get feature count
        feature_count = None
        if hasattr(model, 'n_features_in_'):
            feature_count = model.n_features_in_
        
        return {
            "model_loaded": True,
            "model_type": model_info.get("model_type"),
            "model_file": model_info.get("model_file"),
            "scaler_file": model_info.get("scaler_file"),
            "feature_count": feature_count,
            "model_parameters": model_params
        }
        
    except Exception as e:
        return {"error": f"Error getting model info: {str(e)}"}

@app.get("/stats")
async def get_dataset_stats():
    """Get statistics about the training dataset"""
    try:
        ai_dir = Path("data/raw/ai")
        human_dir = Path("data/raw/human")
        
        ai_count = len(list(ai_dir.glob("*.wav"))) if ai_dir.exists() else 0
        human_count = len(list(human_dir.glob("*.wav"))) if human_dir.exists() else 0
        
        return {
            "ai_samples": ai_count,
            "human_samples": human_count,
            "total_samples": ai_count + human_count,
            "balance_ratio": min(ai_count, human_count) / max(ai_count, human_count, 1)
        }
        
    except Exception as e:
        return {"error": f"Error getting dataset stats: {str(e)}"}

# Error handlers
@app.exception_handler(413)
async def request_entity_too_large_handler(request, exc):
    return HTTPException(
        status_code=413,
        detail="File too large. Please use a smaller audio file."
    )

# For backwards compatibility with existing api.py usage
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)