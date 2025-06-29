from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import joblib
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

try:
    from src.features.audio_features import extract_audio_features
    from config.settings import *
except ImportError:
    print("Please run from project root directory")
    sys.exit(1)

app = FastAPI(title="AI Voice Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
scaler = None

@app.on_event("startup")
async def load_model():
    global model, scaler
    try:
        model = joblib.load(MODELS_DIR / "voice_detector.pkl")
        scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        print("✅ Model loaded")
    except:
        print("❌ Model not found. Train model first with: python train.py")

@app.get("/")
async def root():
    return {"message": "AI Voice Detection API", "model_loaded": model is not None}

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/detect")
async def detect_ai_voice(audio: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not audio.filename.lower().endswith((".wav", ".mp3", ".m4a", ".flac")):
        raise HTTPException(status_code=400, detail="Invalid file format")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    
    try:
        features = extract_audio_features(tmp_path, SAMPLE_RATE)
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        feature_vector_scaled = scaler.transform(feature_vector)
        
        prediction_proba = model.predict_proba(feature_vector_scaled)[0]
        prediction = model.predict(feature_vector_scaled)[0]
        
        return {
            "is_ai_generated": bool(prediction == 1),
            "confidence": float(max(prediction_proba)),
            "ai_probability": float(prediction_proba[1]),
            "human_probability": float(prediction_proba[0])
        }
    finally:
        os.unlink(tmp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)