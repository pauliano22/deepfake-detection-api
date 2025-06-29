 #!/usr/bin/env python3
"""
Complete setup script that creates all project files.
Run this once to set up the entire AI voice detection project.
"""

import os
import subprocess
import sys
from pathlib import Path

def create_file(filepath, content):
    """Create a file with given content."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"✅ Created {filepath}")

def create_all_files():
    """Create all project files."""
    
    # requirements.txt
    requirements_content = """# Core ML/Audio Processing
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
librosa>=0.10.0
scipy>=1.10.0
soundfile>=0.12.0

# Deep Learning (optional for advanced models)
torch>=2.0.0
torchaudio>=2.0.0

# Data Handling
datasets>=2.12.0
huggingface_hub>=0.15.0

# API
fastapi>=0.100.0
uvicorn>=0.22.0
python-multipart>=0.0.6
pydantic>=2.0.0

# Utilities
python-dotenv>=1.0.0
click>=8.1.0
tqdm>=4.65.0
joblib>=1.3.0

# Development
jupyter>=1.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
pytest>=7.4.0

# AI Voice Generation APIs (for data collection)
openai>=1.0.0
elevenlabs>=0.2.0
"""
    
    # .env.example
    env_content = """# AI Voice Detection Environment Variables

# API Keys (get these from respective services)
OPENAI_API_KEY=your_openai_api_key_here
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here

# Optional: Add other AI voice service keys
# MURF_API_KEY=your_murf_key_here
# SPEECHIFY_API_KEY=your_speechify_key_here
"""
    
    # .gitignore
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/

# Data files
data/raw/
data/processed/
*.wav
*.mp3
*.m4a
*.flac

# Models
models/*.pkl
models/*.joblib

# Environment
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Logs
*.log
logs/

# Temporary files
tmp/
temp/
"""
    
    # config/settings.py
    config_content = """import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
    (RAW_DATA_DIR / "human").mkdir(exist_ok=True)
    (RAW_DATA_DIR / "ai").mkdir(exist_ok=True)

# Audio processing settings
SAMPLE_RATE = 16000
MAX_AUDIO_LENGTH = 10.0  # seconds
MIN_AUDIO_LENGTH = 1.0   # seconds

# Feature extraction settings
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512

# Model settings
TEST_SIZE = 0.2
RANDOM_STATE = 42
MODEL_FILENAME = "voice_detector.pkl"

# API Keys (set in .env file)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
"""
    
    # __init__.py files
    init_content = ""
    
    # src/features/audio_features.py (shortened for setup)
    audio_features_content = '''import librosa
import numpy as np
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

def extract_audio_features(audio_path: str, sr: int = 16000) -> Dict[str, Any]:
    """Extract comprehensive audio features for AI vs human voice detection."""
    
    # Load and normalize audio
    y, sr = librosa.load(audio_path, sr=sr)
    y = librosa.util.normalize(y)
    
    features = {}
    
    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features['mfcc_mean'] = np.mean(mfccs, axis=1)
    features['mfcc_std'] = np.std(mfccs, axis=1)
    
    # Spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features['spectral_centroid_mean'] = np.mean(spectral_centroids)
    features['spectral_centroid_std'] = np.std(spectral_centroids)
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)
    
    # RMS energy
    rms = librosa.feature.rms(y=y)[0]
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)
    
    # Flatten features
    flattened = {}
    for key, value in features.items():
        if isinstance(value, np.ndarray):
            if value.ndim > 0:
                for i, v in enumerate(value):
                    flattened[f"{key}_{i}"] = float(v)
            else:
                flattened[key] = float(value)
        else:
            flattened[key] = float(value)
    
    return flattened

def extract_features_batch(audio_paths: list, sr: int = 16000) -> np.ndarray:
    """Extract features from multiple audio files."""
    all_features = []
    
    for path in audio_paths:
        try:
            features = extract_audio_features(path, sr)
            feature_vector = list(features.values())
            all_features.append(feature_vector)
        except Exception as e:
            print(f"Error processing {path}: {e}")
            continue
    
    return np.array(all_features)
'''
    
    # Simple training script
    train_content = '''import os
import glob
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from features.audio_features import extract_features_batch
from config.settings import *

def main():
    """Main training pipeline."""
    print("Loading training data...")
    
    human_files = glob.glob(str(RAW_DATA_DIR / "human" / "*.wav"))
    ai_files = glob.glob(str(RAW_DATA_DIR / "ai" / "*.wav"))
    
    print(f"Found {len(human_files)} human, {len(ai_files)} AI samples")
    
    if len(human_files) == 0 or len(ai_files) == 0:
        print("❌ No training data found!")
        return False
    
    # Extract features
    human_features = extract_features_batch(human_files, SAMPLE_RATE)
    ai_features = extract_features_batch(ai_files, SAMPLE_RATE)
    
    # Combine data
    X = np.vstack([human_features, ai_features])
    y = np.hstack([np.zeros(len(human_features)), np.ones(len(ai_features))])
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred, target_names=['Human', 'AI']))
    
    # Save model
    joblib.dump(model, MODELS_DIR / "voice_detector.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    
    print("✅ Model saved!")
    return True

if __name__ == "__main__":
    main()
'''
    
    # Simple API
    api_content = '''from fastapi import FastAPI, UploadFile, File, HTTPException
import tempfile
import os
import joblib
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from features.audio_features import extract_audio_features
from config.settings import *

app = FastAPI(title="AI Voice Detection API")

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
        print("❌ Model not found")

@app.get("/")
async def root():
    return {"message": "AI Voice Detection API", "model_loaded": model is not None}

@app.post("/detect")
async def detect_ai_voice(audio: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
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
'''
    
    # Create all files
    files_to_create = [
        ("requirements.txt", requirements_content),
        (".env.example", env_content),
        (".gitignore", gitignore_content),
        ("config/__init__.py", init_content),
        ("config/settings.py", config_content),
        ("src/__init__.py", init_content),
        ("src/features/__init__.py", init_content),
        ("src/features/audio_features.py", audio_features_content),
        ("src/models/__init__.py", init_content),
        ("src/models/train.py", train_content),
        ("src/api/__init__.py", init_content),
        ("src/api/main.py", api_content),
    ]
    
    for filepath, content in files_to_create:
        create_file(filepath, content)

def setup_environment():
    """Set up Python environment and install dependencies."""
    print("\n🔧 Setting up Python environment...")
    
    # Create virtual environment
    if not os.path.exists('venv'):
        subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
        print("✅ Created virtual environment")
    
    # Determine pip path
    pip_cmd = "venv/Scripts/pip" if os.name == 'nt' else "venv/bin/pip"
    
    # Install dependencies
    subprocess.run([pip_cmd, 'install', '--upgrade', 'pip'], check=True)
    subprocess.run([pip_cmd, 'install', '-r', 'requirements.txt'], check=True)
    print("✅ Installed dependencies")

def main():
    """Main setup function."""
    print("🚀 Setting up AI Voice Detection project...")
    
    # Create all files
    create_all_files()
    
    # Set up environment
    try:
        setup_environment()
    except subprocess.CalledProcessError as e:
        print(f"❌ Environment setup failed: {e}")
        print("You can manually run: python -m venv venv && pip install -r requirements.txt")
    
    # Copy .env.example to .env
    if not os.path.exists('.env'):
        import shutil
        shutil.copy('.env.example', '.env')
        print("✅ Created .env file")
    
    print("\n🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Edit .env file with your API keys")
    print("2. Activate virtual environment:")
    if os.name == 'nt':
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("3. Generate AI samples: python -c 'from scripts.generate_samples import main; main()'")
    print("4. Add human voice files to data/raw/human/")
    print("5. Train model: python src/models/train.py")
    print("6. Start API: python src/api/main.py")

if __name__ == "__main__":
    main()
