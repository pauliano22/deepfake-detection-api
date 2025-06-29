import joblib
import numpy as np
from pathlib import Path
from src.features.audio_features import extract_audio_features
from config.settings import *

# Load the trained model
model = joblib.load(MODELS_DIR / "voice_detector.pkl")
scaler = joblib.load(MODELS_DIR / "scaler.pkl")

print("🤖 Testing AI Voice Detection Model")
print("=" * 40)

# Test on some files
test_files = [
    "data/raw/ai/elevenlabs_sample_1.wav",
    "data/raw/human/human_sample_1.wav",  # assuming you have WAV files
]

for file_path in test_files:
    if Path(file_path).exists():
        try:
            # Extract features
            features = extract_audio_features(file_path)
            feature_vector = np.array(list(features.values())).reshape(1, -1)
            feature_vector_scaled = scaler.transform(feature_vector)
            
            # Predict
            prediction = model.predict(feature_vector_scaled)[0]
            probabilities = model.predict_proba(feature_vector_scaled)[0]
            
            print(f"\n🎵 File: {Path(file_path).name}")
            print(f"   Prediction: {'AI Generated' if prediction == 1 else 'Human Voice'}")
            print(f"   Human Prob: {probabilities[0]:.3f}")
            print(f"   AI Prob: {probabilities[1]:.3f}")
            
        except Exception as e:
            print(f"❌ Error testing {file_path}: {e}")
    else:
        print(f"❌ File not found: {file_path}")