import os
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
