import librosa
import numpy as np
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

def extract_audio_features(audio_path: str, sr: int = 16000) -> Dict[str, Any]:
    """Extract audio features for AI vs human voice detection."""
    
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