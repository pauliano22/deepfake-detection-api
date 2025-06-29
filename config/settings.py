import os
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
