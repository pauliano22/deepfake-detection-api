import requests
from pathlib import Path

def test_file(file_path, description):
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return
    
    try:
        with open(file_path, 'rb') as f:
            files = {'audio': f}
            response = requests.post("http://localhost:8000/detect", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print(f"🎵 {description}")
            print(f"   File: {Path(file_path).name}")
            print(f"   Detected as: {'AI Generated' if result['is_ai_generated'] else 'Human Voice'}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   AI Probability: {result['ai_probability']:.3f}")
            print(f"   Human Probability: {result['human_probability']:.3f}\n")
        else:
            print(f"❌ Error: {response.text}")
    
    except Exception as e:
        print(f"❌ Error testing {file_path}: {e}")

# Test files
print("🧪 Testing AI Voice Detection API")
print("=" * 40)

test_file("data/raw/ai/elevenlabs_sample_1.wav", "ElevenLabs AI Sample")
test_file("data/raw/ai/elevenlabs_sample_2.wav", "ElevenLabs AI Sample") 

# Test human files (look for WAV files)
human_files = list(Path("data/raw/human").glob("*.wav"))
if human_files:
    test_file(str(human_files[0]), "Human Voice Sample")
else:
    print("No human WAV files found to test")

test_file("quick_test.wav", "NEW ElevenLabs Sample (unseen)")