import os
from elevenlabs.client import ElevenLabs
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Create directories
data_dir = Path("data/raw/ai")
data_dir.mkdir(parents=True, exist_ok=True)

# Sample texts
texts = [
    "The weather today is sunny with a high of seventy-five degrees.",
    "Please call me back when you get this message.",
    "According to the latest research, artificial intelligence is advancing rapidly.",
    "I'm excited about the new restaurant that opened downtown.",
    "The meeting has been rescheduled for tomorrow at three PM."
]

api_key = os.getenv("ELEVENLABS_API_KEY")

if not api_key or api_key == "your_elevenlabs_api_key_here":
    print("❌ Please set your ELEVENLABS_API_KEY in the .env file")
    exit(1)

print("🎤 Generating ElevenLabs samples...")

# Initialize client
client = ElevenLabs(api_key=api_key)

for i, text in enumerate(texts, 1):
    print(f"Generating sample {i}...")
    
    try:
        # Generate audio using text_to_speech
        audio = client.text_to_speech.convert(
            voice_id="21m00Tcm4TlvDq8ikWAM",  # Rachel voice ID
            text=text,
            model_id="eleven_monolingual_v1"
        )
        
        # Save audio
        filename = data_dir / f"elevenlabs_sample_{i}.wav"
        with open(filename, "wb") as f:
            for chunk in audio:
                f.write(chunk)
        
        print(f"✅ Saved: {filename}")
        
    except Exception as e:
        print(f"❌ Error generating sample {i}: {e}")

print("🎉 Done! Now record yourself saying the same sentences and save them in data/raw/human/")