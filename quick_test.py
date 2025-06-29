from elevenlabs.client import ElevenLabs
import os
from dotenv import load_dotenv

load_dotenv()

client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

# Completely new text the model has never seen
audio = client.text_to_speech.convert(
    voice_id="21m00Tcm4TlvDq8ikWAM",
    text="This is a completely new sentence that the AI detector has never heard before.",
    model_id="eleven_monolingual_v1"
)

with open("quick_test.wav", "wb") as f:
    for chunk in audio:
        f.write(chunk)

print("Generated quick_test.wav")