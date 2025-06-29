import os
import openai
from pathlib import Path
from config.settings import RAW_DATA_DIR

# Sample texts to generate
texts = [
    "The weather today is sunny with a high of seventy-five degrees.",
    "Please call me back when you get this message.",
    "According to the latest research, artificial intelligence is advancing rapidly.",
]

client = openai.OpenAI()

for i, text in enumerate(texts):
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    
    with open(RAW_DATA_DIR / "ai" / f"openai_sample_{i}.wav", "wb") as f:
        f.write(response.content)
    
    print(f"Generated sample {i}")