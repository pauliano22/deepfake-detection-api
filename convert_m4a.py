from pydub import AudioSegment
from pathlib import Path
import os

def convert_m4a_to_wav(input_path, output_path):
    """Convert M4A to WAV using pydub"""
    try:
        # Load M4A file
        audio = AudioSegment.from_file(input_path, format="m4a")
        
        # Convert to 16kHz mono WAV (standard for ML)
        audio = audio.set_frame_rate(16000).set_channels(1)
        
        # Export as WAV
        audio.export(output_path, format="wav")
        print(f"✅ Converted {input_path.name} → {output_path.name}")
        return True
    except Exception as e:
        print(f"❌ Error converting {input_path}: {e}")
        return False

# Convert all M4A files in human folder
human_dir = Path("data/raw/human")
print("🔄 Converting M4A files to WAV...")

for m4a_file in human_dir.glob("*.m4a"):
    wav_file = m4a_file.with_suffix(".wav")
    convert_m4a_to_wav(m4a_file, wav_file)

print("✅ Conversion complete!")