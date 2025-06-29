"""
Data Generator for AI Voice Detection
Consolidates all data generation functionality from multiple scripts.
"""

import os
import json
from pathlib import Path
from typing import List, Tuple, Optional
from dotenv import load_dotenv

class DataGenerator:
    def __init__(self):
        load_dotenv()
        self.ai_dir = Path("data/raw/ai")
        self.human_dir = Path("data/raw/human")
        self.ai_dir.mkdir(parents=True, exist_ok=True)
        self.human_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize ElevenLabs client if API key available
        self.elevenlabs_client = None
        if os.getenv("ELEVENLABS_API_KEY"):
            try:
                from elevenlabs.client import ElevenLabs
                self.elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
            except ImportError:
                print("❌ ElevenLabs library not installed. Install with: pip install elevenlabs")
        
        # Initialize OpenAI client if API key available
        self.openai_client = None
        if os.getenv("OPENAI_API_KEY"):
            try:
                import openai
                self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except ImportError:
                print("❌ OpenAI library not installed. Install with: pip install openai")
        
        # Diverse training texts for robust training
        self.training_texts = [
            # News/Formal
            "Breaking news: Scientists have made a breakthrough discovery in renewable energy technology.",
            "The quarterly earnings report shows significant growth across all business sectors.",
            "According to the latest research, artificial intelligence is advancing rapidly.",
            
            # Conversational/Casual  
            "Hey, I just wanted to check in and see how you're doing today.",
            "Can you please send me the report by Friday afternoon? Thanks!",
            "I'm excited about the new restaurant that opened downtown last week.",
            
            # Technical/Professional
            "The new software update includes several security improvements and bug fixes.",
            "Please remember to save your work before closing the application.",
            "This system requires administrator privileges to complete the installation.",
            
            # Customer Service
            "Thank you for calling, how can I assist you today?",
            "We apologize for the inconvenience and appreciate your patience.",
            "Your order has been processed and will arrive within three business days.",
            
            # Weather/Information
            "Today's weather forecast calls for partly cloudy skies with temperatures reaching seventy-five degrees.",
            "Traffic on the interstate is moving slowly due to construction in the left lane.",
            "There's a severe weather warning in effect until six PM this evening.",
            
            # Educational
            "In this tutorial, we'll learn how to create professional presentations.",
            "The human brain processes visual information sixty thousand times faster than text.",
            "Remember to cite your sources when writing academic papers.",
            
            # Messages/Communications
            "Hi, this is a reminder about your appointment tomorrow at ten AM.",
            "Please call me back when you get this message, it's important.",
            "The meeting has been rescheduled for tomorrow at three PM in conference room B.",
            
            # Emergency/Alerts
            "This is a test of the emergency broadcast system, this is only a test.",
            "Attention passengers, the next train will arrive in approximately five minutes.",
            "Please evacuate the building immediately and proceed to the nearest exit."
        ]
        
        # ElevenLabs voices with metadata
        self.elevenlabs_voices = [
            # Female voices
            ("21m00Tcm4TlvDq8ikWAM", "Rachel", "female_professional"),
            ("AZnzlk1XvdvUeBnXmlld", "Domi", "female_confident"),
            ("EXAVITQu4vr4xnSDxMaL", "Bella", "female_soft"),
            ("MF3mGyEYCl7XYWbV9V6O", "Elli", "female_emotional"),
            ("oWAxZDx7w5VEj9dCyTzz", "Grace", "female_narrator"),
            
            # Male voices
            ("ErXwobaYiN019PkySvjV", "Antoni", "male_youthful"),
            ("VR6AewLTigWG4xSOukaG", "Arnold", "male_crisp"),
            ("TxGEqnHWrfWFTfGW9XjX", "Josh", "male_casual"),
            ("pNInz6obpgDQGcFmaJgB", "Adam", "male_deep"),
            ("yoZ06aMxZJJ28mfd3POQ", "Sam", "male_narrative"),
        ]
        
        # OpenAI TTS voices
        self.openai_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    
    def generate_elevenlabs_samples(self, samples_per_voice: int = 12, voice_limit: Optional[int] = None):
        """Generate ElevenLabs AI voice samples"""
        if not self.elevenlabs_client:
            print("❌ ElevenLabs client not initialized. Check your API key in .env file.")
            return 0
        
        voices_to_use = self.elevenlabs_voices[:voice_limit] if voice_limit else self.elevenlabs_voices
        total_samples = len(voices_to_use) * samples_per_voice
        
        print(f"🎤 Generating ElevenLabs samples:")
        print(f"   Voices: {len(voices_to_use)}")
        print(f"   Samples per voice: {samples_per_voice}")
        print(f"   Total new samples: {total_samples}")
        print(f"   Estimated cost: ~${total_samples * 0.0025:.3f}")
        
        confirm = input("Continue? (y/n): ").lower()
        if confirm != 'y':
            return 0
        
        generated_count = 0
        
        for voice_id, voice_name, voice_type in voices_to_use:
            print(f"\n🗣️ Generating {voice_name} ({voice_type}) samples...")
            
            voice_count = 0
            for i, text in enumerate(self.training_texts):
                if voice_count >= samples_per_voice:
                    break
                
                filename = self.ai_dir / f"elevenlabs_{voice_name.lower()}_{i+1:02d}.wav"
                
                if filename.exists():
                    print(f"  ⏭️ Skipping existing {filename.name}")
                    voice_count += 1
                    continue
                
                try:
                    audio = self.elevenlabs_client.text_to_speech.convert(
                        voice_id=voice_id,
                        text=text,
                        model_id="eleven_monolingual_v1"
                    )
                    
                    with open(filename, "wb") as f:
                        for chunk in audio:
                            f.write(chunk)
                    
                    print(f"  ✅ {filename.name}")
                    voice_count += 1
                    generated_count += 1
                    
                except Exception as e:
                    print(f"  ❌ Error: {e}")
        
        print(f"\n🎉 Generated {generated_count} new ElevenLabs samples!")
        return generated_count
    
    def generate_openai_samples(self, samples_per_voice: int = 12):
        """Generate OpenAI TTS voice samples"""
        if not self.openai_client:
            print("❌ OpenAI client not initialized. Check your API key in .env file.")
            return 0
        
        total_samples = len(self.openai_voices) * samples_per_voice
        print(f"🎤 Generating OpenAI TTS samples:")
        print(f"   Voices: {len(self.openai_voices)}")
        print(f"   Samples per voice: {samples_per_voice}")
        print(f"   Total new samples: {total_samples}")
        print(f"   Estimated cost: ~${total_samples * 0.015:.3f}")
        
        confirm = input("Continue? (y/n): ").lower()
        if confirm != 'y':
            return 0
        
        generated_count = 0
        
        for voice in self.openai_voices:
            print(f"\n🗣️ Generating OpenAI {voice} samples...")
            
            voice_count = 0
            for i, text in enumerate(self.training_texts):
                if voice_count >= samples_per_voice:
                    break
                
                filename = self.ai_dir / f"openai_{voice}_{i+1:02d}.wav"
                
                if filename.exists():
                    print(f"  ⏭️ Skipping existing {filename.name}")
                    voice_count += 1
                    continue
                
                try:
                    response = self.openai_client.audio.speech.create(
                        model="tts-1",
                        voice=voice,
                        input=text
                    )
                    
                    with open(filename, "wb") as f:
                        f.write(response.content)
                    
                    print(f"  ✅ {filename.name}")
                    voice_count += 1
                    generated_count += 1
                    
                except Exception as e:
                    print(f"  ❌ Error: {e}")
        
        print(f"\n🎉 Generated {generated_count} new OpenAI samples!")
        return generated_count
    
    def download_human_samples(self, num_samples: int = 100, source: str = "common_voice"):
        """Download human voice samples from various sources"""
        print(f"📥 Downloading {num_samples} human samples from {source}...")
        
        if source == "common_voice":
            return self._download_common_voice(num_samples)
        elif source == "librispeech":
            return self._download_librispeech(num_samples)
        else:
            print(f"❌ Unknown source: {source}")
            return 0
    
    def _download_common_voice(self, num_samples: int):
        """Download from Mozilla Common Voice"""
        try:
            from datasets import load_dataset
            import soundfile as sf
            import librosa
            
            print("📚 Loading Mozilla Common Voice dataset...")
            dataset = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="train", streaming=True)
            
            count = 0
            for sample in dataset:
                if count >= num_samples:
                    break
                
                filename = self.human_dir / f"human_commonvoice_{count+1:03d}.wav"
                
                if filename.exists():
                    count += 1
                    continue
                
                try:
                    audio = sample["audio"]
                    
                    # Resample to 16kHz for consistency
                    y_resampled = librosa.resample(
                        audio["array"], 
                        orig_sr=audio["sampling_rate"], 
                        target_sr=16000
                    )
                    
                    sf.write(filename, y_resampled, 16000)
                    print(f"  ✅ {filename.name}")
                    count += 1
                    
                except Exception as e:
                    print(f"  ❌ Error processing sample {count}: {e}")
                    continue
            
            print(f"✅ Downloaded {count} human samples from Common Voice")
            return count
            
        except ImportError:
            print("❌ Required libraries not installed. Install with:")
            print("   pip install datasets soundfile")
            return 0
        except Exception as e:
            print(f"❌ Error downloading from Common Voice: {e}")
            return 0
    
    def _download_librispeech(self, num_samples: int):
        """Download from LibriSpeech dataset"""
        try:
            from datasets import load_dataset
            import soundfile as sf
            
            print("📚 Loading LibriSpeech dataset...")
            dataset = load_dataset("librispeech_asr", "clean", split="train.360", streaming=True)
            
            count = 0
            for sample in dataset:
                if count >= num_samples:
                    break
                
                filename = self.human_dir / f"human_librispeech_{count+1:03d}.wav"
                
                if filename.exists():
                    count += 1
                    continue
                
                try:
                    audio = sample["audio"]
                    sf.write(filename, audio["array"], audio["sampling_rate"])
                    print(f"  ✅ {filename.name}")
                    count += 1
                    
                except Exception as e:
                    print(f"  ❌ Error processing sample {count}: {e}")
                    continue
            
            print(f"✅ Downloaded {count} human samples from LibriSpeech")
            return count
            
        except ImportError:
            print("❌ datasets library not installed. Install with: pip install datasets")
            return 0
        except Exception as e:
            print(f"❌ Error downloading from LibriSpeech: {e}")
            return 0
    
    def get_stats(self):
        """Show comprehensive dataset statistics"""
        ai_files = list(self.ai_dir.glob("*.wav"))
        human_files = list(self.human_dir.glob("*.wav"))
        
        # Analyze AI samples by source and voice
        ai_by_source = {}
        for file in ai_files:
            parts = file.stem.split("_")
            if len(parts) >= 2:
                source = parts[0]
                voice = parts[1] if len(parts) > 1 else "unknown"
                key = f"{source}_{voice}"
                ai_by_source[key] = ai_by_source.get(key, 0) + 1
        
        # Analyze human samples by source
        human_by_source = {}
        for file in human_files:
            parts = file.stem.split("_")
            source = parts[1] if len(parts) > 1 and parts[0] == "human" else "unknown"
            human_by_source[source] = human_by_source.get(source, 0) + 1
        
        print("\n📊 Dataset Statistics")
        print("=" * 50)
        
        print(f"\n🤖 AI Samples: {len(ai_files)}")
        for source_voice, count in sorted(ai_by_source.items()):
            print(f"   {source_voice}: {count} samples")
        
        print(f"\n👤 Human Samples: {len(human_files)}")
        for source, count in sorted(human_by_source.items()):
            print(f"   {source}: {count} samples")
        
        print(f"\n📈 Total Samples: {len(ai_files) + len(human_files)}")
        
        # Training readiness assessment
        min_samples = min(len(ai_files), len(human_files))
        if min_samples == 0:
            print("🔴 Status: No training data available")
        elif min_samples < 20:
            print("🟠 Status: Insufficient data (need 20+ each)")
        elif min_samples < 50:
            print("🟡 Status: Minimal training data")
        elif min_samples < 100:
            print("🟢 Status: Basic training ready")
        elif min_samples < 500:
            print("🔵 Status: Good training data")
        else:
            print("🟣 Status: Excellent training data")
        
        balance_ratio = min_samples / max(len(ai_files), len(human_files)) if max(len(ai_files), len(human_files)) > 0 else 0
        if balance_ratio < 0.5:
            print(f"⚠️  Warning: Dataset imbalanced (ratio: {balance_ratio:.2f})")
        else:
            print(f"✅ Dataset balance: {balance_ratio:.2f}")
        
        return len(ai_files), len(human_files)
    
    def balance_dataset(self):
        """Balance AI and human samples automatically"""
        ai_count, human_count = self.get_stats()
        
        if ai_count == human_count:
            print("✅ Dataset is already balanced!")
            return
        
        if ai_count > human_count:
            needed = ai_count - human_count
            print(f"📥 Need {needed} more human samples to balance dataset")
            self.download_human_samples(needed)
        else:
            needed = human_count - ai_count
            samples_per_voice = max(1, needed // len(self.elevenlabs_voices))
            print(f"🎤 Need {needed} more AI samples ({samples_per_voice} per voice)")
            self.generate_elevenlabs_samples(samples_per_voice)
    
    def clean_dataset(self):
        """Clean and validate dataset files"""
        print("🧹 Cleaning dataset...")
        
        ai_files = list(self.ai_dir.glob("*"))
        human_files = list(self.human_dir.glob("*"))
        
        removed_count = 0
        
        for file_list, label in [(ai_files, "AI"), (human_files, "Human")]:
            for file in file_list:
                # Remove non-audio files
                if file.suffix.lower() not in ['.wav', '.mp3', '.m4a', '.flac']:
                    print(f"🗑️ Removing non-audio file: {file.name}")
                    file.unlink()
                    removed_count += 1
                    continue
                
                # Check file size (remove very small files)
                if file.stat().st_size < 1000:  # Less than 1KB
                    print(f"🗑️ Removing tiny file: {file.name}")
                    file.unlink()
                    removed_count += 1
                    continue
        
        print(f"✅ Cleaned dataset: removed {removed_count} invalid files")