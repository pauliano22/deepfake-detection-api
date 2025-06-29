"""
Audio Format Converter
Handles conversion between different audio formats for the voice detection system.
"""

import os
import subprocess
from pathlib import Path
from typing import Optional

class AudioConverter:
    """Handles audio format conversion using multiple backends"""
    
    def __init__(self):
        self.backends = self._detect_backends()
    
    def _detect_backends(self):
        """Detect available audio conversion backends"""
        backends = {}
        
        # Check for FFmpeg
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                backends['ffmpeg'] = True
        except FileNotFoundError:
            backends['ffmpeg'] = False
        
        # Check for Python libraries
        try:
            import pydub
            backends['pydub'] = True
        except ImportError:
            backends['pydub'] = False
        
        try:
            import librosa
            import soundfile as sf
            backends['librosa'] = True
        except ImportError:
            backends['librosa'] = False
        
        return backends
    
    def convert_file(self, input_path: str, output_path: str, 
                    target_format: str = 'wav', sample_rate: int = 16000):
        """Convert audio file to target format"""
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Try conversion methods in order of preference
        methods = [
            self._convert_with_librosa,
            self._convert_with_ffmpeg, 
            self._convert_with_pydub
        ]
        
        for method in methods:
            try:
                method(input_path, output_path, target_format, sample_rate)
                print(f"✅ Converted using {method.__name__}")
                return True
            except Exception as e:
                print(f"❌ {method.__name__} failed: {e}")
                continue
        
        raise Exception("All conversion methods failed")
    
    def _convert_with_librosa(self, input_path: Path, output_path: Path, 
                             target_format: str, sample_rate: int):
        """Convert using librosa + soundfile"""
        if not self.backends.get('librosa'):
            raise Exception("Librosa not available")
        
        import librosa
        import soundfile as sf
        
        # Load audio
        y, sr = librosa.load(str(input_path), sr=sample_rate)
        
        # Save in target format
        sf.write(str(output_path), y, sample_rate, format=target_format.upper())
    
    def _convert_with_ffmpeg(self, input_path: Path, output_path: Path,
                           target_format: str, sample_rate: int):
        """Convert using FFmpeg"""
        if not self.backends.get('ffmpeg'):
            raise Exception("FFmpeg not available")
        
        cmd = [
            'ffmpeg', '-i', str(input_path),
            '-acodec', 'pcm_s16le',
            '-ar', str(sample_rate),
            '-ac', '1',  # Mono
            '-y',  # Overwrite output
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"FFmpeg error: {result.stderr}")
    
    def _convert_with_pydub(self, input_path: Path, output_path: Path,
                           target_format: str, sample_rate: int):
        """Convert using pydub"""
        if not self.backends.get('pydub'):
            raise Exception("Pydub not available")
        
        from pydub import AudioSegment
        
        # Load audio file
        audio = AudioSegment.from_file(str(input_path))
        
        # Convert to target sample rate and mono
        audio = audio.set_frame_rate(sample_rate).set_channels(1)
        
        # Export in target format
        audio.export(str(output_path), format=target_format)
    
    def convert_directory(self, input_dir: str, output_dir: str, 
                         target_format: str = 'wav', sample_rate: int = 16000):
        """Convert all audio files in a directory"""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Audio file extensions to convert
        audio_extensions = ['.m4a', '.mp3', '.flac', '.ogg', '.aac', '.wma']
        
        converted_count = 0
        for file_path in input_dir.iterdir():
            if file_path.suffix.lower() in audio_extensions:
                output_file = output_dir / f"{file_path.stem}.{target_format}"
                
                try:
                    self.convert_file(file_path, output_file, target_format, sample_rate)
                    converted_count += 1
                except Exception as e:
                    print(f"❌ Failed to convert {file_path.name}: {e}")
        
        print(f"✅ Converted {converted_count} files to {output_dir}")
        return converted_count
    
    def batch_convert_m4a_to_wav(self, directory: str):
        """Specific method to convert M4A files to WAV (common use case)"""
        dir_path = Path(directory)
        
        m4a_files = list(dir_path.glob("*.m4a"))
        if not m4a_files:
            print(f"No M4A files found in {directory}")
            return 0
        
        print(f"🔄 Converting {len(m4a_files)} M4A files to WAV...")
        
        converted_count = 0
        for m4a_file in m4a_files:
            wav_file = m4a_file.with_suffix('.wav')
            
            try:
                self.convert_file(m4a_file, wav_file, 'wav', 16000)
                print(f"  ✅ {m4a_file.name} → {wav_file.name}")
                converted_count += 1
            except Exception as e:
                print(f"  ❌ Failed to convert {m4a_file.name}: {e}")
        
        return converted_count
    
    def get_audio_info(self, file_path: str):
        """Get information about an audio file"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        info = {
            'file_path': str(file_path),
            'file_size': file_path.stat().st_size,
            'format': file_path.suffix.lower(),
        }
        
        try:
            import librosa
            y, sr = librosa.load(str(file_path), sr=None)
            
            info.update({
                'sample_rate': sr,
                'duration': len(y) / sr,
                'channels': 1 if y.ndim == 1 else y.shape[0],
                'samples': len(y)
            })
            
        except Exception as e:
            info['error'] = str(e)
        
        return info
    
    def print_backend_status(self):
        """Print status of available conversion backends"""
        print("\n🔧 Audio Conversion Backends:")
        print("=" * 30)
        
        for backend, available in self.backends.items():
            status = "✅ Available" if available else "❌ Not available"
            print(f"{backend:10} {status}")
        
        if not any(self.backends.values()):
            print("\n⚠️  No conversion backends available!")
            print("Install at least one of:")
            print("  - pip install librosa soundfile")
            print("  - pip install pydub")
            print("  - Install FFmpeg system-wide")
        
        return self.backends