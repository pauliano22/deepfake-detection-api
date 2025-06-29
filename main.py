#!/usr/bin/env python3
"""
AI Voice Detection System - Main Entry Point

Usage:
    python main.py stats                    # Show dataset statistics
    python main.py generate --ai            # Generate AI voice samples
    python main.py generate --human         # Download human voice samples  
    python main.py train                    # Train the detection model
    python main.py serve                    # Start the API server
    python main.py test                     # Test the trained model
"""

import click
import subprocess
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

@click.group()
def cli():
    """🎯 AI Voice Detection System"""
    pass

@cli.command()
def stats():
    """📊 Show dataset statistics"""
    try:
        from data.generator import DataGenerator
        generator = DataGenerator()
        generator.get_stats()
    except ImportError as e:
        click.echo(f"❌ Error importing modules: {e}")
        click.echo("Make sure you're in the virtual environment: venv\\Scripts\\activate")

@cli.group()
def generate():
    """🎤 Generate training data"""
    pass

@generate.command()
@click.option('--voices', default=5, help='Number of different voices to use')
@click.option('--samples', default=12, help='Number of samples per voice')
@click.option('--voice-type', default='basic', type=click.Choice(['basic', 'extended']), 
              help='Use basic (5 voices) or extended (all voices) voice set')
def ai(voices, samples, voice_type):
    """Generate AI voice samples using ElevenLabs"""
    try:
        from data.generator import DataGenerator
        generator = DataGenerator()
        
        if voice_type == 'extended':
            voices = None  # Use all available voices
        
        generator.generate_elevenlabs_samples(
            samples_per_voice=samples, 
            voice_limit=voices
        )
    except ImportError as e:
        click.echo(f"❌ Error importing modules: {e}")
    except Exception as e:
        click.echo(f"❌ Error generating AI samples: {e}")

@generate.command()
@click.option('--count', default=100, help='Number of human samples to download')
@click.option('--source', default='common_voice', help='Source dataset to use')
def human(count, source):
    """Download human voice samples from free datasets"""
    try:
        from data.generator import DataGenerator
        generator = DataGenerator()
        generator.download_human_samples(num_samples=count, source=source)
    except ImportError as e:
        click.echo(f"❌ Error importing modules: {e}")
    except Exception as e:
        click.echo(f"❌ Error downloading human samples: {e}")

@cli.command()
@click.option('--model-type', default='random_forest', 
              type=click.Choice(['random_forest', 'gradient_boost', 'neural_network']),
              help='Type of model to train')
@click.option('--test-size', default=0.2, help='Fraction of data to use for testing')
def train(model_type, test_size):
    """🧠 Train the voice detection model"""
    try:
        click.echo("🚀 Starting model training...")
        result = subprocess.run([
            sys.executable, 'train.py', 
            '--model-type', model_type,
            '--test-size', str(test_size)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            click.echo("✅ Training completed successfully!")
            click.echo(result.stdout)
        else:
            click.echo("❌ Training failed:")
            click.echo(result.stderr)
            
    except Exception as e:
        click.echo(f"❌ Error during training: {e}")

@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind the server to')
@click.option('--port', default=8000, help='Port to bind the server to')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
def serve(host, port, reload):
    """🌐 Start the API server"""
    try:
        click.echo(f"🚀 Starting API server on {host}:{port}")
        if reload:
            click.echo("🔄 Auto-reload enabled (development mode)")
        
        cmd = [sys.executable, 'serve.py', '--host', host, '--port', str(port)]
        if reload:
            cmd.append('--reload')
            
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        click.echo("\n👋 Server stopped")
    except Exception as e:
        click.echo(f"❌ Error starting server: {e}")

@cli.command()
@click.option('--endpoint', default='http://localhost:8000', help='API endpoint to test')
@click.option('--file', help='Specific audio file to test')
def test(endpoint, file):
    """🧪 Test the trained model and API"""
    try:
        from tests.test_model import ModelTester
        from tests.test_api import ApiTester
        
        if file:
            # Test specific file
            api_tester = ApiTester(endpoint)
            api_tester.test_single_file(file)
        else:
            # Run comprehensive tests
            click.echo("🧪 Running model tests...")
            model_tester = ModelTester()
            model_tester.run_tests()
            
            click.echo("\n🌐 Running API tests...")
            api_tester = ApiTester(endpoint)
            api_tester.run_tests()
            
    except ImportError as e:
        click.echo(f"❌ Error importing test modules: {e}")
    except Exception as e:
        click.echo(f"❌ Error during testing: {e}")

@cli.command()
def balance():
    """⚖️ Balance the dataset (ensure equal AI/human samples)"""
    try:
        from data.generator import DataGenerator
        generator = DataGenerator()
        generator.balance_dataset()
    except ImportError as e:
        click.echo(f"❌ Error importing modules: {e}")
    except Exception as e:
        click.echo(f"❌ Error balancing dataset: {e}")

@cli.command()
@click.argument('input_file')
@click.argument('output_file')
@click.option('--format', default='wav', help='Output format (wav, mp3, m4a)')
def convert(input_file, output_file, format):
    """🔄 Convert audio file formats"""
    try:
        from data.converter import AudioConverter
        converter = AudioConverter()
        converter.convert_file(input_file, output_file, format)
        click.echo(f"✅ Converted {input_file} → {output_file}")
    except ImportError as e:
        click.echo(f"❌ Error importing converter: {e}")
    except Exception as e:
        click.echo(f"❌ Conversion failed: {e}")

@cli.command()
def setup():
    """🛠️ Run initial project setup"""
    try:
        click.echo("🛠️ Running project setup...")
        subprocess.run([sys.executable, 'scripts/setup.py'])
    except Exception as e:
        click.echo(f"❌ Setup failed: {e}")

@cli.command()
def clean():
    """🧹 Clean up temporary files and cache"""
    import shutil
    
    patterns_to_clean = [
        "**/__pycache__",
        "**/*.pyc", 
        "**/*.pyo",
        "*.wav",  # Temporary audio files in root
        "quick_test*"
    ]
    
    click.echo("🧹 Cleaning up temporary files...")
    
    for pattern in patterns_to_clean:
        for path in Path(".").glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
                click.echo(f"🗑️ Removed directory: {path}")
            elif path.is_file():
                path.unlink()
                click.echo(f"🗑️ Removed file: {path}")

if __name__ == '__main__':
    cli()