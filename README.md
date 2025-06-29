# 🎯 AI Voice Detection System

A comprehensive machine learning system that detects AI-generated voices (deepfakes) vs. human voices with high accuracy. Built with Python, scikit-learn, and FastAPI.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)

## 🚀 Features

- **High Accuracy**: 90-97% detection accuracy on ElevenLabs AI voices
- **Real-time Detection**: Fast audio processing and classification  
- **REST API**: Easy integration with web applications
- **CLI Interface**: Powerful command-line tools for all operations
- **Multiple Formats**: Supports WAV, MP3, M4A, FLAC audio files
- **Extensible**: Easy to train on new AI voice generators
- **Professional Structure**: Production-ready, scalable codebase

## 🧠 How It Works

The system extracts acoustic features from audio files that distinguish AI-generated from human speech:

- **Spectral Analysis**: MFCCs, spectral centroids, bandwidth, rolloff
- **Temporal Features**: Zero-crossing rates, energy stability, rhythm patterns
- **Prosodic Features**: Pitch patterns, formant consistency, breathing patterns  
- **Machine Learning**: Random Forest classifier with feature scaling and cross-validation

## 📊 Current Performance

| Voice Type | Accuracy | Confidence | Notes |
|------------|----------|------------|-------|
| ElevenLabs AI (trained) | 97% | High | Trained samples |
| ElevenLabs AI (unseen) | 90% | High | New content/text |
| Human Voice | 96% | High | Various speakers |
| OpenAI TTS | 85% | Medium | Limited training |

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)
- 4GB+ RAM for model training
- Audio processing capabilities

### Quick Setup

1. **Clone and setup**
   ```bash
   git clone https://github.com/yourusername/deepfake-detection-api.git
   cd deepfake-detection-api
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux  
   source venv/bin/activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys (optional for basic usage)
   ```

4. **Verify installation**
   ```bash
   python main.py --help
   ```

## 🎵 Quick Start

### 1. Check System Status
```bash
python main.py stats
```

### 2. Generate Training Data

**Option A: Using AI APIs (Recommended)**
```bash
# Add your API keys to .env file first
python main.py generate ai --voices 5 --samples 12
python main.py generate human --count 60
```

**Option B: Manual Setup**
- Record yourself reading sentences and place in `data/raw/human/`
- Use online TTS tools to generate AI samples in `data/raw/ai/`

### 3. Train the Model
```bash
python main.py train
```

### 4. Start the API
```bash
python main.py serve
```

### 5. Test the System
```bash
python main.py test
```

## 🎮 CLI Usage

The main interface uses a powerful CLI with the following commands:

### Data Management
```bash
# Show dataset statistics
python main.py stats

# Generate AI voice samples
python main.py generate ai --voices 5 --samples 12 --voice-type extended

# Download human voice samples  
python main.py generate human --count 100 --source common_voice

# Balance the dataset automatically
python main.py balance

# Convert audio formats
python main.py convert input.m4a output.wav --format wav
```

### Model Training
```bash
# Train with default settings
python main.py train

# Train with specific model type
python main.py train --model-type gradient_boost --test-size 0.3
```

### API Server
```bash
# Start development server
python main.py serve

# Production server with custom settings
python main.py serve --host 0.0.0.0 --port 8080 --workers 4

# Development with auto-reload
python main.py serve --reload
```

### Testing & Validation
```bash
# Run comprehensive tests
python main.py test

# Test specific file
python main.py test --file path/to/audio.wav

# Test against custom API endpoint
python main.py test --endpoint http://your-server:8000
```

### Maintenance
```bash
# Clean temporary files
python main.py clean

# Run initial setup
python main.py setup
```

## 🌐 API Usage

### Start the Server
```bash
python main.py serve
```
API available at `http://localhost:8000` with automatic documentation at `/docs`

### Endpoints

#### Health Check
```bash
GET /health
GET /
```

#### Single File Detection
```bash
POST /detect
Content-Type: multipart/form-data

# Form data:
audio: <audio_file>
```

**Response:**
```json
{
  "filename": "test_audio.wav",
  "is_ai_generated": true,
  "confidence": 0.97,
  "ai_probability": 0.97,
  "human_probability": 0.03,
  "model_type": "RandomForestClassifier",
  "features_extracted": 45
}
```

#### Batch Detection
```bash
POST /batch_detect
Content-Type: multipart/form-data

# Form data:
files: [<audio_file1>, <audio_file2>, ...]
```

#### Model Information
```bash
GET /model/info
GET /stats
```

### Example Usage

#### cURL
```bash
curl -X POST "http://localhost:8000/detect" \
     -F "audio=@path/to/audio.wav"
```

#### Python
```python
import requests

with open("audio.wav", "rb") as f:
    files = {"audio": f}
    response = requests.post("http://localhost:8000/detect", files=files)
    result = response.json()
    
print(f"AI Generated: {result['is_ai_generated']}")
print(f"Confidence: {result['confidence']:.1%}")
```

#### JavaScript
```javascript
const formData = new FormData();
formData.append('audio', audioFile);

fetch('http://localhost:8000/detect', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('AI Generated:', data.is_ai_generated);
    console.log('Confidence:', data.confidence);
});
```

## 📁 Project Structure

```
deepfake-detector/
├── 🐍 main.py                  # CLI entry point
├── 🐍 train.py                 # Model training script  
├── 🐍 serve.py                 # API server launcher
├── 📄 requirements.txt         # Dependencies
├── 📄 README.md               # This file
├── 📄 .env.example            # Environment template
├── 📂 src/                    # Source code
│   ├── 📂 api/                # API endpoints
│   │   └── 🐍 endpoints.py
│   ├── 📂 data/               # Data processing
│   │   ├── 🐍 generator.py    # Data generation
│   │   └── 🐍 converter.py    # Audio conversion
│   ├── 📂 features/           # Feature extraction
│   │   └── 🐍 audio_features.py
│   └── 📂 models/             # ML models
├── 📂 tests/                  # Test suite
│   ├── 🐍 test_api.py         # API tests
│   └── 🐍 test_model.py       # Model tests
├── 📂 config/                 # Configuration
│   └── 🐍 settings.py
├── 📂 data/                   # Training data
│   └── 📂 raw/
│       ├── 📂 ai/             # AI voice samples
│       └── 📂 human/          # Human voice samples
├── 📂 models/                 # Trained models
└── 📂 scripts/                # Utility scripts
```

## 🔧 Configuration

Key settings in `config/settings.py`:

- **Audio Processing**: Sample rate (16kHz), length limits
- **Feature Extraction**: MFCC parameters, window sizes  
- **Model Training**: Test split ratio, random seed
- **API Settings**: Host, port, file size limits

Environment variables in `.env`:
```bash
# API Keys (optional, for data generation)
ELEVENLABS_API_KEY=your_elevenlabs_key_here
OPENAI_API_KEY=your_openai_key_here

# Optional: Other TTS service keys
MURF_API_KEY=your_murf_key_here
```

## 🚀 Deployment

### Docker (Recommended)

```dockerfile
# Dockerfile included in repo
docker build -t ai-voice-detector .
docker run -p 8000:8000 ai-voice-detector
```

### Production Deployment

```bash
# Install production dependencies
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.api.endpoints:app

# Or use the CLI
python main.py serve --host 0.0.0.0 --port 8000 --workers 4
```

### Production Considerations

- ✅ Use a reverse proxy (nginx)
- ✅ Set up SSL/TLS certificates  
- ✅ Implement rate limiting
- ✅ Add authentication if needed
- ✅ Configure proper logging
- ✅ Set up monitoring and health checks
- ✅ Use environment-specific configurations

## 🧬 Extending the System

### Adding New AI Voice Generators

1. **Extend the data generator:**
   ```python
   # In src/data/generator.py
   def generate_new_ai_samples(self):
       # Add your AI service integration
   ```

2. **Generate training samples:**
   ```bash
   python main.py generate ai --voice-type extended
   ```

3. **Retrain the model:**
   ```bash
   python main.py train
   ```

### Improving Accuracy

**More Training Data:**
```bash
# Scale up data collection
python main.py generate ai --voices 10 --samples 25
python main.py generate human --count 250

# Balance the dataset
python main.py balance
```

**Advanced Models:**
```bash
# Try different model types
python main.py train --model-type gradient_boost
python main.py train --model-type neural_network
```

**Feature Engineering:**
- Modify `src/features/audio_features.py`
- Add new acoustic features
- Experiment with different feature combinations

### Supported AI Voice Generators

Currently optimized for:
- ✅ **ElevenLabs** (High accuracy)
- ✅ **OpenAI TTS** (Good accuracy)
- 🔄 **Google Cloud TTS** (In development)
- 🔄 **Azure Cognitive Services** (Planned)
- 🔄 **Murf.ai** (Planned)

## 📈 Roadmap

### Short Term
- [ ] Web interface for easy testing
- [ ] Real-time audio stream processing
- [ ] Model confidence calibration
- [ ] Batch processing optimizations

### Medium Term  
- [ ] Multi-language support
- [ ] Advanced neural network models
- [ ] Voice cloning detection
- [ ] Integration with popular platforms

### Long Term
- [ ] Real-time voice monitoring
- [ ] Mobile app development
- [ ] Enterprise features
- [ ] AI voice fingerprinting

## 🧪 Testing

### Run All Tests
```bash
python main.py test
```

### Specific Test Categories
```bash
# API functionality tests
python -m pytest tests/test_api.py -v

# Model performance tests  
python -m pytest tests/test_model.py -v

# Manual testing
python main.py test --file your_audio.wav
```

### Performance Benchmarking
```bash
# Built into the test suite
python main.py test --endpoint http://localhost:8000
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make your changes**
4. **Add tests for new functionality**
5. **Run the test suite**
   ```bash
   python main.py test
   ```
6. **Commit your changes**
   ```bash
   git commit -m 'Add amazing feature'
   ```
7. **Push and create a Pull Request**

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run pre-commit hooks
pre-commit install

# Run code formatting
black src/ tests/
flake8 src/ tests/
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Ethical Use

This tool is designed to help detect AI-generated audio for:

- **✅ Media Verification**: Identifying synthetic content in news/media
- **✅ Security**: Preventing voice-based fraud and scams
- **✅ Research**: Understanding AI voice generation capabilities
- **✅ Education**: Teaching about deepfake technology

**Please use responsibly** and in compliance with:
- Applicable laws and regulations
- Platform terms of service  
- Privacy and consent requirements
- Ethical AI principles

## 🙏 Acknowledgments

- [Mozilla Common Voice](https://commonvoice.mozilla.org/) for human voice datasets
- [ElevenLabs](https://elevenlabs.io/) for high-quality AI voice generation
- [OpenAI](https://openai.com/) for TTS capabilities
- [Librosa](https://librosa.org/) for audio processing
- [FastAPI](https://fastapi.tiangolo.com/) for the excellent API framework
- [scikit-learn](https://scikit-learn.org/) for machine learning tools

## 📞 Support

- **Documentation**: Check `/docs` endpoint when API is running
- **Issues**: Create an issue on GitHub
- **Discussions**: Use GitHub Discussions for questions
- **Enterprise**: Contact for enterprise licensing and support

---

**Built with ❤️ for a safer digital world**

*Detecting AI voices to protect against deepfake fraud and misinformation*