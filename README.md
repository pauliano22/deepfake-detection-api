# 🎯 AI Voice Detection API

A machine learning system that detects AI-generated voices (deepfakes) vs. human voices with high accuracy. Built with Python, scikit-learn, and FastAPI.

## 🚀 Features

- **High Accuracy**: 90-97% detection accuracy on ElevenLabs AI voices
- **Real-time Detection**: Fast audio processing and classification
- **REST API**: Easy integration with web applications
- **Multiple Formats**: Supports WAV, MP3, M4A, FLAC audio files
- **Extensible**: Easy to train on new AI voice generators

## 🧠 How It Works

The system extracts acoustic features from audio files that distinguish AI-generated from human speech:
- **Spectral Analysis**: MFCCs, spectral centroids, bandwidth
- **Temporal Features**: Zero-crossing rates, energy stability
- **Prosodic Features**: Pitch patterns, formant consistency
- **Machine Learning**: Random Forest classifier with feature scaling

## 📊 Current Performance

| Voice Type | Accuracy | Confidence |
|------------|----------|------------|
| ElevenLabs AI (trained) | 97% | High |
| ElevenLabs AI (unseen) | 90% | High |
| Human Voice | 96% | High |

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/deepfake-detection-api.git
   cd deepfake-detection-api
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## 🎵 Training Data Setup

### Option 1: Generate AI Samples (Requires API Keys)

1. **Get API Keys**
   - [ElevenLabs](https://elevenlabs.io): For high-quality AI voice generation
   - [OpenAI](https://platform.openai.com): For diverse AI voice samples

2. **Generate AI Voice Samples**
   ```bash
   python data_generator.py
   ```

3. **Add Human Voice Samples**
   - Record yourself reading the same sentences
   - Or download from [Mozilla Common Voice](https://commonvoice.mozilla.org/datasets)
   - Place WAV files in `data/raw/human/`

### Option 2: Use Existing Datasets
- Download human voice datasets from Mozilla Common Voice
- Generate AI samples using free online TTS tools
- Ensure both folders have similar numbers of samples

## 🏋️ Training

Train the detection model:

```bash
python train.py
```

The script will:
- Load audio files from `data/raw/human/` and `data/raw/ai/`
- Extract acoustic features
- Train a Random Forest classifier
- Save the model to `models/`

## 🌐 API Usage

### Start the API Server

```bash
python api.py
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### Health Check
```bash
GET /health
```

#### Detect AI Voice
```bash
POST /detect
Content-Type: multipart/form-data

# Form data:
audio: <audio_file>
```

**Response:**
```json
{
  "is_ai_generated": true,
  "confidence": 0.97,
  "ai_probability": 0.97,
  "human_probability": 0.03
}
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

## 🧪 Testing

Test the trained model:

```bash
python test_api.py
```

## 📁 Project Structure

```
deepfake-detection-api/
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 .env.example
├── 📄 .gitignore
├── 🐍 train.py              # Model training script
├── 🐍 api.py                # FastAPI server
├── 🐍 data_generator.py     # Data generation utilities
├── 🐍 test_api.py           # API testing script
├── 📂 config/
│   ├── 🐍 __init__.py
│   └── 🐍 settings.py       # Configuration settings
├── 📂 src/
│   ├── 🐍 __init__.py
│   ├── 📂 features/
│   │   ├── 🐍 __init__.py
│   │   └── 🐍 audio_features.py  # Feature extraction
│   ├── 📂 models/
│   │   ├── 🐍 __init__.py
│   │   └── 🐍 train.py      # Training utilities
│   └── 📂 api/
│       ├── 🐍 __init__.py
│       └── 🐍 main.py       # API endpoints
├── 📂 data/
│   ├── 📂 raw/
│   │   ├── 📂 human/        # Human voice samples
│   │   └── 📂 ai/           # AI voice samples
│   └── 📂 processed/        # Processed features
├── 📂 models/               # Trained models (.pkl files)
├── 📂 scripts/              # Utility scripts
└── 📂 tests/                # Unit tests
```

## 🔧 Configuration

Key settings in `config/settings.py`:

- **Audio Processing**: Sample rate, length limits
- **Feature Extraction**: MFCC parameters, window sizes
- **Model Training**: Test split, random seed
- **API Settings**: Host, port, file size limits

## 🚀 Deployment

### Docker (Recommended)

```bash
# Build image
docker build -t ai-voice-detector .

# Run container
docker run -p 8000:8000 ai-voice-detector
```

### Production Considerations

- Use a production WSGI server (Gunicorn)
- Set up proper logging and monitoring
- Implement rate limiting
- Add authentication if needed
- Use a reverse proxy (nginx)

## 🧬 Extending the System

### Adding New AI Voice Generators

1. Generate samples using the new AI system
2. Place samples in `data/raw/ai/`
3. Retrain the model with `python train.py`

### Improving Accuracy

- **More Data**: Increase training samples (100+ per category)
- **Diverse Voices**: Multiple speakers, accents, languages
- **Feature Engineering**: Add new acoustic features
- **Advanced Models**: Try neural networks, ensemble methods

### Supported AI Voice Generators

Currently optimized for:
- ✅ ElevenLabs
- 🔄 OpenAI TTS (in progress)
- 🔄 Google Cloud TTS (planned)
- 🔄 Azure Cognitive Services (planned)

## 📈 Roadmap

- [ ] Support for more AI voice generators
- [ ] Real-time audio stream processing
- [ ] Web interface for easy testing
- [ ] Model confidence calibration
- [ ] Multi-language support
- [ ] Advanced neural network models
- [ ] Batch processing API

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Ethical Considerations

This tool is designed to help detect AI-generated audio for:
- **Media Verification**: Identifying synthetic content
- **Security**: Preventing voice-based fraud
- **Research**: Understanding AI voice generation

**Please use responsibly** and in compliance with applicable laws and regulations.

## 🙏 Acknowledgments

- [Mozilla Common Voice](https://commonvoice.mozilla.org/) for human voice datasets
- [ElevenLabs](https://elevenlabs.io/) for high-quality AI voice generation
- [Librosa](https://librosa.org/) for audio processing capabilities
- [FastAPI](https://fastapi.tiangolo.com/) for the excellent API framework

---

**Built with ❤️ for a safer digital world**