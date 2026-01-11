# Polish Voice Command Recognition & Speaker Identification System

A deep learning system for recognizing 15 Polish voice commands and identifying speakers using Multi-View CNN and LSTM architectures.

## Features

- **Multi-View CNN Architecture**: Uses 3 parallel CNN branches processing different spectrogram representations (standard, smooth, and mel spectrograms) for command recognition
- **LSTM Speaker Identification**: Identifies speakers using x-vector style embeddings from filterbank features
- **15 Polish Commands**: Recognizes commands for directions, control, responses, audio, and power
- **2 Speaker Identification**: Distinguishes between different speakers (Filip, Dzmitry)
- **Parallel Inference**: Runs command and speaker predictions simultaneously for optimal speed
- **Web Interface**: Easy-to-use Gradio web interface with real-time microphone recording
- **PyTorch Models**: Both models implemented in PyTorch for consistency and performance
- **Docker Support**: Containerized deployment for easy setup

## Supported Commands

### Directions (Kierunki)
- **GÓRA** - Up
- **DÓŁ** - Down
- **LEWO** - Left
- **PRAWO** - Right
- **DO_PRZODU** - Forward
- **COFNIJ** - Back

### Control (Kontrola)
- **START** - Start
- **STOP** - Stop
- **OD_NOWA** - Restart

### Responses (Odpowiedzi)
- **TAK** - Yes
- **NIE** - No

### Audio (Dźwięk)
- **GŁOŚNIEJ** - Louder
- **CISZEJ** - Quieter

### Power (Zasilanie)
- **WŁĄCZ** - Turn on
- **WYŁĄCZ** - Turn off

## Installation

### Option 1: Docker (Recommended)

#### Prerequisites
- Docker installed on your system
- Trained models in `dataset/models/`:
  - `cnn_model.h5` - Command recognition model
  - `label_encoder.pkl` - Command label encoder
  - `lstm_model.h5` - Speaker identification model
  - `speaker_label_encoder.pkl` - Speaker label encoder

#### Build and Run

**Using Docker Compose (Easiest):**
```bash
# Start the application
docker-compose up

# Run in background
docker-compose up -d

# Stop the application
docker-compose down
```

**Using Docker directly:**
```bash
# Build the Docker image
docker build -t voice-command-recognition .

# Run the container
docker run -p 7860:7860 voice-command-recognition
```

The web interface will be available at: **http://localhost:7860**

### Option 2: Manual Installation

#### Prerequisites
- Python >= 3.12
- UV package manager (or pip)
- Microphone access (for live recording)

#### Install Dependencies

Using UV (recommended):
```bash
uv sync
```

Or using pip:
```bash
pip install gradio librosa matplotlib numpy plotly scikit-learn torch torchaudio tqdm pydantic-settings
```

**Note for WSL users**: If you encounter I/O errors with `uv sync`, you may need to:
1. Delete the `.venv` directory from Windows Explorer (not WSL terminal)
2. Run `uv sync` again
3. Or use pip in a Python virtual environment instead

## Usage

### 1. Launch the Web Interface

**Using Docker Compose (Recommended):**
```bash
docker-compose up
```

**Using Docker:**
```bash
docker run -p 7860:7860 voice-command-recognition
```

**Without Docker:**
```bash
uv run python src/command_recognition/gradio_app.py
```

Or:
```bash
python src/command_recognition/gradio_app.py
```

The web interface will be available at: **http://localhost:7860**

### 2. Using the Interface

1. **Record Audio**: Click the microphone button to record your Polish voice command (approximately 2 seconds)
2. **Upload Audio** (optional): Or upload a WAV file instead of recording
3. **Recognize Command**: Click "Rozpoznaj" to process the audio
4. **View Results**:
   - **🎯 Rozpoznane Polecenie**:
     - Command name
     - Confidence percentage
     - Top 3 predictions with probabilities
   - **🗣️ Rozpoznany Mówca**:
     - Speaker name
     - Confidence percentage

### 3. Tips for Best Results

- Speak clearly and at a normal volume
- Record in a quiet environment
- Say the command once, holding for about 2 seconds
- Make sure your microphone is working and browser has permission to access it
- Ensure audio is at least 0.5 seconds long

## Project Structure

```
command-model/
├── src/
│   └── command_recognition/
│       ├── audio/
│       │   ├── audio_processor.py      # Audio preprocessing
│       │   ├── audio_generator.py      # Audio data generation
│       │   └── extract_features.py     # Feature extraction (spectrograms, filterbanks)
│       ├── models/
│       │   ├── cnn.py                  # Multi-View CNN model (PyTorch)
│       │   └── lstm.py                 # LSTM speaker identification (PyTorch)
│       ├── utils/
│       │   └── enums.py                # Command and speaker enumerations
│       ├── config.py                   # Configuration
│       └── gradio_app.py               # Web interface with dual prediction
├── dataset/
│   ├── models/
│   │   ├── cnn_model.h5                # Trained CNN model
│   │   ├── label_encoder.pkl           # Command label encoder
│   │   ├── lstm_model.h5               # Trained LSTM model
│   │   └── speaker_label_encoder.pkl   # Speaker label encoder
│   ├── person_1/                       # Training audio samples (Filip)
│   └── person_2/                       # Training audio samples (Dzmitry)
├── Dockerfile                          # Docker configuration
├── docker-compose.yml                  # Docker Compose configuration
├── .dockerignore                       # Docker ignore file
├── pyproject.toml                      # Python dependencies
└── README.md                           # This file
```

## Technical Details

### Command Recognition Model

- **Type**: Multi-View Convolutional Neural Network (PyTorch)
- **Input**: Three 64x64 spectrograms (standard, smooth, mel)
- **Architecture**:
  - 3 parallel CNN branches with 5 convolutional layers each
  - Each branch: Conv2D → BatchNorm → ReLU → MaxPool (5 blocks)
  - Filters: 48 → 96 → 192 → 192 → 192
- **Fusion**: Concatenates all branch outputs (9,216 features)
- **Classification**: FC(512) → Dropout(0.5) → FC(15 commands)
- **Output**: 15 command classes with confidence scores

### Speaker Identification Model

- **Type**: LSTM Network with x-vector style embeddings (PyTorch)
- **Input**: Variable-length filterbank features (24 filters)
- **Architecture**:
  - 2 LSTM layers (512 units each)
  - Statistics pooling (mean + std concatenation)
  - FC(512) → BatchNorm → Embedding(256)
  - Speaker classifier (2 classes)
- **Output**: Speaker ID with confidence score

### Audio Processing

- **Sample Rate**: 16,000 Hz
- **Duration**: 2,000 ms (normalized, padded/trimmed)
- **Preprocessing**: Resampling, mono conversion, duration normalization
- **Feature Extraction**:
  - **Standard Spectrogram** (n_fft=128, 64x64)
  - **Smooth Spectrogram** (n_fft=1024, averaged, 64x64)
  - **Mel Spectrogram** (64 mel bands, 64x64)
  - **Filterbank** (24 filters, variable time steps)

### Training Parameters

**CNN (Command Recognition):**
- **Optimizer**: Adam (lr=0.0003)
- **Loss**: CrossEntropyLoss
- **Epochs**: 20
- **Batch Size**: 32

**LSTM (Speaker Identification):**
- **Optimizer**: Adam (lr=0.001)
- **Loss**: CrossEntropyLoss
- **Epochs**: 20
- **Batch Size**: 16

### Parallel Inference

The system uses `ThreadPoolExecutor` to run both models simultaneously:
1. Audio preprocessing (sequential)
2. Feature extraction (sequential)
3. **CNN and LSTM predictions (parallel)** ⚡
4. Results aggregation

This reduces total inference time compared to sequential execution.

## Development

### Training the Command Recognition Model

To retrain the CNN model with your own data:

```bash
uv run python src/command_recognition/models/cnn.py
```

Or without uv:
```bash
python src/command_recognition/models/cnn.py
```

### Training the Speaker Identification Model

To train the LSTM model:

```bash
uv run python src/command_recognition/models/lstm.py
```

Or without uv:
```bash
python src/command_recognition/models/lstm.py
```

### Audio Preprocessing

To normalize audio files in the dataset:

```bash
python src/command_recognition/audio/audio_processor.py
```

### Adding New Commands

1. Add the command to `src/command_recognition/utils/enums.py`:
```python
class Commands(Enum):
    YOUR_COMMAND = 16  # Next available ID
```

2. Record audio samples in `dataset/person_X/your_command/`
3. Update `num_commands` in `config.py`
4. Retrain the CNN model

### Adding New Speakers

1. Add the speaker to `src/command_recognition/utils/enums.py`:
```python
class Speakers(Enum):
    NEW_SPEAKER = 3  # Next available ID
```

2. Create `dataset/person_3/` with command recordings
3. Update `num_speakers` in `config.py`
4. Retrain the LSTM model

## Troubleshooting

### "Brak nagrania audio" (No audio recording)
- Make sure you clicked the microphone button and recorded audio
- Check browser permissions for microphone access

### "Zbyt krótkie nagranie" (Recording too short)
- Record for at least 0.5 seconds
- Speak the command clearly

### "Brak dźwięku" (Silent recording)
- Check microphone volume settings
- Speak closer to the microphone
- Test microphone with other applications

### Model not loading
- Verify model files exist at:
  - `dataset/models/cnn_model.h5`
  - `dataset/models/label_encoder.pkl`
  - `dataset/models/lstm_model.h5`
  - `dataset/models/speaker_label_encoder.pkl`
- Check file permissions
- Ensure models are trained before running the app

### Docker issues
- Make sure Docker is installed and running
- Check port 7860 is not already in use
- Verify model files are in `dataset/models/` before building

### Dependencies installation issues
- Try using pip instead of uv
- Create a fresh virtual environment
- On WSL, try installing from Windows PowerShell instead

## Performance

- **Command Recognition Accuracy**: ~95%+ (on validation set)
- **Speaker Identification Accuracy**: ~90%+ (on validation set)
- **Inference Time**: ~100-200ms per prediction (parallel execution)
- **Model Sizes**:
  - CNN: ~29 MB
  - LSTM: ~5-10 MB

## Docker Commands

### Using Docker Compose

```bash
# Start the application
docker-compose up

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the application
docker-compose down

# Rebuild and start
docker-compose up --build
```

### Using Docker Directly

```bash
# Build the image
docker build -t voice-command-recognition .

# Run the container
docker run -p 7860:7860 voice-command-recognition

# Run with GPU support (NVIDIA Docker)
docker run --gpus all -p 7860:7860 voice-command-recognition

# Run in background (detached mode)
docker run -d -p 7860:7860 voice-command-recognition

# View logs
docker logs <container_id>

# Stop the container
docker stop <container_id>
```

## License

This project is for educational purposes.

## Authors

- MrSpokeMan (Filip)
- Dzmitry

## Acknowledgments

- **PyTorch** for deep learning framework
- **Gradio** for the web interface
- **librosa** for audio processing
- **scikit-learn** for preprocessing utilities
- Multi-View CNN inspired by multi-modal learning research
- x-vector embeddings inspired by speaker verification research

## Future Improvements

- [ ] Add more speakers for robust speaker identification
- [ ] Implement real-time streaming recognition
- [ ] Add noise robustness training
- [ ] Support for more languages
- [ ] Mobile app deployment
- [ ] Model quantization for faster inference
- [ ] Add voice activity detection (VAD)
