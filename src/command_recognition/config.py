from pydantic_settings import BaseSettings
from pathlib import Path


BASE_DIR = Path(__file__).parent.parent.parent

class Config(BaseSettings):
    num_speakers: int = 2
    num_commands: int = 15
    samples_per_class: int = 10
    
    epochs_cnn: int = 20
    batch_size_cnn: int = 32
    
    epochs_lstm: int = 20
    batch_size_lstm: int = 16
    
    cnn_model_path: Path = BASE_DIR / "dataset" / "models" / "cnn_model.h5"
    lstm_model_path: Path = BASE_DIR / "dataset" / "models" / "lstm_model.h5"
    
    dataset_dir: Path = BASE_DIR / "dataset"
    
    audio_extensions: list[str] = ['.wav', '.mp3', '.ogg', '.m4a', '.flac', '.aac', '.wma', '.aiff', '.au', '.raw', '.pcm']
    duration_ms: int = 2000
    
def get_config():
    return Config()