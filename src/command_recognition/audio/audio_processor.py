from pathlib import Path
from pydub import AudioSegment
from tqdm import tqdm
import librosa

from command_recognition.config import get_config
config = get_config()


class AudioProcessor:
    def __init__(self):
        self.extensions = config.audio_extensions
        self.duration_ms = config.duration_ms
        
    def _trim_audio(self, audio_path: Path, extension: str) -> None:
        try:
            audio = AudioSegment.from_file(audio_path, format=extension)
            if len(audio) > self.duration_ms:
                audio = audio[:self.duration_ms]
            else:
                print(f"Audio file is less or equal to {self.duration_ms}ms, skipping...")
                return
            audio.export(audio_path, format=extension)
        except Exception as e:
            print(f"Error trimming {audio_path}: {e}")
    
    def load_audio_dataset(self):
        base_path = Path(config.dataset_dir)
        for dir in tqdm(base_path.iterdir(), desc=f"Processing directories", unit="dir"):
            for subdir in dir.iterdir():
                if subdir.is_dir():
                    for file in subdir.iterdir():
                        if file.suffix in self.extensions:
                            self._trim_audio(file, file.suffix.replace('.', ''))
                        else:
                            print(f"Skipping {file} because it is not a valid audio file")

                
if __name__ == "__main__":
    audio_processor = AudioProcessor()
    audio_processor.load_audio_dataset()
    
    