from pathlib import Path
from pydub import AudioSegment
from tqdm import tqdm
import librosa
import logging

from command_recognition.config import get_config
config = get_config()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioProcessor:
    def __init__(self):
        self.extensions = config.audio_extensions
        self.duration_ms = config.duration_ms
        self.corrupted_files = []
        
    def _trim_audio(self, audio_path: Path, extension: str) -> bool:
        """
        Normalize audio file to specified duration.
        - Trims if longer than target duration
        - Pads with silence if shorter than target duration
        Returns True if successful, False otherwise.
        """
        try:
            audio = AudioSegment.from_file(audio_path, format=extension)
            audio_length = len(audio)
            
            if audio_length == self.duration_ms:
                logger.debug(f"Audio file {audio_path.name} already at target duration, skipping...")
                return True
            elif audio_length > self.duration_ms:
                # Trim audio if too long
                logger.debug(f"Trimming {audio_path.name} from {audio_length}ms to {self.duration_ms}ms")
                audio = audio[:self.duration_ms]
            else:
                # Pad with silence if too short
                silence_duration = self.duration_ms - audio_length
                logger.debug(f"Padding {audio_path.name} from {audio_length}ms to {self.duration_ms}ms (adding {silence_duration}ms silence)")
                silence = AudioSegment.silent(duration=silence_duration)
                audio = audio + silence
            
            audio.export(audio_path, format=extension)
            return True
        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")
            self.corrupted_files.append(str(audio_path))
            return False
    
    def load_audio_dataset(self):
        base_path = Path(config.dataset_dir)
        total_files = 0
        processed_files = 0
        skipped_files = 0
        
        for dir in tqdm(base_path.iterdir(), desc=f"Processing directories", unit="dir"):
            for subdir in dir.iterdir():
                if subdir.is_dir():
                    for file in subdir.iterdir():
                        if file.suffix in self.extensions:
                            total_files += 1
                            if self._trim_audio(file, file.suffix.replace('.', '')):
                                processed_files += 1
                            else:
                                skipped_files += 1
                        else:
                            logger.debug(f"Skipping {file} because it is not a valid audio file")
        
        # Summary report
        logger.info("\n" + "="*60)
        logger.info("PROCESSING SUMMARY")
        logger.info("="*60)
        logger.info(f"Total audio files found: {total_files}")
        logger.info(f"Successfully processed: {processed_files}")
        logger.info(f"Failed/Corrupted: {skipped_files}")
        
        if self.corrupted_files:
            logger.warning("\nCORRUPTED FILES (need to be re-recorded or deleted):")
            for corrupted_file in self.corrupted_files:
                logger.warning(f"  - {corrupted_file}")
        logger.info("="*60)

                
if __name__ == "__main__":
    audio_processor = AudioProcessor()
    audio_processor.load_audio_dataset()
    
    