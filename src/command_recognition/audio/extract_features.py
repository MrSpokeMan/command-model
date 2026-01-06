import numpy as np
import librosa
from tqdm import tqdm

from command_recognition.config import get_config

from pathlib import Path
config = get_config()

class AudioFeatureExtractor:    
    def __init__(self, sample_rate=16000):
        self.sr = sample_rate
        self.n_mel = 64
        self.n_mfcc = 13
        
    def load_dataset(self, path):
        base_path = Path(path)
        
        X_audio = []
        y_commands = []
        y_speakers = []
        
        for speaker_dir in base_path.iterdir():
            if speaker_dir.is_dir():
                speaker_id = int(speaker_dir.name)
                
                for command_dir in speaker_dir.iterdir():
                    if command_dir.is_dir():
                        command_id = int(command_dir.name.replace('command_', ''))
                        
                        for file in command_dir.iterdir():
                            if file.suffix == '.npy':
                                audio = np.load(file)
                                X_audio.append(audio)
                                y_commands.append(command_id)
                                y_speakers.append(speaker_id)
        
        return {
            'X_audio': np.array(X_audio),
            'y_commands': np.array(y_commands, dtype=np.int32),
            'y_speakers': np.array(y_speakers, dtype=np.int32)
        }
        
    def get_mel_spectrogram(self, audio, target_shape=(64, 99)):
        """Extract mel-spectrogram for CNN"""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=self.n_mel,
            n_fft=512,
            hop_length=160,
            fmin=0,
            fmax=8000
        )

        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Normalize to [0, 1]
        mel_spec_norm = (mel_spec_db + 80) / 80
        mel_spec_norm = np.clip(mel_spec_norm, 0, 1)
        
        # Pad or truncate to target shape
        if mel_spec_norm.shape[1] < target_shape[1]:
            pad_width = ((0, 0), (0, target_shape[1] - mel_spec_norm.shape[1]))
            mel_spec_norm = np.pad(mel_spec_norm, pad_width, mode='constant')
        else:
            mel_spec_norm = mel_spec_norm[:, :target_shape[1]]
        
        return mel_spec_norm
    
    def get_filterbank(self, audio, n_filters=24):
        """Extract filterbanks for LSTM (speaker identification)"""
        # STFT
        spec = np.abs(librosa.stft(audio, n_fft=512, hop_length=160))
        
        # Mel-filterbanks
        mel_fb = librosa.filters.mel(
            sr=self.sr,
            n_fft=512,
            n_mels=n_filters
        )
        
        filterbank = np.dot(mel_fb, spec)
        filterbank = np.log(filterbank + 1e-9)
        
        # Transpose to (time_steps, n_filters)
        return filterbank.T.astype(np.float32)
    
    def extract_features(self, X_audio):
        print("Extracting features...")
        
        X_mel = []
        X_filterbank = []
        
        for i, audio in tqdm(enumerate(X_audio), total=len(X_audio), desc="Extracting features", unit="sample"):
            mel_spec = self.get_mel_spectrogram(audio)
            X_mel.append(mel_spec)
            
            filterbank = self.get_filterbank(audio)
            X_filterbank.append(filterbank)
        
        print(f"Feature extraction completed")
        
        return {
            'mel_specs': np.array(X_mel),
            'filterbanks': X_filterbank
        }
        
        
if __name__ == "__main__":
    extractor = AudioFeatureExtractor(sample_rate=16000)
    data = extractor.load_dataset(config.dataset_dir)
    features = extractor.extract_features(data['X_audio'])
    
    X_mel = features['mel_specs']  # (N, 64, 99)
    X_filterbank = features['filterbanks']  # List of (T, 24) arrays
    y_commands = data['y_commands']
    y_speakers = data['y_speakers']