import numpy as np
import librosa
from tqdm import tqdm

from command_recognition.config import get_config
from command_recognition.utils.enums import Commands

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
                speaker_id = int(speaker_dir.name.replace('person_', ''))
                
                for command_dir in speaker_dir.iterdir():
                    if command_dir.is_dir():
                        command_id = int(Commands[command_dir.name.upper()].value)
                        
                        for file in command_dir.iterdir():
                            if file.suffix == '.npy':
                                audio = np.load(file)
                                X_audio.append(audio)
                                y_commands.append(command_id)
                                y_speakers.append(speaker_id)
                            elif file.suffix in config.audio_extensions:
                                audio, _ = librosa.load(file, sr=16000)
                                X_audio.append(audio)
                                y_commands.append(command_id)
                                y_speakers.append(speaker_id)
        
        return {
            'X_audio': X_audio,
            'y_commands': np.array(y_commands, dtype=str),
            'y_speakers': np.array(y_speakers, dtype=np.int32)
        }
        
    def get_spectrogram(self, audio):
        n_fft = 128
        target_frames = 64
        hop_length = int(len(audio) / target_frames)
        
        spec = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, center=False))
        spec = spec[:64, :64]
        
        spec_db = librosa.power_to_db(spec, ref=np.max)
        spec_norm = (spec_db + 80) / 80
        spec_norm = np.clip(spec_norm, 0, 1)
        
        return spec_norm

    def get_smooth_spectrogram(self, audio):
        n_fft = 1024
        target_frames = 64
        hop_length = int(len(audio) / target_frames)
        
        spec = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, center=False))
        spec = spec[:512, :]
        
        if spec.shape[1] < target_frames:
            pad_width = target_frames - spec.shape[1]
            spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant')
        else:
            spec = spec[:, :target_frames]
        
        reshaped_spec = spec.reshape(64, 8, 64)
        smooth_spec = np.mean(reshaped_spec, axis=1)
        
        spec_db = librosa.power_to_db(smooth_spec, ref=np.max)
        spec_norm = (spec_db + 80) / 80
        spec_norm = np.clip(spec_norm, 0, 1)
        
        return spec_norm

    def get_mel_spectrogram(self, audio):
        n_fft = 1024
        n_mels = 64
        target_frames = 64
        hop_length = int(len(audio) / target_frames)
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=self.sr, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            n_mels=n_mels, 
            center=False
        )
        
        mel_spec = mel_spec[:, :64]
        
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_norm = (mel_spec_db + 80) / 80
        mel_spec_norm = np.clip(mel_spec_norm, 0, 1)
        
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
        X_smooth = []
        X_spec = []
        
        for i, audio in tqdm(enumerate(X_audio), total=len(X_audio), desc="Extracting features", unit="sample"):
            mel_spec = self.get_mel_spectrogram(audio)
            spec_smooth = self.get_smooth_spectrogram(audio)
            spec = self.get_spectrogram(audio)
            X_mel.append(mel_spec)
            X_smooth.append(spec_smooth)
            X_spec.append(spec)
            
            filterbank = self.get_filterbank(audio)
            X_filterbank.append(filterbank)
        
        print(f"Feature extraction completed")
        
        return {
            'mel_specs': np.array(X_mel),
            'filterbanks': X_filterbank,
            'smooth_specs': np.array(X_smooth),
            'specs': np.array(X_spec)
        }
        
        
if __name__ == "__main__":
    extractor = AudioFeatureExtractor(sample_rate=16000)
    data = extractor.load_dataset(config.dataset_dir)
    features = extractor.extract_features(data['X_audio'])
    
    X_mel = features['mel_specs']  # (N, 64, 99)
    X_filterbank = features['filterbanks']  # List of (T, 24) arrays
    y_commands = data['y_commands']
    y_speakers = data['y_speakers']