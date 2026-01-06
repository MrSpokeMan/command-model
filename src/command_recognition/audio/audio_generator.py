import numpy as np
from scipy import signal
from tqdm import tqdm
from pathlib import Path

from command_recognition.config import get_config

config = get_config()

class SyntheticAudioGenerator:
    def __init__(self, sample_rate=16000):
        self.sr = sample_rate
        self.dataset = self.create_dataset(config.num_speakers, config.num_commands, config.samples_per_class)
        self.save_dataset(self.dataset, config.dataset_dir)
        
    def _generate_command_audio(self, command_id, speaker_id, duration=2.0, noise_level=0.01):
        t = np.linspace(0, duration, int(self.sr * duration), False)
        
        # Base frequency depends on command (100-300 Hz for speech)
        base_freq = 150 + command_id * 20
        
        # Speaker adds variation (pitch difference)
        speaker_variation = 1.0 + (speaker_id * 0.15)
        freq = base_freq * speaker_variation
        
        # Generate fundamental frequency
        fundamental = 0.5 * np.sin(2 * np.pi * freq * t)
        
        # Add harmonics (mimic natural speech)
        harmonic1 = 0.3 * np.sin(2 * np.pi * freq * 2 * t)
        harmonic2 = 0.15 * np.sin(2 * np.pi * freq * 3 * t)
        
        # Combine
        audio = fundamental + harmonic1 + harmonic2
        
        # Add amplitude envelope (speech-like attack/decay)
        envelope = signal.get_window('hann', len(audio))
        audio = audio * envelope

        noise = np.random.normal(0, noise_level, len(audio))
        audio = audio + noise

        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        return audio.astype(np.float32)
    
    def create_dataset(self, num_speakers=2, num_commands=15, samples_per_class=10):
        print(f"Generating synthetic dataset...")
        print(f"Speakers: {num_speakers}, Commands: {num_commands}, Samples/class: {samples_per_class}")
        
        X_audio = []
        y_commands = []
        y_speakers = []
        
        total = num_speakers * num_commands * samples_per_class

        with tqdm(total=total, desc="Generating synthetic audio", unit="sample") as pbar:
            for speaker_id in range(num_speakers):
                for command_id in range(num_commands):
                    for _ in range(samples_per_class):
                        noise = 0.01 + np.random.uniform(0, 0.005)

                        audio = self._generate_command_audio(
                            command_id=command_id,
                            speaker_id=speaker_id,
                            duration=2.0,
                            noise_level=noise
                        )
                        
                        X_audio.append(audio)
                        y_commands.append(command_id)
                        y_speakers.append(speaker_id)

                        pbar.update(1)

        print(f"Generated {total} samples")
        
        return {
            'X_audio': np.array(X_audio),
            'y_commands': np.array(y_commands),
            'y_speakers': np.array(y_speakers)
        }
        
    def save_dataset(self, dataset, path):
        base_path = Path(path)
        base_path.mkdir(parents=True, exist_ok=True)

        X_audio = dataset['X_audio']
        y_commands = dataset['y_commands']
        y_speakers = dataset['y_speakers']

        num_samples = X_audio.shape[0]
        print(f"Saving {num_samples} samples to {base_path}...")

        for idx in tqdm(range(num_samples), desc="Saving samples", unit="sample"):
            speaker_id = int(y_speakers[idx])
            command_id = int(y_commands[idx])

            speaker_dir = base_path / str(speaker_id)
            command_dir = speaker_dir / f"command_{command_id}"
            command_dir.mkdir(parents=True, exist_ok=True)

            file_path = command_dir / f"sample_{idx}.npy"
            np.save(file_path, X_audio[idx])