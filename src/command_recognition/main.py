from command_recognition.audio.extract_features import AudioFeatureExtractor
import matplotlib.pyplot as plt

from command_recognition.config import get_config
config = get_config()


if __name__ == "__main__":
    # audio_generator = SyntheticAudioGenerator()
    audio_feature_extractor = AudioFeatureExtractor()

    data = audio_feature_extractor.load_dataset(config.dataset_dir)
    
    print(data['X_audio'][0].shape)
    
    # Display sepctograms
    spectrogram = audio_feature_extractor.get_spectrogram(data['X_audio'][0])
    smooth_spectrogram = audio_feature_extractor.get_smooth_spectrogram(data['X_audio'][0])
    mel_spectrogram = audio_feature_extractor.get_mel_spectrogram(data['X_audio'][0])
    
    # Display spectrograms
    plt.figure(figsize=(10, 4))
    plt.subplot(2, 2, 1)
    plt.title('Spectrogram')
    plt.imshow(spectrogram, aspect='auto', origin='lower')
    plt.colorbar()
    plt.subplot(2, 2, 2)
    plt.title('Smooth Spectrogram')
    plt.imshow(smooth_spectrogram, aspect='auto', origin='lower')
    plt.colorbar()
    plt.subplot(2, 2, 3)
    plt.title('Mel Spectrogram')
    plt.imshow(mel_spectrogram, aspect='auto', origin='lower')
    plt.colorbar()
    plt.show()