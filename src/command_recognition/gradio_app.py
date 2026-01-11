"""
Gradio Web Interface for Polish Voice Command Recognition System
Simple interface: Record -> Playback -> Predict
"""

import gradio as gr
import numpy as np
import librosa
import torch
import logging
from concurrent.futures import ThreadPoolExecutor

from command_recognition.models.cnn import CommandRecognitionCNN
from command_recognition.models.lstm import SpeakerIdentificationLSTM
from command_recognition.audio.extract_features import AudioFeatureExtractor
from command_recognition.config import get_config
from command_recognition.utils.enums import Commands, Speakers

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global model instance
model_inference = None


class ModelInference:
    """Manages model loading and inference"""

    def __init__(self):
        logger.info("Initializing Model Inference...")
        self.config = get_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize model
        self.model = CommandRecognitionCNN(num_commands=self.config.num_commands)
        self.lstm = SpeakerIdentificationLSTM(num_speakers=self.config.num_speakers)

        # Load trained weights and label encoder
        model_path = self.config.cnn_model_path
        label_encoder_path = self.config.cnn_model_path.parent / "label_encoder.pkl"

        lstm_model_path = self.config.lstm_model_path
        lstm_label_encoder_path = self.config.lstm_model_path.parent / "speaker_label_encoder.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"CNN model file not found: {model_path}")
        if not label_encoder_path.exists():
            raise FileNotFoundError(f"CNN label encoder not found: {label_encoder_path}")
        if not lstm_model_path.exists():
            raise FileNotFoundError(f"LSTM model file not found: {lstm_model_path}")
        if not lstm_label_encoder_path.exists():
            raise FileNotFoundError(f"LSTM label encoder not found: {lstm_label_encoder_path}")

        self.model.load_model(model_path=model_path, label_encoder_path=label_encoder_path)
        self.lstm.load_model(model_path=lstm_model_path, label_encoder_path=lstm_label_encoder_path)
        logger.info("Models loaded successfully!")

        # Initialize feature extractor
        self.extractor = AudioFeatureExtractor(sample_rate=16000)

        # Get command names from label encoder (these will be numeric strings like "1", "2", etc.)
        self.label_encoder_classes = self.model.get_class_names()
        self.lstm_label_encoder_classes = self.lstm.get_class_names()

        # Create mapping from numeric ID to command name
        self.id_to_command = {str(cmd.value): cmd.name for cmd in Commands}
        self.id_to_speaker = {str(speaker.value): speaker.name for speaker in Speakers}

        # Create ordered list of command names matching label encoder order
        self.command_names = []
        for label in self.label_encoder_classes:
            command_name = self.id_to_command.get(str(label), f"UNKNOWN_{label}")
            self.command_names.append(command_name)

        self.speaker_names = []
        for label in self.lstm_label_encoder_classes:
            speaker_name = self.id_to_speaker.get(str(label), f"UNKNOWN_{label}")
            self.speaker_names.append(speaker_name)

        logger.info(f"Loaded {len(self.command_names)} commands: {self.command_names}")
        logger.info(f"Loaded {len(self.speaker_names)} speakers: {self.speaker_names}")


def preprocess_audio(audio_data, original_sr):
    """
    Normalize audio: mono, 16kHz, 2 seconds (32000 samples)
    """
    # Convert to float32
    if audio_data.dtype != np.float32:
        if np.issubdtype(audio_data.dtype, np.integer):
            audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
        else:
            audio_data = audio_data.astype(np.float32)

    # Convert stereo to mono
    if audio_data.ndim == 2:
        audio_data = np.mean(audio_data, axis=1)

    # Resample to 16kHz
    if original_sr != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=original_sr, target_sr=16000)

    # Normalize to 2 seconds (32000 samples)
    target_length = 32000
    if len(audio_data) < target_length:
        audio_data = np.pad(audio_data, (0, target_length - len(audio_data)), mode='constant')
    elif len(audio_data) > target_length:
        audio_data = audio_data[:target_length]

    return audio_data


def extract_features(audio):
    """Extract all features: 3 spectrograms for CNN and filterbank for LSTM"""
    extractor = model_inference.extractor

    # Extract all features
    spec = extractor.get_spectrogram(audio)
    smooth_spec = extractor.get_smooth_spectrogram(audio)
    mel_spec = extractor.get_mel_spectrogram(audio)
    filterbank = extractor.get_filterbank(audio)

    # Add batch dimension for spectrograms: (64, 64) -> (1, 64, 64)
    spec = np.expand_dims(spec, axis=0)
    smooth_spec = np.expand_dims(smooth_spec, axis=0)
    mel_spec = np.expand_dims(mel_spec, axis=0)

    return (spec, smooth_spec, mel_spec), filterbank


def predict_command_and_speaker(audio_input):
    """
    Main prediction function for Gradio.
    Predicts both command (CNN) and speaker (LSTM) in parallel.
    Returns: (predicted_command, command_confidence, top3_commands, predicted_speaker, speaker_confidence)
    """
    if audio_input is None:
        return "Brak nagrania", "0%", "Nagraj lub wgraj plik audio", "N/A", "0%"

    try:
        sample_rate, audio_data = audio_input

        # Validate audio
        duration = len(audio_data) / sample_rate
        if duration < 0.3:
            return "Zbyt krótkie nagranie", "0%", "Nagraj co najmniej 0.5 sekundy", "N/A", "0%"

        if np.max(np.abs(audio_data)) < 0.001:
            return "Brak dźwięku", "0%", "Nagranie jest ciche lub puste", "N/A", "0%"

        # Preprocess
        audio_processed = preprocess_audio(audio_data, sample_rate)

        # Extract features
        spectrograms, filterbank = extract_features(audio_processed)

        # Run predictions in parallel for speed
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both predictions
            command_future = executor.submit(model_inference.model.predict, spectrograms)
            speaker_future = executor.submit(model_inference.lstm.predict, filterbank)

            # Get results
            command_probabilities = command_future.result()[0]
            speaker_probabilities = speaker_future.result()[0]

        # Command results
        predicted_cmd_idx = np.argmax(command_probabilities)
        predicted_command = model_inference.command_names[predicted_cmd_idx]
        command_confidence = command_probabilities[predicted_cmd_idx] * 100

        # Top 3 command predictions
        top3_cmd_indices = np.argsort(command_probabilities)[::-1][:3]
        top3_text = "\n".join([
            f"{i+1}. {model_inference.command_names[idx]} ({command_probabilities[idx]*100:.1f}%)"
            for i, idx in enumerate(top3_cmd_indices)
        ])

        # Speaker results
        predicted_spk_idx = np.argmax(speaker_probabilities)
        predicted_speaker = model_inference.speaker_names[predicted_spk_idx]
        speaker_confidence = speaker_probabilities[predicted_spk_idx] * 100

        return (
            predicted_command,
            f"{command_confidence:.1f}%",
            top3_text,
            predicted_speaker,
            f"{speaker_confidence:.1f}%"
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return f"Błąd: {str(e)}", "0%", "Spróbuj ponownie", "N/A", "0%"


def create_interface():
    """Create simple Gradio interface"""

    with gr.Blocks(title="Rozpoznawanie Poleceń i Mówcy", theme=gr.themes.Soft()) as demo:

        gr.Markdown("# 🎤 Rozpoznawanie Poleceń Głosowych i Identyfikacja Mówcy")
        gr.Markdown("Nagraj polskie polecenie (~2 sekundy) i kliknij **Rozpoznaj** aby zidentyfikować polecenie i mówcę")

        with gr.Row():
            with gr.Column(scale=1):
                # Audio input with microphone and upload
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="numpy",
                    label="Nagraj lub wgraj audio",
                    waveform_options=gr.WaveformOptions(
                        waveform_color="#3b82f6",
                        waveform_progress_color="#1d4ed8"
                    )
                )

                predict_btn = gr.Button("Rozpoznaj", variant="primary", size="lg")

                gr.Markdown("""
                ### Obsługiwane polecenia:
                `GÓRA` `DÓŁ` `LEWO` `PRAWO` `DO_PRZODU` `COFNIJ`
                `START` `STOP` `OD_NOWA` `TAK` `NIE`
                `GŁOŚNIEJ` `CISZEJ` `WŁĄCZ` `WYŁĄCZ`
                """)

            with gr.Column(scale=1):
                # Command Results
                gr.Markdown("### 🎯 Rozpoznane Polecenie")
                prediction_output = gr.Textbox(
                    label="Polecenie",
                    interactive=False
                )

                confidence_output = gr.Textbox(
                    label="Pewność",
                    interactive=False
                )

                top3_output = gr.Textbox(
                    label="Top 3 predykcje",
                    interactive=False,
                    lines=3
                )

                # Speaker Results
                gr.Markdown("### 🗣️ Rozpoznany Mówca")
                speaker_output = gr.Textbox(
                    label="Mówca",
                    interactive=False
                )

                speaker_confidence_output = gr.Textbox(
                    label="Pewność",
                    interactive=False
                )

        # Connect button
        predict_btn.click(
            fn=predict_command_and_speaker,
            inputs=audio_input,
            outputs=[
                prediction_output,
                confidence_output,
                top3_output,
                speaker_output,
                speaker_confidence_output
            ]
        )

        gr.Markdown("---")
        gr.Markdown("*Polecenia: Multi-View CNN | Mówca: LSTM | PyTorch | 16kHz | 2000ms*")

    return demo


def main():
    """Launch the Gradio app"""
    global model_inference

    try:
        logger.info("=" * 60)
        logger.info("Starting Voice Command Recognition & Speaker ID System")
        logger.info("=" * 60)

        # Load model
        model_inference = ModelInference()

        # Create and launch interface
        demo = create_interface()

        logger.info("Launching Gradio...")
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False
        )

    except FileNotFoundError as e:
        logger.error(f"Model files not found: {e}")
        print(f"\nBłąd: {e}")
        print("Upewnij się, że model jest wytrenowany w dataset/models/")
    except Exception as e:
        logger.error(f"Failed to start: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
