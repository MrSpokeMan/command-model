import numpy as np
import matplotlib.pyplot as plt
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from command_recognition.config import get_config
from sklearn.preprocessing import LabelEncoder

config = get_config()


class SpeakerLSTMNetwork(nn.Module):
    """PyTorch LSTM network for speaker identification with x-vector style embeddings"""

    def __init__(self, num_speakers=2, embedding_dim=256, feature_dim=24):
        super(SpeakerLSTMNetwork, self).__init__()
        self.num_speakers = num_speakers
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim

        # LSTM frame-level layers
        self.lstm1 = nn.LSTM(feature_dim, 512, batch_first=True)
        self.lstm2 = nn.LSTM(512, 512, batch_first=True)

        # Segment-level layers (after statistics pooling)
        self.fc1 = nn.Linear(1024, 512)  # 512*2 because of mean+std pooling
        self.bn1 = nn.BatchNorm1d(512)

        # Embedding layer
        self.embedding_layer = nn.Linear(512, embedding_dim)

        # Speaker classification layer
        self.classifier = nn.Linear(embedding_dim, num_speakers)

    def statistics_pooling(self, x, lengths=None):
        """
        Statistics pooling: concatenate mean and std across time dimension
        Args:
            x: (batch, time, features)
            lengths: actual sequence lengths (optional)
        Returns:
            pooled: (batch, features*2)
        """
        if lengths is not None:
            # Mask padding for accurate statistics
            mask = torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]
            mask = mask.unsqueeze(-1)

            # Compute mean and std only over non-padded values
            mean = (x * mask).sum(dim=1) / lengths.unsqueeze(-1).float()

            # For std, compute variance then sqrt
            diff = (x - mean.unsqueeze(1)) * mask
            var = (diff ** 2).sum(dim=1) / lengths.unsqueeze(-1).float()
            std = torch.sqrt(var + 1e-8)
        else:
            # Simple mean and std over all timesteps
            mean = torch.mean(x, dim=1)
            std = torch.std(x, dim=1)

        pooled = torch.cat([mean, std], dim=1)
        return pooled

    def forward(self, x, lengths=None, return_embedding=False):
        """
        Forward pass
        Args:
            x: (batch, time, features)
            lengths: actual sequence lengths (optional)
            return_embedding: if True, return embedding instead of classification
        """
        # LSTM layers
        x, _ = self.lstm1(x)
        x = F.relu(x)
        x, _ = self.lstm2(x)
        x = F.relu(x)

        # Statistics pooling
        x = self.statistics_pooling(x, lengths)

        # Segment-level processing
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # Embedding
        embedding = self.embedding_layer(x)
        embedding = F.relu(embedding)

        if return_embedding:
            return embedding

        # Classification
        output = self.classifier(embedding)
        return output


class SpeakerIdentificationLSTM:
    """LSTM-based speaker identification with x-vector style embeddings"""

    def __init__(self, num_speakers=2, embedding_dim=256, feature_dim=24):
        self.num_speakers = num_speakers
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize PyTorch model
        self.model = SpeakerLSTMNetwork(
            num_speakers=num_speakers,
            embedding_dim=embedding_dim,
            feature_dim=feature_dim
        ).to(self.device)

        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        self.label_encoder = LabelEncoder()

    def _collate_fn(self, batch):
        """Custom collate function to handle variable-length sequences"""
        sequences, labels = zip(*batch)

        # Get lengths
        lengths = torch.tensor([len(seq) for seq in sequences])

        # Pad sequences
        sequences_padded = pad_sequence(
            [torch.tensor(seq, dtype=torch.float32) for seq in sequences],
            batch_first=True,
            padding_value=0.0
        )

        labels = torch.tensor(labels, dtype=torch.long)

        return sequences_padded, labels, lengths

    def _prepare_dataloader(self, X_data, y_data, batch_size, shuffle=True):
        """Prepare DataLoader for variable-length sequences"""
        dataset = list(zip(X_data, y_data))
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn
        )
        return dataloader
    
    def get_class_names(self):
        """Get all class names in order"""
        if hasattr(self.label_encoder, 'classes_'):
            return self.label_encoder.classes_
        return None

    def train(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=16, label_encoder=None):
        """Train LSTM model"""
        print(f"\n[*] Training LSTM Model (Speaker Identification) on {self.device}...")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Unique speakers: {len(np.unique(y_train))}")

        # Store label encoder if provided
        if label_encoder is not None:
            self.label_encoder = label_encoder

        # Prepare DataLoaders
        train_loader = self._prepare_dataloader(X_train, y_train, batch_size, shuffle=True)
        val_loader = self._prepare_dataloader(X_val, y_val, batch_size, shuffle=False)

        # Setup optimizer and loss
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for sequences, labels, lengths in train_loader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                lengths = lengths.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(sequences, lengths)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Metrics
                running_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / total
            epoch_acc = correct / total

            # Validation
            val_loss, val_acc = self._evaluate(val_loader, criterion)

            # Update history
            self.history['loss'].append(epoch_loss)
            self.history['accuracy'].append(epoch_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)

            print(f"Epoch {epoch+1}/{epochs} - "
                  f"loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f} - "
                  f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")

        # Save model
        if not config.lstm_model_path.parent.exists():
            config.lstm_model_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), config.lstm_model_path)
        print(f"[+] Saved model to {config.lstm_model_path}")

        # Save label encoder if exists
        if self.label_encoder is not None:
            label_encoder_path = config.lstm_model_path.parent / "speaker_label_encoder.pkl"
            with open(label_encoder_path, 'wb') as f:
                pickle.dump(self.label_encoder, f)
            print(f"[+] Saved speaker label encoder to {label_encoder_path}")

        return self.history

    def _evaluate(self, dataloader, criterion):
        """Evaluate model on validation/test set"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for sequences, labels, lengths in dataloader:
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                lengths = lengths.to(self.device)

                outputs = self.model(sequences, lengths)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return running_loss / total, correct / total

    def predict(self, filterbank):
        """
        Predict speaker probabilities from filterbank
        Returns: numpy array of probabilities
        """
        self.model.eval()

        # Prepare input
        if len(filterbank.shape) == 2:
            filterbank = np.expand_dims(filterbank, axis=0)

        x = torch.tensor(filterbank, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probabilities = torch.softmax(logits, dim=1)

        return probabilities.cpu().numpy()

    def extract_embedding(self, filterbank):
        """Extract speaker embedding from filterbank"""
        self.model.eval()

        # Prepare input
        if len(filterbank.shape) == 2:
            filterbank = np.expand_dims(filterbank, axis=0)

        x = torch.tensor(filterbank, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            embedding = self.model(x, return_embedding=True)

        return embedding.cpu().numpy()[0]

    def identify_speaker(self, filterbank, speaker_embeddings, threshold=0.7):
        """
        Identify speaker by comparing embeddings using cosine similarity

        Args:
            filterbank: input audio features
            speaker_embeddings: dict of {speaker_id: reference_embedding}
            threshold: minimum similarity threshold

        Returns:
            (best_speaker, best_score)
        """
        test_embedding = self.extract_embedding(filterbank)

        similarities = {}
        for speaker_id, ref_embedding in speaker_embeddings.items():
            # Cosine similarity
            sim = np.dot(test_embedding, ref_embedding) / (
                np.linalg.norm(test_embedding) * np.linalg.norm(ref_embedding) + 1e-8
            )
            similarities[speaker_id] = sim

        best_speaker = max(similarities, key=similarities.get)
        best_score = similarities[best_speaker]

        return best_speaker, best_score

    def load_model(self, model_path=None, label_encoder_path=None):
        """Load a trained model and its label encoder"""
        if model_path is None:
            model_path = config.lstm_model_path
        if label_encoder_path is None:
            label_encoder_path = config.lstm_model_path.parent / "speaker_label_encoder.pkl"

        # Load model weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        print(f"[+] Loaded LSTM model from {model_path}")

        # Load label encoder if exists
        if label_encoder_path.exists():
            with open(label_encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            print(f"[+] Loaded speaker label encoder from {label_encoder_path}")
        else:
            print(f"[!] Warning: Label encoder not found at {label_encoder_path}")

    def plot_history(self):
        """Plot training history"""
        if not self.history['loss']:
            print("No training history available")
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss plot
        axes[0].plot(self.history['loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_title('LSTM Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)

        # Accuracy plot
        axes[1].plot(self.history['accuracy'], label='Train Accuracy')
        axes[1].plot(self.history['val_accuracy'], label='Val Accuracy')
        axes[1].set_title('LSTM Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig('lstm_training_history.png', dpi=100, bbox_inches='tight')
        print("[+] Saved: lstm_training_history.png")
        plt.close()


if __name__ == "__main__":
    from command_recognition.audio.extract_features import AudioFeatureExtractor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    extractor = AudioFeatureExtractor(sample_rate=16000)
    data = extractor.load_dataset(config.dataset_dir)
    features = extractor.extract_features(data['X_audio'])
    X_filterbank = features['filterbanks']
    y_speakers = data['y_speakers']

    # Convert speaker IDs to 0-indexed labels (CrossEntropyLoss requires [0, num_classes-1])
    label_encoder = LabelEncoder()
    y_speakers = label_encoder.fit_transform(y_speakers)
    print(f"   Speaker classes: {label_encoder.classes_} -> {list(range(len(label_encoder.classes_)))}")

    X_fb_train, X_fb_test, y_spk_fb_train, y_spk_fb_test = train_test_split(
        X_filterbank, y_speakers, test_size=0.2, random_state=42
    )

    X_fb_train, X_fb_val, y_spk_fb_train, y_spk_fb_val = train_test_split(
        X_fb_train, y_spk_fb_train, test_size=0.2, random_state=42
    )

    print(f"   LSTM Training: {len(X_fb_train)}, Validation: {len(X_fb_val)}, Test: {len(X_fb_test)}")

    lstm = SpeakerIdentificationLSTM(num_speakers=len(label_encoder.classes_))
    lstm.train(X_fb_train, y_spk_fb_train, X_fb_val, y_spk_fb_val,
               epochs=config.epochs_lstm, batch_size=config.batch_size_lstm,
               label_encoder=label_encoder)
    lstm.plot_history()

    # Evaluate LSTM
    test_loader = lstm._prepare_dataloader(X_fb_test, y_spk_fb_test, batch_size=16, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    lstm_loss, lstm_acc = lstm._evaluate(test_loader, criterion)
    print(f"\n[+] LSTM Test Accuracy: {lstm_acc:.2%}")
