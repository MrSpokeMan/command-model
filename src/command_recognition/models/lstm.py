import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from command_recognition.config import get_config

config = get_config()

class SpeakerIdentificationLSTM:
    """LSTM-based speaker identification with x-vector style embeddings"""
    
    def __init__(self, num_speakers=2, embedding_dim=256, feature_dim=24):
        self.num_speakers = num_speakers
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        self.model = self._build_model()
        self.embedding_model = self._build_embedding_model()
        self.scaler = StandardScaler()
        self.history = None
    
    def _build_model(self):
        """Build LSTM model for speaker classification"""
        inputs = layers.Input(shape=(None, self.feature_dim))
        
        # LSTM frame-level layers
        x = layers.LSTM(512, return_sequences=True, activation='relu')(inputs)
        x = layers.LSTM(512, return_sequences=True, activation='relu')(x)
        
        # Statistics pooling (mean + std)
        mean = layers.GlobalAveragePooling1D()(x)
        std = layers.Lambda(lambda x: tf.math.reduce_std(x, axis=1))(x)
        pooled = layers.Concatenate()([mean, std])  # (batch, 1024)
        
        # Segment-level layers
        x = layers.Dense(512, activation='relu')(pooled)
        x = layers.BatchNormalization()(x)
        
        # Embedding (speaker representation)
        embedding = layers.Dense(self.embedding_dim, activation='relu')(x)
        
        # Speaker classification
        speaker_out = layers.Dense(self.num_speakers, activation='softmax')(embedding)
        
        model = keras.Model(inputs=inputs, outputs=speaker_out)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _build_embedding_model(self):
        """Build model to extract speaker embeddings"""
        inputs = layers.Input(shape=(None, self.feature_dim))
        
        x = layers.LSTM(512, return_sequences=True, activation='relu')(inputs)
        x = layers.LSTM(512, return_sequences=True, activation='relu')(x)
        
        mean = layers.GlobalAveragePooling1D()(x)
        std = layers.Lambda(lambda x: tf.math.reduce_std(x, axis=1))(x)
        pooled = layers.Concatenate()([mean, std])
        
        x = layers.Dense(512, activation='relu')(pooled)
        x = layers.BatchNormalization()(x)
        embedding = layers.Dense(self.embedding_dim, activation='relu')(x)
        
        model = keras.Model(inputs=inputs, outputs=embedding)
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=16):
        """Train LSTM model"""
        print(f"\n🎙️  Training LSTM Model (Speaker Identification)...")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Unique speakers: {len(np.unique(y_train))}")
        
        # Pad sequences
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        X_train_padded = pad_sequences(X_train, dtype='float32', padding='post')
        X_val_padded = pad_sequences(X_val, dtype='float32', padding='post')
        
        # Convert to one-hot
        y_train_cat = keras.utils.to_categorical(y_train, self.num_speakers)
        y_val_cat = keras.utils.to_categorical(y_val, self.num_speakers)
        
        self.history = self.model.fit(
            X_train_padded, y_train_cat,
            validation_data=(X_val_padded, y_val_cat),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        if not config.lstm_model_path.parent.exists():
            config.lstm_model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(config.lstm_model_path)
        
        return self.history
    
    def extract_embedding(self, filterbank):
        """Extract speaker embedding from filterbank"""
        if len(filterbank.shape) == 2:
            filterbank = np.expand_dims(filterbank, axis=0)
        
        embedding = self.embedding_model.predict(filterbank, verbose=0)
        return embedding[0]
    
    def identify_speaker(self, filterbank, speaker_embeddings, threshold=0.7):
        """
        Identify speaker by comparing embeddings
        Uses cosine similarity
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
    
    def plot_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(self.history.history['loss'], label='Train Loss')
        axes[0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0].set_title('LSTM Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[1].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[1].set_title('LSTM Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('lstm_training_history.png', dpi=100, bbox_inches='tight')
        print("✓ Saved: lstm_training_history.png")
        plt.close()
        
        
if __name__ == "__main__":
    from command_recognition.audio.extract_features import AudioFeatureExtractor
    from sklearn.model_selection import train_test_split
    
    extractor = AudioFeatureExtractor(sample_rate=16000)
    data = extractor.load_dataset(config.dataset_dir)
    features = extractor.extract_features(data['X_audio'])
    X_filterbank = features['filterbanks']
    y_speakers = data['y_speakers']
    
    X_fb_train, X_fb_test, y_spk_fb_train, y_spk_fb_test = train_test_split(
        X_filterbank, y_speakers, test_size=0.2, random_state=42
    )
    
    X_fb_train, X_fb_val, y_spk_fb_train, y_spk_fb_val = train_test_split(
        X_fb_train, y_spk_fb_train, test_size=0.2, random_state=42
    )
    
    print(f"   LSTM Training: {len(X_fb_train)}, Validation: {len(X_fb_val)}, Test: {len(X_fb_test)}")

    lstm = SpeakerIdentificationLSTM(num_speakers=config.num_speakers)
    lstm.train(X_fb_train, y_spk_fb_train, X_fb_val, y_spk_fb_val,
               epochs=config.epochs_lstm, batch_size=config.batch_size_lstm)
    lstm.plot_history()
    
    # Evaluate LSTM
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    X_fb_test_padded = pad_sequences(X_fb_test, dtype='float32', padding='post')
    y_spk_test_cat = keras.utils.to_categorical(y_spk_fb_test, config.num_speakers)
    lstm_loss, lstm_acc = lstm.model.evaluate(X_fb_test_padded, y_spk_test_cat, verbose=0)
    print(f"\n✓ LSTM Test Accuracy: {lstm_acc:.2%}")