from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from command_recognition.config import get_config

config = get_config()

class CommandRecognitionCNN:
    """CNN for voice command classification"""
    
    def __init__(self, num_commands=13, input_shape=(64, 99)):
        self.num_commands = num_commands
        self.input_shape = input_shape
        self.model = self._build_model()
        self.history = None
    
    def _build_model(self):
        """Build 5-layer CNN architecture"""
        model = keras.Sequential([
            # Input
            layers.Input(shape=self.input_shape + (1,)),
            
            # Block 1: Conv + BatchNorm + ReLU + MaxPool
            layers.Conv2D(48, kernel_size=(3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
            
            # Block 2
            layers.Conv2D(96, kernel_size=(3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
            
            # Block 3
            layers.Conv2D(192, kernel_size=(3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
            
            # Block 4
            layers.Conv2D(192, kernel_size=(3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            
            # Block 5
            layers.Conv2D(192, kernel_size=(3, 3), padding='same'),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'),
            
            # Fully connected
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_commands, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0003),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
        """Train CNN model"""
        print(f"\n🚀 Training CNN Model...")
        print(f"   Training samples: {X_train.shape[0]}")
        print(f"   Input shape: {X_train.shape[1:]}")
        
        # Add channel dimension
        X_train = np.expand_dims(X_train, axis=-1)
        X_val = np.expand_dims(X_val, axis=-1)
        
        # Convert to one-hot
        y_train_cat = keras.utils.to_categorical(y_train, self.num_commands)
        y_val_cat = keras.utils.to_categorical(y_val, self.num_commands)
        
        self.history = self.model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        if not config.cnn_model_path.parent.exists():
            config.cnn_model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(config.cnn_model_path)
        
        return self.history
    
    def predict(self, X):
        """Predict command from mel-spectrogram"""
        X = np.expand_dims(X, axis=-1)
        predictions = self.model.predict(X, verbose=0)
        return predictions
    
    def plot_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(self.history.history['loss'], label='Train Loss')
        axes[0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        axes[1].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[1].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('cnn_training_history.png', dpi=100, bbox_inches='tight')
        print("✓ Saved: cnn_training_history.png")
        plt.close()
        

if __name__ == "__main__":
    from command_recognition.audio.extract_features import AudioFeatureExtractor
    from sklearn.model_selection import train_test_split

    extractor = AudioFeatureExtractor(sample_rate=16000)
    data = extractor.load_dataset(config.dataset_dir)
    features = extractor.extract_features(data['X_audio'])
    X_mel = features['mel_specs']  # (N, 64, 99)

    y_commands = data['y_commands']
    y_speakers = data['y_speakers']
    
    # Split data
    X_mel_train, X_mel_test, y_cmd_train, y_cmd_test, y_spk_train, y_spk_test = train_test_split(
        X_mel, y_commands, y_speakers, test_size=0.2, random_state=42
    )
    
    X_mel_train, X_mel_val, y_cmd_train, y_cmd_val, y_spk_train_val, y_spk_val = train_test_split(
        X_mel_train, y_cmd_train, y_spk_train, test_size=0.2, random_state=42
    )


    print(f"\n📊 Data Split:")
    print(f"   CNN Training: {X_mel_train.shape[0]}, Validation: {X_mel_val.shape[0]}, Test: {X_mel_test.shape[0]}")
    
    cnn = CommandRecognitionCNN(num_commands=config.num_commands)
    cnn.train(X_mel_train, y_cmd_train, X_mel_val, y_cmd_val, 
              epochs=config.epochs_cnn, batch_size=config.batch_size_cnn)
    cnn.plot_history()
    
    # Evaluate CNN
    cnn_preds = cnn.predict(X_mel_test)
    cnn_acc = np.mean(np.argmax(cnn_preds, axis=1) == y_cmd_test)
    print(f"\n✓ CNN Test Accuracy: {cnn_acc:.2%}")