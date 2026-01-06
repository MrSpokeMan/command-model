import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
from datetime import datetime

# ============================================================================
# 2. AUDIO FEATURE EXTRACTION
# ============================================================================



# ============================================================================
# 3. CNN MODEL FOR COMMAND RECOGNITION
# ============================================================================


# ============================================================================
# 4. LSTM MODEL FOR SPEAKER IDENTIFICATION
# ============================================================================

# ============================================================================
# 5. COMPLETE PIPELINE
# ============================================================================

def main():
    """Complete pipeline: generate data -> train both models"""
    
    print("="*70)
    print("VOICE COMMAND RECOGNITION WITH SPEAKER IDENTIFICATION")
    print("="*70)
    
    # Configuration
    NUM_SPEAKERS = 2
    NUM_COMMANDS = 13
    SAMPLES_PER_CLASS = 5
    EPOCHS_CNN = 20
    EPOCHS_LSTM = 20
    BATCH_SIZE_CNN = 32
    BATCH_SIZE_LSTM = 16
    
    COMMANDS = [
        "graj", "stop", "głośniej", "ciszej", "do_przodu",
        "do_tyłu", "lewo", "prawo", "włącz", "wyłącz",
        "pauza", "wznów", "reset"
    ]
    
    # Step 1: Generate synthetic dataset
    print("\n[1/4] GENERATING SYNTHETIC DATASET")
    print("-" * 70)
    generator = SyntheticAudioGenerator(sample_rate=16000)
    data = generator.create_dataset(
        num_speakers=NUM_SPEAKERS,
        num_commands=NUM_COMMANDS,
        samples_per_class=SAMPLES_PER_CLASS
    )
    
    # Step 2: Extract features
    print("\n[2/4] EXTRACTING AUDIO FEATURES")
    print("-" * 70)
    extractor = AudioFeatureExtractor(sample_rate=16000)
    features = extractor.extract_features(data['X_audio'])
    
    X_mel = features['mel_specs']  # (N, 64, 99)
    X_filterbank = features['filterbanks']  # List of (T, 24) arrays
    y_commands = data['y_commands']
    y_speakers = data['y_speakers']
    
    # Split data
    X_mel_train, X_mel_test, y_cmd_train, y_cmd_test, y_spk_train, y_spk_test = train_test_split(
        X_mel, y_commands, y_speakers, test_size=0.2, random_state=42
    )
    
    X_mel_train, X_mel_val, y_cmd_train, y_cmd_val, y_spk_train_val, y_spk_val = train_test_split(
        X_mel_train, y_cmd_train, y_spk_train, test_size=0.2, random_state=42
    )
    
    # For LSTM (filterbanks)
    X_fb_train, X_fb_test, y_spk_fb_train, y_spk_fb_test = train_test_split(
        X_filterbank, y_speakers, test_size=0.2, random_state=42
    )
    
    X_fb_train, X_fb_val, y_spk_fb_train, y_spk_fb_val = train_test_split(
        X_fb_train, y_spk_fb_train, test_size=0.2, random_state=42
    )
    
    print(f"\n📊 Data Split:")
    print(f"   CNN Training: {X_mel_train.shape[0]}, Validation: {X_mel_val.shape[0]}, Test: {X_mel_test.shape[0]}")
    print(f"   LSTM Training: {len(X_fb_train)}, Validation: {len(X_fb_val)}, Test: {len(X_fb_test)}")
    
    # Step 3: Train CNN
    print("\n[3/4] TRAINING CNN MODEL FOR COMMAND RECOGNITION")
    print("-" * 70)
    cnn = CommandRecognitionCNN(num_commands=NUM_COMMANDS)
    cnn.train(X_mel_train, y_cmd_train, X_mel_val, y_cmd_val, 
              epochs=EPOCHS_CNN, batch_size=BATCH_SIZE_CNN)
    cnn.plot_history()
    
    # Evaluate CNN
    cnn_preds = cnn.predict(X_mel_test)
    cnn_acc = np.mean(np.argmax(cnn_preds, axis=1) == y_cmd_test)
    print(f"\n✓ CNN Test Accuracy: {cnn_acc:.2%}")
    
    # Step 4: Train LSTM
    print("\n[4/4] TRAINING LSTM MODEL FOR SPEAKER IDENTIFICATION")
    print("-" * 70)
    lstm = SpeakerIdentificationLSTM(num_speakers=NUM_SPEAKERS)
    lstm.train(X_fb_train, y_spk_fb_train, X_fb_val, y_spk_fb_val,
               epochs=EPOCHS_LSTM, batch_size=BATCH_SIZE_LSTM)
    lstm.plot_history()
    
    # Evaluate LSTM
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    X_fb_test_padded = pad_sequences(X_fb_test, dtype='float32', padding='post')
    y_spk_test_cat = keras.utils.to_categorical(y_spk_fb_test, NUM_SPEAKERS)
    lstm_loss, lstm_acc = lstm.model.evaluate(X_fb_test_padded, y_spk_test_cat, verbose=0)
    print(f"\n✓ LSTM Test Accuracy: {lstm_acc:.2%}")
    
    # Step 5: Demo - Combined inference
    print("\n" + "="*70)
    print("DEMO: REAL-TIME INFERENCE")
    print("="*70)
    
    # Enroll speakers (create reference embeddings)
    print("\n🎙️  Enrolling speakers...")
    speaker_embeddings = {}
    for speaker_id in range(NUM_SPEAKERS):
        # Get samples from this speaker
        speaker_samples = [X_fb_train[i] for i in range(len(X_fb_train)) if y_spk_fb_train[i] == speaker_id]
        
        # Extract embeddings
        embeddings = []
        for fb in speaker_samples[:3]:  # Use first 3 samples
            emb = lstm.extract_embedding(fb)
            embeddings.append(emb)
        
        # Average embedding
        speaker_embeddings[f"Speaker_{speaker_id}"] = np.mean(embeddings, axis=0)
        print(f"   ✓ Speaker_{speaker_id} enrolled")
    
    # Test on random samples
    print(f"\n📝 Testing on {5} random samples:")
    print("-" * 70)
    
    for test_idx in np.random.choice(len(X_mel_test), 5, replace=False):
        mel_spec = X_mel_test[test_idx]
        filterbank = X_fb_test[test_idx]
        true_cmd = y_cmd_test[test_idx]
        true_spk = y_spk_fb_test[test_idx]
        
        # Predict command
        cmd_probs = cnn.predict(mel_spec)[0]
        pred_cmd = np.argmax(cmd_probs)
        cmd_conf = cmd_probs[pred_cmd]
        
        # Identify speaker
        pred_spk, spk_conf = lstm.identify_speaker(filterbank, speaker_embeddings)
        
        print(f"\n🎯 Sample {test_idx + 1}:")
        print(f"   Command: {COMMANDS[true_cmd]} → Predicted: {COMMANDS[pred_cmd]} ({cmd_conf:.1%}) {'✓' if pred_cmd == true_cmd else '✗'}")
        print(f"   Speaker: Speaker_{true_spk} → Predicted: {pred_spk} ({spk_conf:.1%})")
    
    # Save models
    print("\n" + "="*70)
    print("SAVING MODELS")
    print("="*70)
    cnn.model.save('cnn_command_model.h5')
    lstm.model.save('lstm_speaker_model.h5')
    lstm.embedding_model.save('lstm_embedding_model.h5')
    
    # Save configuration
    config = {
        'num_speakers': NUM_SPEAKERS,
        'num_commands': NUM_COMMANDS,
        'commands': COMMANDS,
        'sample_rate': 16000,
        'cnn_input_shape': (64, 99),
        'lstm_feature_dim': 24,
        'lstm_embedding_dim': 256
    }
    
    import json
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✓ Models saved:")
    print("   - cnn_command_model.h5")
    print("   - lstm_speaker_model.h5")
    print("   - lstm_embedding_model.h5")
    print("   - config.json")
    print("   - cnn_training_history.png")
    print("   - lstm_training_history.png")
    
    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETE!")
    print("="*70)
    
    return {
        'cnn': cnn,
        'lstm': lstm,
        'speaker_embeddings': speaker_embeddings,
        'extractor': extractor,
        'commands': COMMANDS,
        'config': config
    }

if __name__ == "__main__":
    results = main()
