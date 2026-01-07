from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from command_recognition.config import get_config

config = get_config()

class CommandRecognitionCNN:
    def __init__(self, num_commands, input_shape=(64, 64)):
        self.num_commands = num_commands
        self.input_shape = input_shape
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the PyTorch Model
        self.model = MultiViewCNN(num_commands=num_commands).to(self.device)
        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        self.label_encoder = LabelEncoder()

    def _prepare_tensors(self, X_data, y_data=None):
        """Helper to convert numpy arrays to PyTorch tensors with correct shape (N, 1, H, W)"""
        # X_data is expected to be a tuple/list: (X_spec, X_smooth, X_mel)
        X_spec, X_smooth, X_mel = X_data
        
        # Add channel dimension: (N, 64, 64) -> (N, 1, 64, 64)
        t_spec = torch.tensor(X_spec, dtype=torch.float32).unsqueeze(1)
        t_smooth = torch.tensor(X_smooth, dtype=torch.float32).unsqueeze(1)
        t_mel = torch.tensor(X_mel, dtype=torch.float32).unsqueeze(1)
        
        if y_data is not None:
            # PyTorch CrossEntropyLoss expects class indices (Long), not One-Hot
            t_y = torch.tensor(y_data, dtype=torch.long)
            return t_spec, t_smooth, t_mel, t_y
        
        return t_spec, t_smooth, t_mel

    def train(self, X_train, y_train, X_val, y_val, epochs=25, batch_size=32):
        """
        Train the Multi-View CNN
        X_train/X_val must be tuples: (spectrograms, smooth_spectrograms, mel_spectrograms)
        """
        print(f"\n[*] Training Multi-View CNN Model on {self.device}...")
        
        # 1. Encode string labels to integers
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)
        
        # 2. Prepare DataLoaders
        train_inputs = self._prepare_tensors(X_train, y_train_encoded)
        val_inputs = self._prepare_tensors(X_val, y_val_encoded)
        
        train_dataset = TensorDataset(*train_inputs)
        val_dataset = TensorDataset(*val_inputs)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 3. Setup Optimizer & Loss
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.0003)
        
        # 4. Training Loop
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for spec, smooth, mel, labels in train_loader:
                spec, smooth, mel, labels = spec.to(self.device), smooth.to(self.device), mel.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(spec, smooth, mel)
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
            
            # 5. Validation Step
            val_loss, val_acc = self._evaluate(val_loader, criterion)
            
            # Update History
            self.history['loss'].append(epoch_loss)
            self.history['accuracy'].append(epoch_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_acc)
            
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f} - "
                  f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")

        # 6. Save Model
        if not config.cnn_model_path.parent.exists():
            config.cnn_model_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(self.model.state_dict(), config.cnn_model_path)
        print(f"[+] Saved model to {config.cnn_model_path}")
        
        # Save label encoder
        label_encoder_path = config.cnn_model_path.parent / "label_encoder.pkl"
        with open(label_encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"[+] Saved label encoder to {label_encoder_path}")
        
        return self.history

    def _evaluate(self, dataloader, criterion):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for spec, smooth, mel, labels in dataloader:
                spec, smooth, mel, labels = spec.to(self.device), smooth.to(self.device), mel.to(self.device), labels.to(self.device)
                
                outputs = self.model(spec, smooth, mel)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        return running_loss / total, correct / total

    def predict(self, X):
        """Predict command. X should be tuple (spec, smooth, mel)"""
        self.model.eval()
        
        # Prepare inputs (no labels)
        inputs = self._prepare_tensors(X)
        spec, smooth, mel = [t.to(self.device) for t in inputs]
        
        with torch.no_grad():
            logits = self.model(spec, smooth, mel)
            probabilities = torch.softmax(logits, dim=1)
            
        return probabilities.cpu().numpy()
    
    def predict_class(self, X):
        """Predict command class name. X should be tuple (spec, smooth, mel)"""
        probabilities = self.predict(X)
        predicted_indices = np.argmax(probabilities, axis=1)
        predicted_classes = self.label_encoder.inverse_transform(predicted_indices)
        return predicted_classes
    
    def get_class_names(self):
        """Get all class names in order"""
        if hasattr(self.label_encoder, 'classes_'):
            return self.label_encoder.classes_
        return None
    
    def load_model(self, model_path=None, label_encoder_path=None):
        """Load a trained model and its label encoder"""
        if model_path is None:
            model_path = config.cnn_model_path
        if label_encoder_path is None:
            label_encoder_path = config.cnn_model_path.parent / "label_encoder.pkl"
        
        # Load model weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        print(f"[+] Loaded model from {model_path}")
        
        # Load label encoder
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        print(f"[+] Loaded label encoder from {label_encoder_path}")

    def plot_history(self):
        """Plot training history"""
        if not self.history['loss']:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss Plot
        axes[0].plot(self.history['loss'], label='Train Loss')
        axes[0].plot(self.history['val_loss'], label='Val Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy Plot
        axes[1].plot(self.history['accuracy'], label='Train Accuracy')
        axes[1].plot(self.history['val_accuracy'], label='Val Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('cnn_training_history.png', dpi=100, bbox_inches='tight')
        print("[+] Saved: cnn_training_history.png")
        plt.close()


class CNNBranch(nn.Module):
    def __init__(self):
        super(CNNBranch, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(1, 48, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(48)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Block 2
        self.conv2 = nn.Conv2d(48, 96, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(96)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Block 3
        self.conv3 = nn.Conv2d(96, 192, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(192)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Block 4 (Note: Your Keras code had no pooling here)
        self.conv4 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(192)
        
        # Block 5
        self.conv5 = nn.Conv2d(192, 192, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(192)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # x shape: (Batch, 1, 64, 64)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x))) # No pool
        x = self.pool5(F.relu(self.bn5(self.conv5(x))))
        
        # Flatten: (Batch, Channels * H * W)
        x = torch.flatten(x, 1)
        return x
    
class MultiViewCNN(nn.Module):
    def __init__(self, num_commands):
        super(MultiViewCNN, self).__init__()
        self.num_commands = num_commands
        self.branch_spec = CNNBranch()
        self.branch_smooth = CNNBranch()
        self.branch_mel = CNNBranch()
        
        flattented_size = 192 * 4 * 4 * 3
        
        self.fc1 = nn.Linear(flattented_size, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_commands)
        
    def forward(self, x_spec, x_smooth, x_mel):
        x_spec = self.branch_spec(x_spec)
        x_smooth = self.branch_smooth(x_smooth)
        x_mel = self.branch_mel(x_mel)
        
        combined = torch.cat((x_spec, x_smooth, x_mel), dim=1)
        
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        output = self.fc2(x)
        return output

if __name__ == "__main__":
    from command_recognition.audio.extract_features import AudioFeatureExtractor
    from sklearn.model_selection import train_test_split

    extractor = AudioFeatureExtractor(sample_rate=16000)
    data = extractor.load_dataset(config.dataset_dir)
    features = extractor.extract_features(data['X_audio'])
    X_mel = features['mel_specs']  # (N, 64, 99)
    X_smooth = features['smooth_specs']  # (N, 64, 99)
    X_spec = features['specs']  # (N, 64, 99)
    y = data['y_commands']
    
    num_classes = len(np.unique(y))
    cnn = CommandRecognitionCNN(num_commands=num_classes)
    
    X_spec_train, X_spec_val, \
    X_smooth_train, X_smooth_val, \
    X_mel_train, X_mel_val, \
    y_train, y_val = train_test_split(
    X_spec, X_smooth, X_mel, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y # Keeps class balance even
    )
    
    X_train_tuple = (X_spec_train, X_smooth_train, X_mel_train)
    X_val_tuple = (X_spec_val, X_smooth_val, X_mel_val)
    
    history = cnn.train(X_train_tuple, y_train, X_val_tuple, y_val, epochs=25, batch_size=32)
    cnn.plot_history()