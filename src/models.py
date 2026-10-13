"""
AI Models for Lorenz Surrogate Modeling
Implements MLP, LSTM, and baseline persistence model.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm


class PersistenceBaseline:
    """
    Simple persistence baseline: assume next state equals current state.
    """
    
    def predict(self, X):
        """
        Predict next state as last input state.
        
        Args:
            X: Input sequences of shape (num_samples, lookback, 3)
            
        Returns:
            predictions: Predicted next states (num_samples, 3)
        """
        # Take the last state of each sequence
        return X[:, -1, :]
    
    def forward(self, X):
        """Alias for predict."""
        return self.predict(X)


class MLPSurrogate(nn.Module):
    """
    Multi-layer perceptron surrogate model for Lorenz prediction.
    """
    
    def __init__(self, lookback=10, hidden_dims=[128, 64, 32], dropout=0.2):
        """
        Initialize MLP model.
        
        Args:
            lookback: Window size (default: 10)
            hidden_dims: Hidden layer dimensions (default: [128, 64, 32])
            dropout: Dropout rate (default: 0.2)
        """
        super(MLPSurrogate, self).__init__()
        
        self.lookback = lookback
        self.input_size = lookback * 3  # 3D state
        
        layers = []
        prev_dim = self.input_size
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 3))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, X):
        """
        Forward pass.
        
        Args:
            X: Input of shape (batch_size, lookback, 3)
            
        Returns:
            predictions: Output of shape (batch_size, 3)
        """
        # Flatten input
        batch_size = X.shape[0]
        X_flat = X.view(batch_size, -1)
        return self.network(X_flat)


class LSTMSurrogate(nn.Module):
    """
    LSTM surrogate model for Lorenz prediction.
    """
    
    def __init__(self, lookback=10, lstm_hidden=64, num_layers=2, dropout=0.2):
        """
        Initialize LSTM model.
        
        Args:
            lookback: Window size (default: 10)
            lstm_hidden: LSTM hidden dimension (default: 64)
            num_layers: Number of LSTM layers (default: 2)
            dropout: Dropout rate (default: 0.2)
        """
        super(LSTMSurrogate, self).__init__()
        
        self.lookback = lookback
        self.lstm_hidden = lstm_hidden
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=3,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3)
        )
    
    def forward(self, X):
        """
        Forward pass.
        
        Args:
            X: Input of shape (batch_size, lookback, 3)
            
        Returns:
            predictions: Output of shape (batch_size, 3)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(X)
        
        # Take last hidden state
        last_hidden = h_n[-1]  # Shape: (batch_size, lstm_hidden)
        
        # Prediction
        predictions = self.head(last_hidden)
        return predictions


class residualMLPSurrogate(nn.Module):
    """
    MLP that predicts residuals (error corrections).
    """
    
    def __init__(self, lookback=10, hidden_dims=[64, 32], dropout=0.2):
        """
        Initialize residual MLP model.
        
        Args:
            lookback: Window size (default: 10)
            hidden_dims: Hidden layer dimensions (default: [64, 32])
            dropout: Dropout rate (default: 0.2)
        """
        super(residualMLPSurrogate, self).__init__()
        
        self.lookback = lookback
        self.input_size = lookback * 3 + 3  # states + last step prediction
        
        layers = []
        prev_dim = self.input_size
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 3))
        self.network = nn.Sequential(*layers)
    
    def forward(self, X, base_prediction):
        """
        Forward pass.
        
        Args:
            X: Input sequences of shape (batch_size, lookback, 3)
            base_prediction: Baseline predictions of shape (batch_size, 3)
            
        Returns:
            corrections: Residual corrections of shape (batch_size, 3)
        """
        batch_size = X.shape[0]
        X_flat = X.view(batch_size, -1)
        
        # Concatenate with base prediction
        X_with_base = torch.cat([X_flat, base_prediction], dim=1)
        
        return self.network(X_with_base)


class SurrogateTrainer:
    """
    Trainer for surrogate models.
    """
    
    def __init__(self, model, learning_rate=0.001, device='cpu'):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            learning_rate: Learning rate (default: 0.001)
            device: Device to use ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.history = {'train_loss': [], 'val_loss': []}
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            predictions = self.model(X_batch)
            loss = self.criterion(predictions, y_batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        self.history['train_loss'].append(avg_loss)
        return avg_loss
    
    def evaluate(self, val_loader):
        """
        Evaluate on validation set.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                predictions = self.model(X_batch)
                loss = self.criterion(predictions, y_batch)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        self.history['val_loss'].append(avg_loss)
        return avg_loss
    
    def fit(self, train_loader, val_loader, epochs=200, verbose=True):
        """
        Train model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs (default: 200)
            verbose: Print progress (default: True)
            
        Returns:
            history: Training history
        """
        epoch_iterator = tqdm(range(epochs), desc="Training", unit="epoch") if verbose else range(epochs)
        
        for epoch in epoch_iterator:
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)
            
            if verbose:
                epoch_iterator.set_description(
                    f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}"
                )
        
        return self.history
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Input data
            
        Returns:
            Predictions
        """
        self.model.eval()
        with torch.no_grad():
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X, dtype=torch.float32)
            X = X.to(self.device)
            predictions = self.model(X).cpu().numpy()
        return predictions
