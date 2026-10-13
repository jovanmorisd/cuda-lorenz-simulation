"""
Dataset Generation for Lorenz Surrogate Modeling
Creates sliding window datasets with normalization.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


class LorenzDataset:
    """
    Generates sliding window datasets from Lorenz trajectories.
    """
    
    def __init__(self, lookback=10):
        """
        Initialize dataset generator.
        
        Args:
            lookback: Number of past states to use as input (default: 10)
        """
        self.lookback = lookback
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def create_dataset(self, trajectories, lookback=None):
        """
        Create sliding window dataset from trajectories.
        
        Args:
            trajectories: List of trajectories, each of shape (num_steps, 3)
            lookback: Override default lookback window
            
        Returns:
            X: Input sequences of shape (num_samples, lookback, 3)
            y: Target outputs of shape (num_samples, 3)
        """
        if lookback is not None:
            self.lookback = lookback
        
        X = []
        y = []
        
        for trajectory in trajectories:
            # Create sliding windows
            for i in range(len(trajectory) - self.lookback):
                X.append(trajectory[i:i+self.lookback])
                y.append(trajectory[i+self.lookback])
        
        X = np.array(X)
        y = np.array(y)
        
        return X, y
    
    def normalize(self, X, y, fit=False):
        """
        Normalize input and output data.
        
        Args:
            X: Input sequences
            y: Target outputs
            fit: If True, fit the scaler on this data
            
        Returns:
            X_normalized: Normalized inputs
            y_normalized: Normalized outputs
        """
        # Reshape for fitting the scaler
        X_shape = X.shape
        X_flat = X.reshape(-1, X.shape[-1])
        
        if fit:
            # Fit on combined data
            all_data = np.vstack([X_flat, y])
            self.scaler.fit(all_data)
            self.is_fitted = True
        
        # Transform
        X_normalized = self.scaler.transform(X_flat).reshape(X_shape)
        y_normalized = self.scaler.transform(y)
        
        return X_normalized, y_normalized
    
    def inverse_normalize(self, data):
        """
        Inverse normalize data (denormalize).
        
        Args:
            data: Normalized data
            
        Returns:
            Denormalized data
        """
        if not self.is_fitted:
            return data
        return self.scaler.inverse_transform(data)
    
    def split_dataset(self, X, y, train_ratio=0.7, val_ratio=0.15):
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            X: Input sequences
            y: Target outputs
            train_ratio: Fraction for training (default: 0.7)
            val_ratio: Fraction for validation (default: 0.15)
            
        Returns:
            X_train, X_val, X_test: Input splits
            y_train, y_val, y_test: Output splits
        """
        num_samples = len(X)
        
        # Shuffle indices
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        # Split points
        train_size = int(num_samples * train_ratio)
        val_size = int(num_samples * val_ratio)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size+val_size]
        test_idx = indices[train_size+val_size:]
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    @staticmethod
    def create_residual_dataset(X, y_pred, y_true):
        """
        Create a residual dataset for learning prediction errors.
        
        Args:
            X: Input features
            y_pred: Predicted outputs
            y_true: Ground truth outputs
            
        Returns:
            residuals: Difference between true and predicted
        """
        return y_true - y_pred


def generate_training_data(trajectories, lookback=10, normalize=True):
    """
    Generate complete training dataset from trajectories.
    
    Args:
        trajectories: List of trajectory arrays
        lookback: Window size for sliding window
        normalize: Whether to normalize the data
        
    Returns:
        X: Input sequences
        y: Target outputs
        scaler: Fitted scaler (if normalize=True)
        dataset: LorenzDataset object
    """
    dataset = LorenzDataset(lookback=lookback)
    X, y = dataset.create_dataset(trajectories, lookback=lookback)
    
    scaler = None
    if normalize:
        X, y = dataset.normalize(X, y, fit=True)
        scaler = dataset.scaler
    
    return X, y, scaler, dataset
