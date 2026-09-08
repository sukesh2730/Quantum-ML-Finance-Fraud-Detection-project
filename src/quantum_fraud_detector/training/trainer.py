"""Training module for optimizing VQC parameters.

This module provides functions for training the Variational Quantum Classifier
using gradient descent and computing validation metrics.
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp  # PennyLane's numpy for autograd compatibility
from typing import Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: Ground truth labels, shape (n_samples,), values in {0, 1}
        y_pred: Predicted fraud scores, shape (n_samples,), values in [0, 1]
        threshold: Classification threshold (default 0.5)
        
    Returns:
        Dictionary with accuracy, precision, recall, F1-score
        
    Example:
        >>> y_true = np.array([0, 1, 1, 0])
        >>> y_pred = np.array([0.2, 0.8, 0.6, 0.3])
        >>> metrics = compute_metrics(y_true, y_pred)
        >>> metrics['accuracy']
        1.0
    """
    # Apply threshold to convert fraud scores to binary predictions
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Calculate binary classification metrics using scikit-learn
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    
    # Return dictionary with all metrics
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model,  # VariationalQuantumClassifier instance
    epochs: int = 50,
    learning_rate: float = 0.01,
    optimizer: str = "adam"
) -> Dict[str, list]:
    """
    Train VQC using gradient descent.
    
    This function trains the Variational Quantum Classifier by optimizing its
    parameters using gradient descent on the training data. For each epoch, it
    computes predictions, calculates binary cross-entropy loss, computes gradients
    using PennyLane's automatic differentiation, and updates parameters using the
    specified optimizer. Training and validation metrics are tracked throughout.
    
    Args:
        X_train: Training features, shape (n_train, n_qubits).
            Values should be preprocessed and scaled to [0, π] range.
        y_train: Training labels, shape (n_train,), values in {0, 1}.
            Binary labels where 1 indicates fraud and 0 indicates legitimate.
        X_val: Validation features, shape (n_val, n_qubits).
        y_val: Validation labels, shape (n_val,), values in {0, 1}.
        model: VariationalQuantumClassifier instance to train.
            Must have initialized parameters (model.params should not be None).
        epochs: Number of training epochs (default 50).
        learning_rate: Step size for optimizer (default 0.01).
        optimizer: Optimizer type, one of "adam", "sgd", "nesterov" (default "adam").
    
    Returns:
        Dictionary containing training history with lists populated during training:
        {
            "train_loss": [loss per epoch],
            "val_loss": [validation loss per epoch],
            "val_accuracy": [validation accuracy per epoch],
            "val_precision": [validation precision per epoch],
            "val_recall": [validation recall per epoch],
            "val_f1": [validation F1-score per epoch]
        }
    
    Raises:
        ValueError: If model parameters are not initialized.
        ValueError: If optimizer is not one of the supported types.
        ValueError: If input dimensions don't match model configuration.
    
    Examples:
        >>> from quantum_fraud_detector.quantum_model.vqc import VariationalQuantumClassifier
        >>> model = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        >>> model.initialize_params(seed=42)
        >>> X_train = np.random.uniform(0, np.pi, (100, 4))
        >>> y_train = np.random.randint(0, 2, 100)
        >>> X_val = np.random.uniform(0, np.pi, (20, 4))
        >>> y_val = np.random.randint(0, 2, 20)
        >>> history = train_model(X_train, y_train, X_val, y_val, model, epochs=10)
        >>> assert len(history['train_loss']) == 10
        >>> assert len(history['val_accuracy']) == 10
    
    Notes:
        - Uses binary cross-entropy loss with epsilon=1e-7 to prevent log(0)
        - PennyLane's automatic differentiation computes gradients
        - Model parameters are updated in-place via optimizer.step()
        - Progress is printed every 10 epochs
    
    References:
        Requirements 3.2, 3.3: Optimize Model_Parameters using gradient descent
        and track validation metrics during training.
    """
    # Validate model has initialized parameters
    if model.params is None:
        raise ValueError(
            "Model parameters must be initialized before training. "
            "Call model.initialize_params() first."
        )
    
    # Validate input dimensions
    if X_train.shape[1] != model.n_qubits:
        raise ValueError(
            f"Training features have {X_train.shape[1]} dimensions, "
            f"but model expects {model.n_qubits} qubits."
        )
    
    if X_val.shape[1] != model.n_qubits:
        raise ValueError(
            f"Validation features have {X_val.shape[1]} dimensions, "
            f"but model expects {model.n_qubits} qubits."
        )
    
    # Initialize PennyLane optimizer based on optimizer parameter
    if optimizer.lower() == "adam":
        opt = qml.AdamOptimizer(stepsize=learning_rate)
    elif optimizer.lower() == "sgd":
        opt = qml.GradientDescentOptimizer(stepsize=learning_rate)
    elif optimizer.lower() == "nesterov":
        opt = qml.NesterovMomentumOptimizer(stepsize=learning_rate)
    else:
        raise ValueError(
            f"Unsupported optimizer: {optimizer}. "
            f"Choose from 'adam', 'sgd', or 'nesterov'."
        )
    
    # Initialize training history dictionary to track metrics over epochs
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": []
    }
    
    # Define cost function for gradient computation
    def cost_function(params):
        """Compute binary cross-entropy loss over all training samples.
        
        Args:
            params: Current model parameters
            
        Returns:
            float: Mean binary cross-entropy loss
        """
        # Update model parameters
        model.params = params
        
        # Compute loss by iterating over training samples
        # This avoids autograd issues with predict_batch
        epsilon = 1e-7
        total_loss = 0.0
        
        for i in range(len(X_train)):
            # Get prediction for single sample
            features = X_train[i]
            label = y_train[i]
            
            # Execute circuit directly
            pred = model.predict(features)
            
            # Compute binary cross-entropy for this sample
            sample_loss = -(label * pnp.log(pred + epsilon) + 
                           (1 - label) * pnp.log(1 - pred + epsilon))
            total_loss += sample_loss
        
        # Return mean loss
        return total_loss / len(X_train)
    
    # Training loop - iterate for specified number of epochs
    for epoch in range(epochs):
        # Compute gradients and update parameters using optimizer
        model.params = opt.step(cost_function, model.params)
        
        # Compute training loss for this epoch
        train_predictions = model.predict_batch(X_train)
        epsilon = 1e-7
        train_loss = -np.mean(
            y_train * np.log(train_predictions + epsilon) + 
            (1 - y_train) * np.log(1 - train_predictions + epsilon)
        )
        
        # Compute validation loss and metrics
        val_predictions = model.predict_batch(X_val)
        val_loss = -np.mean(
            y_val * np.log(val_predictions + epsilon) + 
            (1 - y_val) * np.log(1 - val_predictions + epsilon)
        )
        
        # Compute validation metrics using compute_metrics function
        val_metrics = compute_metrics(y_val, val_predictions)
        
        # Store training history
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_f1"].append(val_metrics["f1"])
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Accuracy: {val_metrics['accuracy']:.4f}, "
                  f"Val F1: {val_metrics['f1']:.4f}")
    
    return history
