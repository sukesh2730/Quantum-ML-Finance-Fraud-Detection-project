"""Model serialization module for saving and loading trained models.

This module provides functions to persist trained VQC models and preprocessors
to disk and restore them for inference.
"""

import json
import os
from typing import Any, Dict, Optional, Tuple
import numpy as np
import joblib


def save_model(
    model,  # VariationalQuantumClassifier instance
    preprocessor,  # TransactionPreprocessor instance
    save_dir: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save complete model state to disk.
    
    Creates a directory structure containing all model artifacts:
    - model_params.npy: VQC parameter array
    - model_config.json: VQC architecture configuration
    - preprocessor.pkl: Fitted TransactionPreprocessor
    - metadata.json: Optional training metadata
    
    Args:
        model: Trained VariationalQuantumClassifier instance
        preprocessor: Fitted TransactionPreprocessor instance
        save_dir: Directory path to save artifacts
        metadata: Optional dictionary with training info (accuracy, timestamp, etc.)
        
    Raises:
        ValueError: If model parameters are not initialized
        ValueError: If preprocessor is not fitted
        
    Example:
        >>> from quantum_fraud_detector.quantum_model.vqc import VariationalQuantumClassifier
        >>> from quantum_fraud_detector.preprocessing.preprocessor import TransactionPreprocessor
        >>> model = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        >>> model.initialize_params()
        >>> preprocessor = TransactionPreprocessor([], [])
        >>> save_model(model, preprocessor, "./saved_model")
        
    References:
        Requirements 6.1, 6.2: Serialize Model_Parameters and preprocessing state
    """
    # Validate model has parameters
    if model.params is None:
        raise ValueError(
            "Model parameters must be initialized before saving. "
            "Train the model first or call model.initialize_params()."
        )
    
    # Validate preprocessor is fitted
    if not preprocessor.is_fitted:
        raise ValueError(
            "Preprocessor must be fitted before saving. "
            "Call preprocessor.fit() on training data first."
        )
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save VQC parameters as numpy array
    params_path = os.path.join(save_dir, "model_params.npy")
    np.save(params_path, model.params)
    
    # Save VQC configuration as JSON
    config = {
        "n_qubits": model.n_qubits,
        "n_layers": model.n_layers,
        "device_name": model.device_name
    }
    config_path = os.path.join(save_dir, "model_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save fitted preprocessor using joblib
    preprocessor_path = os.path.join(save_dir, "preprocessor.pkl")
    joblib.dump(preprocessor, preprocessor_path)
    
    # Save optional metadata
    if metadata is not None:
        metadata_path = os.path.join(save_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    print(f"Model saved successfully to {save_dir}")


def load_model(load_dir: str) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Restore model from disk.
    
    Loads all model artifacts from the specified directory and reconstructs
    the VQC and preprocessor in their trained state.
    
    Args:
        load_dir: Directory containing saved model artifacts
        
    Returns:
        Tuple of (model, preprocessor, metadata):
        - model: Reconstructed VariationalQuantumClassifier with loaded parameters
        - preprocessor: Loaded TransactionPreprocessor in fitted state
        - metadata: Dictionary with training metadata (empty dict if not found)
        
    Raises:
        FileNotFoundError: If required model files are missing
        ValueError: If model configuration is invalid
        
    Example:
        >>> model, preprocessor, metadata = load_model("./saved_model")
        >>> print(f"Model has {model.n_qubits} qubits")
        >>> fraud_score = model.predict(preprocessor.transform(transaction_df))
        
    References:
        Requirements 6.3, 6.4, 6.5: Deserialize and restore model state
    """
    # Import here to avoid circular dependencies
    from quantum_fraud_detector.quantum_model.vqc import VariationalQuantumClassifier
    
    # Check if directory exists
    if not os.path.exists(load_dir):
        raise FileNotFoundError(f"Model directory not found: {load_dir}")
    
    # Load VQC configuration
    config_path = os.path.join(load_dir, "model_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Model configuration not found: {config_path}. "
            "The model directory may be incomplete."
        )
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Reconstruct VQC with loaded configuration
    model = VariationalQuantumClassifier(
        n_qubits=config["n_qubits"],
        n_layers=config["n_layers"],
        device_name=config["device_name"]
    )
    
    # Load parameters
    params_path = os.path.join(load_dir, "model_params.npy")
    if not os.path.exists(params_path):
        raise FileNotFoundError(
            f"Model parameters not found: {params_path}. "
            "The model directory may be incomplete."
        )
    
    model.params = np.load(params_path)
    
    # Load preprocessor
    preprocessor_path = os.path.join(load_dir, "preprocessor.pkl")
    if not os.path.exists(preprocessor_path):
        raise FileNotFoundError(
            f"Preprocessor not found: {preprocessor_path}. "
            "The model directory may be incomplete."
        )
    
    preprocessor = joblib.load(preprocessor_path)
    
    # Load metadata (optional)
    metadata = {}
    metadata_path = os.path.join(load_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    print(f"Model loaded successfully from {load_dir}")
    
    return model, preprocessor, metadata
