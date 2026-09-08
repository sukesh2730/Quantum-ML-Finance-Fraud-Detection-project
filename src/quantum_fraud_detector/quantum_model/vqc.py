"""Variational Quantum Classifier for fraud detection.

This module implements the VQC with angle encoding and parameterized
quantum circuits using PennyLane.
"""

import pennylane as qml
import numpy as np
import json
import os
from pathlib import Path


class VariationalQuantumClassifier:
    """Quantum binary classifier with angle encoding and variational layers.
    
    This class implements a Variational Quantum Classifier (VQC) using PennyLane.
    Features are encoded using angle encoding (rotation gates), and the circuit
    includes parameterized variational layers for learning.
    
    Attributes:
        n_qubits (int): Number of qubits in the quantum circuit.
        n_layers (int): Depth of variational layers in the circuit.
        device (qml.Device): PennyLane quantum device for circuit execution.
        params (np.ndarray or None): Trainable parameters of shape (n_layers, n_qubits, 3).
            Each parameter triplet corresponds to a Rot gate (RZ, RY, RZ rotations).
    """
    
    def __init__(self, n_qubits: int, n_layers: int, device_name: str = "default.qubit"):
        """Initialize VQC with circuit specifications.
        
        Args:
            n_qubits (int): Number of qubits in circuit (must match n_features).
            n_layers (int): Depth of variational layers.
            device_name (str, optional): PennyLane device name (simulator or hardware).
                Defaults to "default.qubit".
        
        Raises:
            ValueError: If n_qubits or n_layers are not positive integers.
        """
        if n_qubits <= 0:
            raise ValueError(f"n_qubits must be positive, got {n_qubits}")
        if n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {n_layers}")
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device_name = device_name  # Store device name for serialization
        self.device = qml.device(device_name, wires=n_qubits)
        self.params = None  # Shape: (n_layers, n_qubits, 3) for Rot gates
        
        # Create the quantum circuit as a QNode for execution
        self._qnode = qml.QNode(self._circuit_definition, self.device)
    
    def initialize_params(self, seed: int = 42) -> None:
        """Initialize variational parameters randomly.
        
        Generates random parameters for the variational quantum circuit with
        shape (n_layers, n_qubits, 3). Each parameter triplet corresponds to
        a Rot gate's three rotation angles (RZ, RY, RZ).
        
        Args:
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
        
        Notes:
            Parameters are initialized uniformly in the range [-π, π] following
            common practice for quantum circuit initialization.
            Parameters are marked as trainable for PennyLane autodiff.
        """
        rng = np.random.RandomState(seed)
        params_array = rng.uniform(
            low=-np.pi,
            high=np.pi,
            size=(self.n_layers, self.n_qubits, 3)
        )
        # Convert to PennyLane array with requires_grad=True for autodiff
        self.params = qml.numpy.array(params_array, requires_grad=True)
    
    def _circuit_definition(self, features: np.ndarray, params: np.ndarray) -> float:
        """Internal circuit definition used by QNode.
        
        This method implements the quantum circuit structure without being wrapped.
        It is used internally by the QNode wrapper created in __init__.
        
        Args:
            features (np.ndarray): Input feature vector of length n_qubits.
            params (np.ndarray): Trainable parameters of shape (n_layers, n_qubits, 3).
        
        Returns:
            float: Expectation value of Pauli-Z measurement on qubit 0.
        """
        # 1. Angle encoding layer - encode classical features as quantum states
        for i in range(self.n_qubits):
            qml.RY(features[i], wires=i)
        
        # 2. Variational layers - trainable parameterized gates repeated n_layers times
        for layer in range(self.n_layers):
            # 2a. Rotation layer - trainable single-qubit gates
            for i in range(self.n_qubits):
                qml.Rot(params[layer, i, 0], params[layer, i, 1], params[layer, i, 2], wires=i)
            
            # 2b. Entanglement layer - CNOT gates in ring topology
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
        
        # 3. Measurement - return expectation value of Pauli-Z on qubit 0
        return qml.expval(qml.PauliZ(0))
    
    def circuit(self, features: np.ndarray, params: np.ndarray) -> float:
        """Define quantum circuit with angle encoding and variational layers.
        
        This method implements a Variational Quantum Classifier circuit with:
        1. Angle encoding layer: Features encoded using RY rotation gates
        2. Variational layers: Parameterized Rot gates for learning
        3. Entanglement: CNOT gates in ring topology for qubit interaction
        
        The circuit structure follows the pattern:
        - Angle encoding (non-trainable)
        - [Variational rotations → Entanglement] × n_layers
        - Measurement on qubit 0
        
        Args:
            features (np.ndarray): Input feature vector of length n_qubits.
                Values should be in the range [0, π] for angle encoding.
            params (np.ndarray): Trainable parameters of shape (n_layers, n_qubits, 3).
                Each triplet (α, β, γ) parameterizes a Rot gate: RZ(α)RY(β)RZ(γ).
        
        Returns:
            float: Expectation value of Pauli-Z measurement on qubit 0.
                Returns a value in the range [-1, 1].
        
        Notes:
            - The Rot gate applies three consecutive rotations: RZ(α)RY(β)RZ(γ)
            - CNOT gates are applied in ring topology: qubit i to qubit (i+1) mod n_qubits
            - The final measurement extracts information from qubit 0's state
        
        References:
            Requirements 2.2, 2.3, 2.4, 2.5, 2.6
        """
        # Execute the quantum circuit via the QNode
        return self._qnode(features, params)
    
    def predict(self, features: np.ndarray) -> float:
        """Run inference on a single feature vector.
        
        This method executes the quantum circuit with the current trained parameters
        and transforms the quantum measurement result into a fraud probability score.
        
        The quantum circuit produces an expectation value in the range [-1, 1], which
        is linearly mapped to [0, 1] to represent fraud probability:
        - -1 → 0.0 (lowest fraud probability)
        -  0 → 0.5 (neutral)
        - +1 → 1.0 (highest fraud probability)
        
        Args:
            features (np.ndarray): Single feature vector of length n_qubits.
                Values should be preprocessed and scaled to [0, π] range.
        
        Returns:
            float: Fraud score in the range [0, 1], where values closer to 1
                indicate higher probability of fraud.
        
        Raises:
            ValueError: If params have not been initialized (self.params is None).
            ValueError: If features length does not match n_qubits.
        
        Examples:
            >>> vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
            >>> vqc.initialize_params(seed=42)
            >>> features = np.array([0.5, 1.0, 1.5, 2.0])
            >>> fraud_score = vqc.predict(features)
            >>> assert 0.0 <= fraud_score <= 1.0
        
        References:
            Requirement 2.7: Map quantum measurement results to Fraud_Score between 0 and 1
        """
        if self.params is None:
            raise ValueError("Model parameters not initialized. Call initialize_params() first.")
        
        if len(features) != self.n_qubits:
            raise ValueError(
                f"Feature vector length ({len(features)}) must match n_qubits ({self.n_qubits})"
            )
        
        # Execute quantum circuit to get expectation value in [-1, 1]
        expectation_value = self.circuit(features, self.params)
        
        # Map expectation value from [-1, 1] to fraud score [0, 1]
        fraud_score = (expectation_value + 1) / 2
        
        # Return as native type (handles both regular floats and autograd ArrayBox)
        return fraud_score
    
    def predict_batch(self, features: np.ndarray) -> np.ndarray:
        """Run inference on multiple feature vectors.
        
        This method processes a batch of transactions by iterating over each feature
        vector and calling predict() for individual predictions. Results are returned
        as a NumPy array of fraud scores.
        
        Args:
            features (np.ndarray): Array of feature vectors with shape (batch_size, n_qubits).
                Each row represents a single transaction's preprocessed features.
        
        Returns:
            np.ndarray: Array of fraud scores with shape (batch_size,), where each
                value is in the range [0, 1] representing fraud probability.
        
        Raises:
            ValueError: If params have not been initialized (self.params is None).
            ValueError: If features is not a 2D array.
            ValueError: If feature vector length does not match n_qubits.
        
        Examples:
            >>> vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
            >>> vqc.initialize_params(seed=42)
            >>> features_batch = np.array([
            ...     [0.5, 1.0, 1.5, 2.0],
            ...     [1.0, 1.5, 2.0, 2.5],
            ...     [0.3, 0.8, 1.2, 1.8]
            ... ])
            >>> fraud_scores = vqc.predict_batch(features_batch)
            >>> assert fraud_scores.shape == (3,)
            >>> assert all(0.0 <= score <= 1.0 for score in fraud_scores)
        
        References:
            Requirement 2.7: Batch inference for multiple transactions
        """
        if self.params is None:
            raise ValueError("Model parameters not initialized. Call initialize_params() first.")
        
        if features.ndim != 2:
            raise ValueError(
                f"Features must be a 2D array with shape (batch_size, n_qubits), got shape {features.shape}"
            )
        
        if features.shape[1] != self.n_qubits:
            raise ValueError(
                f"Feature vector length ({features.shape[1]}) must match n_qubits ({self.n_qubits})"
            )
        
        # Loop over batch of feature vectors and collect fraud scores
        fraud_scores = []
        for feature_vector in features:
            fraud_score = self.predict(feature_vector)
            fraud_scores.append(fraud_score)
        
        # Return numpy array of fraud scores
        return np.array(fraud_scores)
    
    def save(self, save_dir: str) -> None:
        """Save model parameters and configuration to disk.
        
        This method serializes the model's trainable parameters and configuration
        to disk for persistence. The saved artifacts can be loaded later to restore
        the exact model state.
        
        Files created:
        - model_params.npy: Trainable parameters array (NumPy binary format)
        - model_config.json: Model configuration (n_qubits, n_layers, device_name)
        
        Args:
            save_dir (str): Directory path where model artifacts will be saved.
                The directory will be created if it doesn't exist.
        
        Raises:
            ValueError: If params have not been initialized (self.params is None).
            OSError: If unable to create directory or write files.
        
        Examples:
            >>> vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
            >>> vqc.initialize_params(seed=42)
            >>> vqc.save("models/my_model")
            # Creates:
            #   models/my_model/model_params.npy
            #   models/my_model/model_config.json
        
        References:
            Requirement 6.1: Serialize Model_Parameters to disk in standard format
        """
        if self.params is None:
            raise ValueError("Model parameters not initialized. Cannot save uninitialized model.")
        
        # Create save directory if it doesn't exist
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save parameters array using numpy.save (binary format)
        # Convert PennyLane array to regular numpy array for saving
        params_file = save_path / "model_params.npy"
        params_to_save = np.array(self.params) if hasattr(self.params, 'requires_grad') else self.params
        np.save(str(params_file), params_to_save)
        
        # Save model configuration as JSON
        config = {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "device_name": self.device.name
        }
        config_file = save_path / "model_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load(self, load_dir: str) -> None:
        """Load model parameters and configuration from disk.
        
        This method restores the model state from saved artifacts, including
        trainable parameters and configuration. The model must have been previously
        saved using the save() method.
        
        Expected files in load_dir:
        - model_params.npy: Trainable parameters array
        - model_config.json: Model configuration
        
        Args:
            load_dir (str): Directory path containing saved model artifacts.
        
        Raises:
            FileNotFoundError: If load_dir doesn't exist or required files are missing.
            ValueError: If loaded configuration doesn't match current model configuration.
            OSError: If unable to read files.
        
        Examples:
            >>> vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
            >>> vqc.load("models/my_model")
            >>> # Model is now ready for inference with loaded parameters
        
        Notes:
            - The loaded configuration must match the current model's n_qubits and n_layers
            - This ensures parameter shape compatibility
            - The device may differ (e.g., loading model trained on simulator for hardware use)
        
        References:
            Requirement 6.3: Deserialize Model_Parameters and restore VQC state
        """
        load_path = Path(load_dir)
        
        # Check if directory exists
        if not load_path.exists():
            raise FileNotFoundError(f"Load directory does not exist: {load_dir}")
        
        # Load configuration
        config_file = load_path / "model_config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Validate configuration matches current model
        if config["n_qubits"] != self.n_qubits:
            raise ValueError(
                f"Configuration mismatch: saved model has n_qubits={config['n_qubits']}, "
                f"but current model has n_qubits={self.n_qubits}"
            )
        
        if config["n_layers"] != self.n_layers:
            raise ValueError(
                f"Configuration mismatch: saved model has n_layers={config['n_layers']}, "
                f"but current model has n_layers={self.n_layers}"
            )
        
        # Load parameters
        params_file = load_path / "model_params.npy"
        if not params_file.exists():
            raise FileNotFoundError(f"Parameters file not found: {params_file}")
        
        params_array = np.load(str(params_file))
        # Convert to PennyLane array with requires_grad=True
        self.params = qml.numpy.array(params_array, requires_grad=True)
        
        # Validate parameter shape
        expected_shape = (self.n_layers, self.n_qubits, 3)
        if self.params.shape != expected_shape:
            raise ValueError(
                f"Parameter shape mismatch: expected {expected_shape}, "
                f"but loaded parameters have shape {self.params.shape}"
            )

