"""Unit tests for VariationalQuantumClassifier.

Tests will be implemented in later tasks.
"""

import pytest
import numpy as np
import json
import tempfile
from pathlib import Path
import pennylane as qml
from quantum_fraud_detector.quantum_model.vqc import VariationalQuantumClassifier


class TestVariationalQuantumClassifierInitialization:
    """Test suite for VQC initialization (Task 3.1)."""
    
    def test_init_with_valid_parameters(self):
        """Test VQC initialization with valid parameters."""
        n_qubits = 4
        n_layers = 2
        device_name = "default.qubit"
        
        vqc = VariationalQuantumClassifier(
            n_qubits=n_qubits,
            n_layers=n_layers,
            device_name=device_name
        )
        
        assert vqc.n_qubits == n_qubits
        assert vqc.n_layers == n_layers
        assert vqc.params is None
        assert vqc.device is not None
        assert len(vqc.device.wires) == n_qubits
    
    def test_init_with_default_device(self):
        """Test VQC initialization uses default.qubit by default."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        
        assert vqc.device is not None
        assert vqc.device.name == "default.qubit"
    
    def test_init_with_custom_device(self):
        """Test VQC initialization with custom device."""
        vqc = VariationalQuantumClassifier(
            n_qubits=4,
            n_layers=2,
            device_name="lightning.qubit"
        )
        
        assert vqc.device.name == "lightning.qubit"
    
    def test_init_params_is_none(self):
        """Test that params attribute is initialized to None."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        
        assert vqc.params is None
    
    def test_init_with_different_qubit_counts(self):
        """Test VQC initialization with various qubit counts."""
        for n_qubits in [2, 4, 8]:
            vqc = VariationalQuantumClassifier(n_qubits=n_qubits, n_layers=2)
            assert vqc.n_qubits == n_qubits
            assert len(vqc.device.wires) == n_qubits
    
    def test_init_with_different_layer_counts(self):
        """Test VQC initialization with various layer counts."""
        for n_layers in [1, 2, 5]:
            vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=n_layers)
            assert vqc.n_layers == n_layers
    
    def test_init_with_invalid_n_qubits(self):
        """Test VQC raises ValueError for invalid n_qubits."""
        with pytest.raises(ValueError, match="n_qubits must be positive"):
            VariationalQuantumClassifier(n_qubits=0, n_layers=2)
        
        with pytest.raises(ValueError, match="n_qubits must be positive"):
            VariationalQuantumClassifier(n_qubits=-1, n_layers=2)
    
    def test_init_with_invalid_n_layers(self):
        """Test VQC raises ValueError for invalid n_layers."""
        with pytest.raises(ValueError, match="n_layers must be positive"):
            VariationalQuantumClassifier(n_qubits=4, n_layers=0)
        
        with pytest.raises(ValueError, match="n_layers must be positive"):
            VariationalQuantumClassifier(n_qubits=4, n_layers=-1)


class TestInitializeParams:
    """Test suite for initialize_params method (Task 3.2)."""
    
    def test_initialize_params_shape(self):
        """Test that initialize_params creates parameters with correct shape."""
        n_qubits = 4
        n_layers = 2
        vqc = VariationalQuantumClassifier(n_qubits=n_qubits, n_layers=n_layers)
        
        vqc.initialize_params(seed=42)
        
        assert vqc.params is not None
        assert vqc.params.shape == (n_layers, n_qubits, 3)
    
    def test_initialize_params_range(self):
        """Test that parameters are initialized in range [-π, π]."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        
        vqc.initialize_params(seed=42)
        
        assert np.all(vqc.params >= -np.pi)
        assert np.all(vqc.params <= np.pi)
    
    def test_initialize_params_reproducibility(self):
        """Test that same seed produces identical parameters."""
        vqc1 = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc2 = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        
        vqc1.initialize_params(seed=42)
        vqc2.initialize_params(seed=42)
        
        np.testing.assert_array_equal(vqc1.params, vqc2.params)
    
    def test_initialize_params_different_seeds(self):
        """Test that different seeds produce different parameters."""
        vqc1 = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc2 = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        
        vqc1.initialize_params(seed=42)
        vqc2.initialize_params(seed=99)
        
        assert not np.array_equal(vqc1.params, vqc2.params)
    
    def test_initialize_params_with_various_dimensions(self):
        """Test parameter initialization with various circuit dimensions."""
        test_cases = [
            (2, 1),  # Small circuit
            (4, 2),  # Medium circuit
            (8, 3),  # Larger circuit
        ]
        
        for n_qubits, n_layers in test_cases:
            vqc = VariationalQuantumClassifier(n_qubits=n_qubits, n_layers=n_layers)
            vqc.initialize_params(seed=42)
            
            assert vqc.params.shape == (n_layers, n_qubits, 3)
            assert np.all(vqc.params >= -np.pi)
            assert np.all(vqc.params <= np.pi)
    
    def test_initialize_params_default_seed(self):
        """Test that default seed (42) is used when not specified."""
        vqc1 = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc2 = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        
        vqc1.initialize_params()  # Use default seed
        vqc2.initialize_params(seed=42)  # Explicitly use seed=42
        
        np.testing.assert_array_equal(vqc1.params, vqc2.params)
    
    def test_initialize_params_overwrites_existing(self):
        """Test that calling initialize_params multiple times overwrites previous parameters."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        
        vqc.initialize_params(seed=42)
        params_first = vqc.params.copy()
        
        vqc.initialize_params(seed=99)
        params_second = vqc.params.copy()
        
        assert not np.array_equal(params_first, params_second)


class TestQuantumCircuit:
    """Test suite for quantum circuit implementation (Task 3.3)."""
    
    def test_circuit_is_callable(self):
        """Test that circuit method is callable."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.initialize_params(seed=42)
        
        features = np.array([0.5, 1.0, 1.5, 2.0])
        
        # Circuit should be callable with features and params
        result = vqc.circuit(features, vqc.params)
        assert result is not None
    
    def test_circuit_returns_float(self):
        """Test that circuit returns a float (expectation value)."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.initialize_params(seed=42)
        
        features = np.array([0.5, 1.0, 1.5, 2.0])
        result = vqc.circuit(features, vqc.params)
        
        assert isinstance(result, (float, np.floating, np.ndarray))
    
    def test_circuit_output_range(self):
        """Test that circuit output is in range [-1, 1] (expectation value of Pauli-Z)."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.initialize_params(seed=42)
        
        # Test with various feature inputs
        test_features = [
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([np.pi/4, np.pi/2, np.pi, 2*np.pi]),
            np.array([1.0, 1.5, 2.0, 2.5]),
            np.random.uniform(0, np.pi, size=4),
        ]
        
        for features in test_features:
            result = vqc.circuit(features, vqc.params)
            result_val = float(result) if isinstance(result, np.ndarray) else result
            assert -1.0 <= result_val <= 1.0, f"Result {result_val} outside [-1, 1] range"
    
    def test_circuit_with_different_features(self):
        """Test that different feature inputs produce different outputs."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.initialize_params(seed=42)
        
        features1 = np.array([0.5, 1.0, 1.5, 2.0])
        features2 = np.array([1.0, 2.0, 2.5, 3.0])
        
        result1 = vqc.circuit(features1, vqc.params)
        result2 = vqc.circuit(features2, vqc.params)
        
        # Different inputs should generally produce different outputs
        assert not np.isclose(result1, result2), "Different features should produce different results"
    
    def test_circuit_with_different_params(self):
        """Test that different parameters produce different outputs."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        
        features = np.array([0.5, 1.0, 1.5, 2.0])
        
        vqc.initialize_params(seed=42)
        params1 = vqc.params.copy()
        result1 = vqc.circuit(features, params1)
        
        vqc.initialize_params(seed=99)
        params2 = vqc.params.copy()
        result2 = vqc.circuit(features, params2)
        
        # Different parameters should generally produce different outputs
        assert not np.isclose(result1, result2), "Different parameters should produce different results"
    
    def test_circuit_reproducibility(self):
        """Test that circuit produces consistent results with same inputs."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.initialize_params(seed=42)
        
        features = np.array([0.5, 1.0, 1.5, 2.0])
        
        result1 = vqc.circuit(features, vqc.params)
        result2 = vqc.circuit(features, vqc.params)
        
        np.testing.assert_allclose(result1, result2, rtol=1e-10)
    
    def test_circuit_with_various_dimensions(self):
        """Test circuit works with different numbers of qubits and layers."""
        test_cases = [
            (2, 1),  # Small circuit
            (4, 2),  # Medium circuit
            (8, 3),  # Larger circuit
        ]
        
        for n_qubits, n_layers in test_cases:
            vqc = VariationalQuantumClassifier(n_qubits=n_qubits, n_layers=n_layers)
            vqc.initialize_params(seed=42)
            
            features = np.random.uniform(0, np.pi, size=n_qubits)
            result = vqc.circuit(features, vqc.params)
            
            result_val = float(result) if isinstance(result, np.ndarray) else result
            assert -1.0 <= result_val <= 1.0
    
    def test_circuit_zero_features(self):
        """Test circuit with all-zero features (edge case)."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.initialize_params(seed=42)
        
        features = np.zeros(4)
        result = vqc.circuit(features, vqc.params)
        
        result_val = float(result) if isinstance(result, np.ndarray) else result
        assert -1.0 <= result_val <= 1.0
    
    def test_circuit_pi_features(self):
        """Test circuit with π-valued features (edge case)."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.initialize_params(seed=42)
        
        features = np.full(4, np.pi)
        result = vqc.circuit(features, vqc.params)
        
        result_val = float(result) if isinstance(result, np.ndarray) else result
        assert -1.0 <= result_val <= 1.0
    
    def test_circuit_angle_encoding_layer(self):
        """Test that angle encoding layer is applied correctly."""
        # This test verifies the circuit has the expected structure
        vqc = VariationalQuantumClassifier(n_qubits=2, n_layers=1)
        vqc.initialize_params(seed=42)
        
        # With minimal circuit, we can verify it runs without error
        features = np.array([np.pi/2, np.pi/4])
        result = vqc.circuit(features, vqc.params)
        
        assert result is not None
        result_val = float(result) if isinstance(result, np.ndarray) else result
        assert -1.0 <= result_val <= 1.0
    
    def test_circuit_with_single_layer(self):
        """Test circuit with minimum number of layers (n_layers=1)."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=1)
        vqc.initialize_params(seed=42)
        
        features = np.array([0.5, 1.0, 1.5, 2.0])
        result = vqc.circuit(features, vqc.params)
        
        result_val = float(result) if isinstance(result, np.ndarray) else result
        assert -1.0 <= result_val <= 1.0
    
    def test_circuit_with_multiple_layers(self):
        """Test circuit with multiple variational layers."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=5)
        vqc.initialize_params(seed=42)
        
        features = np.array([0.5, 1.0, 1.5, 2.0])
        result = vqc.circuit(features, vqc.params)
        
        result_val = float(result) if isinstance(result, np.ndarray) else result
        assert -1.0 <= result_val <= 1.0


class TestPredictMethod:
    """Test suite for predict method (Task 3.4)."""
    
    def test_predict_returns_float(self):
        """Test that predict returns a float."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.initialize_params(seed=42)
        
        features = np.array([0.5, 1.0, 1.5, 2.0])
        result = vqc.predict(features)
        
        assert isinstance(result, float)
    
    def test_predict_output_range(self):
        """Test that predict output is in range [0, 1]."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.initialize_params(seed=42)
        
        # Test with various feature inputs
        test_features = [
            np.array([0.0, 0.0, 0.0, 0.0]),
            np.array([np.pi/4, np.pi/2, np.pi, 2*np.pi]),
            np.array([1.0, 1.5, 2.0, 2.5]),
            np.random.uniform(0, np.pi, size=4),
        ]
        
        for features in test_features:
            result = vqc.predict(features)
            assert 0.0 <= result <= 1.0, f"Predict result {result} outside [0, 1] range"
    
    def test_predict_maps_expectation_values_correctly(self):
        """Test that predict correctly maps expectation values from [-1, 1] to [0, 1]."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.initialize_params(seed=42)
        
        features = np.array([0.5, 1.0, 1.5, 2.0])
        
        # Get raw expectation value from circuit
        expectation_value = vqc.circuit(features, vqc.params)
        
        # Get fraud score from predict
        fraud_score = vqc.predict(features)
        
        # Verify mapping: (expval + 1) / 2
        expected_score = (expectation_value + 1) / 2
        np.testing.assert_allclose(fraud_score, expected_score, rtol=1e-10)
    
    def test_predict_with_different_features(self):
        """Test that different feature inputs produce different predictions."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.initialize_params(seed=42)
        
        features1 = np.array([0.5, 1.0, 1.5, 2.0])
        features2 = np.array([1.0, 2.0, 2.5, 3.0])
        
        result1 = vqc.predict(features1)
        result2 = vqc.predict(features2)
        
        # Different inputs should generally produce different outputs
        assert not np.isclose(result1, result2), "Different features should produce different predictions"
    
    def test_predict_reproducibility(self):
        """Test that predict produces consistent results with same inputs."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.initialize_params(seed=42)
        
        features = np.array([0.5, 1.0, 1.5, 2.0])
        
        result1 = vqc.predict(features)
        result2 = vqc.predict(features)
        
        np.testing.assert_allclose(result1, result2, rtol=1e-10)
    
    def test_predict_raises_error_without_params(self):
        """Test that predict raises ValueError if params not initialized."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        
        features = np.array([0.5, 1.0, 1.5, 2.0])
        
        with pytest.raises(ValueError, match="Model parameters not initialized"):
            vqc.predict(features)
    
    def test_predict_raises_error_for_wrong_feature_length(self):
        """Test that predict raises ValueError if feature length doesn't match n_qubits."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.initialize_params(seed=42)
        
        # Wrong length (3 instead of 4)
        features_wrong = np.array([0.5, 1.0, 1.5])
        
        with pytest.raises(ValueError, match="Feature vector length"):
            vqc.predict(features_wrong)
    
    def test_predict_with_various_dimensions(self):
        """Test predict works with different numbers of qubits and layers."""
        test_cases = [
            (2, 1),  # Small circuit
            (4, 2),  # Medium circuit
            (8, 3),  # Larger circuit
        ]
        
        for n_qubits, n_layers in test_cases:
            vqc = VariationalQuantumClassifier(n_qubits=n_qubits, n_layers=n_layers)
            vqc.initialize_params(seed=42)
            
            features = np.random.uniform(0, np.pi, size=n_qubits)
            result = vqc.predict(features)
            
            assert isinstance(result, float)
            assert 0.0 <= result <= 1.0
    
    def test_predict_zero_features(self):
        """Test predict with all-zero features (edge case)."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.initialize_params(seed=42)
        
        features = np.zeros(4)
        result = vqc.predict(features)
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
    
    def test_predict_pi_features(self):
        """Test predict with π-valued features (edge case)."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.initialize_params(seed=42)
        
        features = np.full(4, np.pi)
        result = vqc.predict(features)
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
    
    def test_predict_boundary_mapping(self):
        """Test that extreme expectation values map to correct fraud scores."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.initialize_params(seed=42)
        
        # Test multiple feature vectors to find edge cases
        for _ in range(10):
            features = np.random.uniform(0, np.pi, size=4)
            
            expectation_value = vqc.circuit(features, vqc.params)
            fraud_score = vqc.predict(features)
            
            # Verify the mapping formula
            expected = (expectation_value + 1) / 2
            np.testing.assert_allclose(fraud_score, expected, rtol=1e-10)
            
            # Verify bounds
            assert 0.0 <= fraud_score <= 1.0
    
    def test_predict_with_different_seeds(self):
        """Test that different parameter initializations produce different predictions."""
        features = np.array([0.5, 1.0, 1.5, 2.0])
        
        vqc1 = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc1.initialize_params(seed=42)
        result1 = vqc1.predict(features)
        
        vqc2 = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc2.initialize_params(seed=99)
        result2 = vqc2.predict(features)
        
        # Different parameters should generally produce different predictions
        assert not np.isclose(result1, result2)


class TestPredictBatchMethod:
    """Test suite for predict_batch method (Task 3.5)."""
    
    def test_predict_batch_returns_numpy_array(self):
        """Test that predict_batch returns a numpy array."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.initialize_params(seed=42)
        
        features_batch = np.array([
            [0.5, 1.0, 1.5, 2.0],
            [1.0, 1.5, 2.0, 2.5],
            [0.3, 0.8, 1.2, 1.8]
        ])
        
        result = vqc.predict_batch(features_batch)
        
        assert isinstance(result, np.ndarray)
    
    def test_predict_batch_output_shape(self):
        """Test that predict_batch returns array with correct shape."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.initialize_params(seed=42)
        
        batch_sizes = [1, 3, 5, 10]
        
        for batch_size in batch_sizes:
            features_batch = np.random.uniform(0, np.pi, size=(batch_size, 4))
            result = vqc.predict_batch(features_batch)
            
            assert result.shape == (batch_size,), f"Expected shape ({batch_size},), got {result.shape}"
    
    def test_predict_batch_output_range(self):
        """Test that all predictions are in range [0, 1]."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.initialize_params(seed=42)
        
        features_batch = np.array([
            [0.0, 0.0, 0.0, 0.0],
            [np.pi/4, np.pi/2, np.pi, 2*np.pi],
            [1.0, 1.5, 2.0, 2.5],
            [0.5, 1.0, 1.5, 2.0],
            [np.pi, np.pi, np.pi, np.pi]
        ])
        
        result = vqc.predict_batch(features_batch)
        
        for score in result:
            assert 0.0 <= score <= 1.0, f"Fraud score {score} outside [0, 1] range"
    
    def test_predict_batch_matches_individual_predict(self):
        """Test that predict_batch produces same results as individual predict calls."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.initialize_params(seed=42)
        
        features_batch = np.array([
            [0.5, 1.0, 1.5, 2.0],
            [1.0, 1.5, 2.0, 2.5],
            [0.3, 0.8, 1.2, 1.8]
        ])
        
        # Get batch predictions
        batch_result = vqc.predict_batch(features_batch)
        
        # Get individual predictions
        individual_results = [vqc.predict(features) for features in features_batch]
        
        # Compare
        np.testing.assert_allclose(batch_result, individual_results, rtol=1e-10)
    
    def test_predict_batch_single_sample(self):
        """Test predict_batch with single sample (edge case)."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.initialize_params(seed=42)
        
        features_batch = np.array([[0.5, 1.0, 1.5, 2.0]])
        
        result = vqc.predict_batch(features_batch)
        
        assert result.shape == (1,)
        assert 0.0 <= result[0] <= 1.0
        
        # Should match single predict call
        single_result = vqc.predict(features_batch[0])
        np.testing.assert_allclose(result[0], single_result, rtol=1e-10)
    
    def test_predict_batch_large_batch(self):
        """Test predict_batch with larger batch size."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.initialize_params(seed=42)
        
        batch_size = 50
        features_batch = np.random.uniform(0, np.pi, size=(batch_size, 4))
        
        result = vqc.predict_batch(features_batch)
        
        assert result.shape == (batch_size,)
        assert all(0.0 <= score <= 1.0 for score in result)
    
    def test_predict_batch_raises_error_without_params(self):
        """Test that predict_batch raises ValueError if params not initialized."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        
        features_batch = np.array([
            [0.5, 1.0, 1.5, 2.0],
            [1.0, 1.5, 2.0, 2.5]
        ])
        
        with pytest.raises(ValueError, match="Model parameters not initialized"):
            vqc.predict_batch(features_batch)
    
    def test_predict_batch_raises_error_for_1d_array(self):
        """Test that predict_batch raises ValueError for 1D input."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.initialize_params(seed=42)
        
        # 1D array instead of 2D
        features_1d = np.array([0.5, 1.0, 1.5, 2.0])
        
        with pytest.raises(ValueError, match="Features must be a 2D array"):
            vqc.predict_batch(features_1d)
    
    def test_predict_batch_raises_error_for_wrong_feature_length(self):
        """Test that predict_batch raises ValueError if feature length doesn't match n_qubits."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.initialize_params(seed=42)
        
        # Wrong feature length (3 instead of 4)
        features_wrong = np.array([
            [0.5, 1.0, 1.5],
            [1.0, 1.5, 2.0]
        ])
        
        with pytest.raises(ValueError, match="Feature vector length"):
            vqc.predict_batch(features_wrong)
    
    def test_predict_batch_with_various_dimensions(self):
        """Test predict_batch works with different numbers of qubits."""
        test_cases = [
            (2, 1, 5),   # 2 qubits, 1 layer, 5 samples
            (4, 2, 10),  # 4 qubits, 2 layers, 10 samples
            (8, 3, 3),   # 8 qubits, 3 layers, 3 samples
        ]
        
        for n_qubits, n_layers, batch_size in test_cases:
            vqc = VariationalQuantumClassifier(n_qubits=n_qubits, n_layers=n_layers)
            vqc.initialize_params(seed=42)
            
            features_batch = np.random.uniform(0, np.pi, size=(batch_size, n_qubits))
            result = vqc.predict_batch(features_batch)
            
            assert result.shape == (batch_size,)
            assert all(0.0 <= score <= 1.0 for score in result)
    
    def test_predict_batch_reproducibility(self):
        """Test that predict_batch produces consistent results with same inputs."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.initialize_params(seed=42)
        
        features_batch = np.array([
            [0.5, 1.0, 1.5, 2.0],
            [1.0, 1.5, 2.0, 2.5],
            [0.3, 0.8, 1.2, 1.8]
        ])
        
        result1 = vqc.predict_batch(features_batch)
        result2 = vqc.predict_batch(features_batch)
        
        np.testing.assert_allclose(result1, result2, rtol=1e-10)
    
    def test_predict_batch_different_inputs_produce_different_outputs(self):
        """Test that different batches produce different results."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.initialize_params(seed=42)
        
        features_batch1 = np.array([
            [0.5, 1.0, 1.5, 2.0],
            [1.0, 1.5, 2.0, 2.5]
        ])
        
        features_batch2 = np.array([
            [1.5, 2.0, 2.5, 3.0],
            [0.2, 0.5, 0.8, 1.1]
        ])
        
        result1 = vqc.predict_batch(features_batch1)
        result2 = vqc.predict_batch(features_batch2)
        
        # At least some predictions should be different
        assert not np.allclose(result1, result2)
    
    def test_predict_batch_zero_features(self):
        """Test predict_batch with all-zero features (edge case)."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.initialize_params(seed=42)
        
        features_batch = np.zeros((3, 4))
        result = vqc.predict_batch(features_batch)
        
        assert result.shape == (3,)
        assert all(0.0 <= score <= 1.0 for score in result)
    
    def test_predict_batch_pi_features(self):
        """Test predict_batch with π-valued features (edge case)."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.initialize_params(seed=42)
        
        features_batch = np.full((3, 4), np.pi)
        result = vqc.predict_batch(features_batch)
        
        assert result.shape == (3,)
        assert all(0.0 <= score <= 1.0 for score in result)
    
    def test_predict_batch_empty_batch(self):
        """Test predict_batch with empty batch (edge case)."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.initialize_params(seed=42)
        
        features_batch = np.empty((0, 4))
        result = vqc.predict_batch(features_batch)
        
        assert result.shape == (0,)
        assert isinstance(result, np.ndarray)
    
    def test_predict_batch_order_independence(self):
        """Test that predict_batch results match the order of inputs."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.initialize_params(seed=42)
        
        features_batch = np.array([
            [0.5, 1.0, 1.5, 2.0],
            [1.0, 1.5, 2.0, 2.5],
            [0.3, 0.8, 1.2, 1.8]
        ])
        
        # Get batch result
        batch_result = vqc.predict_batch(features_batch)
        
        # Verify each result matches the corresponding individual prediction
        for i, features in enumerate(features_batch):
            individual_result = vqc.predict(features)
            np.testing.assert_allclose(batch_result[i], individual_result, rtol=1e-10)
    
    def test_predict_batch_with_different_seeds(self):
        """Test that different parameter initializations produce different batch predictions."""
        features_batch = np.array([
            [0.5, 1.0, 1.5, 2.0],
            [1.0, 1.5, 2.0, 2.5]
        ])
        
        vqc1 = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc1.initialize_params(seed=42)
        result1 = vqc1.predict_batch(features_batch)
        
        vqc2 = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc2.initialize_params(seed=99)
        result2 = vqc2.predict_batch(features_batch)
        
        # Different parameters should produce different predictions
        assert not np.allclose(result1, result2)
        """Test that different parameter initializations produce different batch predictions."""
        features_batch = np.array([
            [0.5, 1.0, 1.5, 2.0],
            [1.0, 1.5, 2.0, 2.5]
        ])
        
        vqc1 = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc1.initialize_params(seed=42)
        result1 = vqc1.predict_batch(features_batch)
        
        vqc2 = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc2.initialize_params(seed=99)
        result2 = vqc2.predict_batch(features_batch)
        
        # Different parameters should generally produce different predictions
        assert not np.allclose(result1, result2)


class TestSaveLoadMethods:
    """Test suite for save and load methods (Task 3.6)."""
    
    def test_save_creates_directory(self, tmp_path):
        """Test that save creates the save directory if it doesn't exist."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.initialize_params(seed=42)
        
        save_dir = tmp_path / "new_model_dir"
        assert not save_dir.exists()
        
        vqc.save(str(save_dir))
        
        assert save_dir.exists()
        assert save_dir.is_dir()
    
    def test_save_creates_params_file(self, tmp_path):
        """Test that save creates model_params.npy file."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.initialize_params(seed=42)
        
        save_dir = tmp_path / "model"
        vqc.save(str(save_dir))
        
        params_file = save_dir / "model_params.npy"
        assert params_file.exists()
        assert params_file.is_file()
    
    def test_save_creates_config_file(self, tmp_path):
        """Test that save creates model_config.json file."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.initialize_params(seed=42)
        
        save_dir = tmp_path / "model"
        vqc.save(str(save_dir))
        
        config_file = save_dir / "model_config.json"
        assert config_file.exists()
        assert config_file.is_file()
    
    def test_save_params_file_contains_correct_data(self, tmp_path):
        """Test that saved params file contains the correct parameter array."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.initialize_params(seed=42)
        
        save_dir = tmp_path / "model"
        vqc.save(str(save_dir))
        
        # Load params file directly
        params_file = save_dir / "model_params.npy"
        loaded_params = np.load(str(params_file))
        
        # Should match original params
        np.testing.assert_array_equal(loaded_params, vqc.params)
    
    def test_save_config_file_contains_correct_data(self, tmp_path):
        """Test that saved config file contains correct configuration."""
        n_qubits = 4
        n_layers = 2
        device_name = "default.qubit"
        
        vqc = VariationalQuantumClassifier(
            n_qubits=n_qubits,
            n_layers=n_layers,
            device_name=device_name
        )
        vqc.initialize_params(seed=42)
        
        save_dir = tmp_path / "model"
        vqc.save(str(save_dir))
        
        # Load config file directly
        config_file = save_dir / "model_config.json"
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        assert config["n_qubits"] == n_qubits
        assert config["n_layers"] == n_layers
        assert config["device_name"] == device_name
    
    def test_save_raises_error_without_params(self, tmp_path):
        """Test that save raises ValueError if params not initialized."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        
        save_dir = tmp_path / "model"
        
        with pytest.raises(ValueError, match="Model parameters not initialized"):
            vqc.save(str(save_dir))
    
    def test_save_with_nested_directory_path(self, tmp_path):
        """Test save with nested directory path."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.initialize_params(seed=42)
        
        save_dir = tmp_path / "models" / "experiment1" / "version1"
        vqc.save(str(save_dir))
        
        assert save_dir.exists()
        assert (save_dir / "model_params.npy").exists()
        assert (save_dir / "model_config.json").exists()
    
    def test_load_restores_params(self, tmp_path):
        """Test that load restores model parameters correctly."""
        # Create and save a model
        vqc_save = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc_save.initialize_params(seed=42)
        original_params = vqc_save.params.copy()
        
        save_dir = tmp_path / "model"
        vqc_save.save(str(save_dir))
        
        # Create new model and load
        vqc_load = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        assert vqc_load.params is None
        
        vqc_load.load(str(save_dir))
        
        # Check params are restored
        assert vqc_load.params is not None
        np.testing.assert_array_equal(vqc_load.params, original_params)
    
    def test_load_raises_error_for_nonexistent_directory(self):
        """Test that load raises FileNotFoundError for nonexistent directory."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        
        with pytest.raises(FileNotFoundError, match="Load directory does not exist"):
            vqc.load("/nonexistent/directory")
    
    def test_load_raises_error_for_missing_config_file(self, tmp_path):
        """Test that load raises FileNotFoundError if config file is missing."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        
        # Create directory without config file
        load_dir = tmp_path / "model"
        load_dir.mkdir()
        
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            vqc.load(str(load_dir))
    
    def test_load_raises_error_for_missing_params_file(self, tmp_path):
        """Test that load raises FileNotFoundError if params file is missing."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.initialize_params(seed=42)
        
        # Create directory with only config file
        load_dir = tmp_path / "model"
        load_dir.mkdir()
        
        config = {
            "n_qubits": 4,
            "n_layers": 2,
            "device_name": "default.qubit"
        }
        config_file = load_dir / "model_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f)
        
        with pytest.raises(FileNotFoundError, match="Parameters file not found"):
            vqc.load(str(load_dir))
    
    def test_load_raises_error_for_mismatched_n_qubits(self, tmp_path):
        """Test that load raises ValueError if n_qubits doesn't match."""
        # Save model with 4 qubits
        vqc_save = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc_save.initialize_params(seed=42)
        
        save_dir = tmp_path / "model"
        vqc_save.save(str(save_dir))
        
        # Try to load with 8 qubits
        vqc_load = VariationalQuantumClassifier(n_qubits=8, n_layers=2)
        
        with pytest.raises(ValueError, match="Configuration mismatch.*n_qubits"):
            vqc_load.load(str(save_dir))
    
    def test_load_raises_error_for_mismatched_n_layers(self, tmp_path):
        """Test that load raises ValueError if n_layers doesn't match."""
        # Save model with 2 layers
        vqc_save = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc_save.initialize_params(seed=42)
        
        save_dir = tmp_path / "model"
        vqc_save.save(str(save_dir))
        
        # Try to load with 3 layers
        vqc_load = VariationalQuantumClassifier(n_qubits=4, n_layers=3)
        
        with pytest.raises(ValueError, match="Configuration mismatch.*n_layers"):
            vqc_load.load(str(save_dir))
    
    def test_save_load_roundtrip_preserves_predictions(self, tmp_path):
        """Test that save/load roundtrip preserves model predictions (Requirement 6.5)."""
        # Create and initialize model
        vqc_original = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc_original.initialize_params(seed=42)
        
        # Get predictions before saving
        test_features = np.array([
            [0.5, 1.0, 1.5, 2.0],
            [1.0, 1.5, 2.0, 2.5],
            [0.3, 0.8, 1.2, 1.8]
        ])
        original_predictions = vqc_original.predict_batch(test_features)
        
        # Save model
        save_dir = tmp_path / "model"
        vqc_original.save(str(save_dir))
        
        # Load into new model
        vqc_loaded = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc_loaded.load(str(save_dir))
        
        # Get predictions after loading
        loaded_predictions = vqc_loaded.predict_batch(test_features)
        
        # Predictions should be identical within numerical tolerance
        np.testing.assert_allclose(
            loaded_predictions,
            original_predictions,
            rtol=1e-10,
            atol=1e-12
        )
    
    def test_save_load_with_different_device(self, tmp_path):
        """Test that model can be loaded with a different device."""
        # Save model with default.qubit
        vqc_save = VariationalQuantumClassifier(
            n_qubits=4,
            n_layers=2,
            device_name="default.qubit"
        )
        vqc_save.initialize_params(seed=42)
        
        save_dir = tmp_path / "model"
        vqc_save.save(str(save_dir))
        
        # Load with lightning.qubit (different device)
        vqc_load = VariationalQuantumClassifier(
            n_qubits=4,
            n_layers=2,
            device_name="lightning.qubit"
        )
        vqc_load.load(str(save_dir))
        
        # Should load successfully
        assert vqc_load.params is not None
        np.testing.assert_array_equal(vqc_load.params, vqc_save.params)
    
    def test_save_overwrite_existing_files(self, tmp_path):
        """Test that save can overwrite existing model files."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        
        save_dir = tmp_path / "model"
        
        # Save first version
        vqc.initialize_params(seed=42)
        params_v1 = vqc.params.copy()
        vqc.save(str(save_dir))
        
        # Save second version with different params
        vqc.initialize_params(seed=99)
        params_v2 = vqc.params.copy()
        vqc.save(str(save_dir))
        
        # Load and verify it's the second version
        vqc_load = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc_load.load(str(save_dir))
        
        np.testing.assert_array_equal(vqc_load.params, params_v2)
        assert not np.array_equal(vqc_load.params, params_v1)
    
    def test_save_load_with_various_dimensions(self, tmp_path):
        """Test save/load with different model dimensions."""
        test_cases = [
            (2, 1),  # Small circuit
            (4, 2),  # Medium circuit
            (8, 3),  # Larger circuit
        ]
        
        for n_qubits, n_layers in test_cases:
            # Save model
            vqc_save = VariationalQuantumClassifier(n_qubits=n_qubits, n_layers=n_layers)
            vqc_save.initialize_params(seed=42)
            
            save_dir = tmp_path / f"model_{n_qubits}q_{n_layers}l"
            vqc_save.save(str(save_dir))
            
            # Load model
            vqc_load = VariationalQuantumClassifier(n_qubits=n_qubits, n_layers=n_layers)
            vqc_load.load(str(save_dir))
            
            # Verify
            np.testing.assert_array_equal(vqc_load.params, vqc_save.params)
            assert vqc_load.params.shape == (n_layers, n_qubits, 3)
    
    def test_save_load_multiple_models_same_directory_structure(self, tmp_path):
        """Test saving and loading multiple models in separate directories."""
        models = []
        save_dirs = []
        
        for i in range(3):
            vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
            vqc.initialize_params(seed=42 + i)
            
            save_dir = tmp_path / f"model_{i}"
            vqc.save(str(save_dir))
            
            models.append(vqc)
            save_dirs.append(save_dir)
        
        # Load and verify each model
        for i, (original_model, save_dir) in enumerate(zip(models, save_dirs)):
            vqc_load = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
            vqc_load.load(str(save_dir))
            
            np.testing.assert_array_equal(vqc_load.params, original_model.params)
    
    def test_load_validates_parameter_shape(self, tmp_path):
        """Test that load validates parameter shape matches configuration."""
        # Save model normally
        vqc_save = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc_save.initialize_params(seed=42)
        
        save_dir = tmp_path / "model"
        vqc_save.save(str(save_dir))
        
        # Manually corrupt params file with wrong shape
        wrong_params = np.random.uniform(-np.pi, np.pi, size=(3, 5, 3))
        params_file = save_dir / "model_params.npy"
        np.save(str(params_file), wrong_params)
        
        # Try to load
        vqc_load = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        
        with pytest.raises(ValueError, match="Parameter shape mismatch"):
            vqc_load.load(str(save_dir))
    
    def test_save_load_json_config_is_human_readable(self, tmp_path):
        """Test that saved config JSON is properly formatted and human-readable."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.initialize_params(seed=42)
        
        save_dir = tmp_path / "model"
        vqc.save(str(save_dir))
        
        # Read config file as text
        config_file = save_dir / "model_config.json"
        with open(config_file, 'r') as f:
            config_text = f.read()
        
        # Should contain expected fields (formatted with indentation)
        assert '"n_qubits"' in config_text
        assert '"n_layers"' in config_text
        assert '"device_name"' in config_text
        assert '\n' in config_text  # Should have newlines (indented JSON)
    
    def test_save_load_integration_with_predict(self, tmp_path):
        """Integration test: save, load, and verify predictions work correctly."""
        # Create and train (initialize) model
        vqc_original = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc_original.initialize_params(seed=42)
        
        # Make predictions
        features = np.array([0.5, 1.0, 1.5, 2.0])
        original_prediction = vqc_original.predict(features)
        
        # Save model
        save_dir = tmp_path / "model"
        vqc_original.save(str(save_dir))
        
        # Create new model instance and load
        vqc_loaded = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc_loaded.load(str(save_dir))
        
        # Make prediction with loaded model
        loaded_prediction = vqc_loaded.predict(features)
        
        # Should be identical
        np.testing.assert_allclose(loaded_prediction, original_prediction, rtol=1e-10)
        assert 0.0 <= loaded_prediction <= 1.0
    
    def test_save_load_integration_with_predict_batch(self, tmp_path):
        """Integration test: save, load, and verify batch predictions work correctly."""
        # Create and initialize model
        vqc_original = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc_original.initialize_params(seed=42)
        
        # Make batch predictions
        features_batch = np.random.uniform(0, np.pi, size=(10, 4))
        original_predictions = vqc_original.predict_batch(features_batch)
        
        # Save model
        save_dir = tmp_path / "model"
        vqc_original.save(str(save_dir))
        
        # Create new model instance and load
        vqc_loaded = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc_loaded.load(str(save_dir))
        
        # Make batch predictions with loaded model
        loaded_predictions = vqc_loaded.predict_batch(features_batch)
        
        # Should be identical
        np.testing.assert_allclose(
            loaded_predictions,
            original_predictions,
            rtol=1e-10,
            atol=1e-12
        )
