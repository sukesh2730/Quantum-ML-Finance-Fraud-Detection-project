"""Unit tests for model serialization (Task 3.6).

Tests for save_model and load_model functions from serialization module.
"""

import pytest
import numpy as np
import json
import tempfile
import shutil
from pathlib import Path
from quantum_fraud_detector.quantum_model.vqc import VariationalQuantumClassifier
from quantum_fraud_detector.utils.serialization import save_model, load_model
from quantum_fraud_detector.preprocessing.preprocessor import TransactionPreprocessor


@pytest.fixture
def mock_preprocessor():
    """Create a minimal mock preprocessor that appears fitted."""
    preprocessor = TransactionPreprocessor(
        categorical_columns=[],
        numerical_columns=[]
    )
    # Make it appear fitted
    preprocessor.is_fitted = True
    return preprocessor


class TestSaveModelFunction:
    """Test suite for save_model function."""
    
    def test_save_creates_directory(self, mock_preprocessor):
        """Test that save_model creates the target directory if it doesn't exist."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "new_dir" / "model"
            
            save_model(vqc, mock_preprocessor, str(save_path))
            
            assert save_path.exists()
            assert save_path.is_dir()
    
    def test_save_creates_params_file(self, mock_preprocessor):
        """Test that save_model creates model_params.npy file."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            
            save_model(vqc, mock_preprocessor, str(save_path))
            
            params_file = save_path / "model_params.npy"
            assert params_file.exists()
    
    def test_save_creates_config_file(self, mock_preprocessor):
        """Test that save_model creates model_config.json file."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            
            save_model(vqc, mock_preprocessor, str(save_path))
            
            config_file = save_path / "model_config.json"
            assert config_file.exists()
    
    def test_save_params_file_content(self, mock_preprocessor):
        """Test that saved params file contains correct parameter array."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            
            original_params = vqc.weights.copy()
            save_model(vqc, mock_preprocessor, str(save_path))
            
            # Load saved params and verify
            params_file = save_path / "model_params.npy"
            loaded_params = np.load(str(params_file))
            
            np.testing.assert_array_equal(loaded_params, original_params)
    
    def test_save_config_file_content(self, mock_preprocessor):
        """Test that saved config file contains correct configuration."""
        n_qubits = 4
        n_layers = 2
        
        vqc = VariationalQuantumClassifier(
            n_qubits=n_qubits, 
            n_layers=n_layers
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            
            save_model(vqc, mock_preprocessor, str(save_path))
            
            # Load and verify config
            config_file = save_path / "model_config.json"
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            assert config["n_qubits"] == n_qubits
            assert config["n_layers"] == n_layers
            assert "device_name" in config
    
    def test_save_raises_error_without_params(self, mock_preprocessor):
        """Test that save_model raises ValueError if weights not initialized."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.weights = None  # Simulate uninitialized weights
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            
            with pytest.raises(ValueError, match="Model parameters must be initialized"):
                save_model(vqc, mock_preprocessor, str(save_path))
    
    def test_save_with_various_dimensions(self, mock_preprocessor):
        """Test save_model with different circuit dimensions."""
        test_cases = [
            (2, 1),
            (4, 2),
            (8, 3),
        ]
        
        for n_qubits, n_layers in test_cases:
            vqc = VariationalQuantumClassifier(n_qubits=n_qubits, n_layers=n_layers)
            
            with tempfile.TemporaryDirectory() as tmpdir:
                save_path = Path(tmpdir) / "model"
                
                save_model(vqc, mock_preprocessor, str(save_path))
                
                # Verify files exist
                assert (save_path / "model_params.npy").exists()
                assert (save_path / "model_config.json").exists()
                
                # Verify params shape
                loaded_params = np.load(str(save_path / "model_params.npy"))
                assert loaded_params.shape == (n_layers, n_qubits, 3)
    
    def test_save_overwrites_existing_files(self, mock_preprocessor):
        """Test that save_model overwrites existing model files."""
        vqc1 = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc2 = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            
            # Save first model
            params1 = vqc1.weights.copy()
            save_model(vqc1, mock_preprocessor, str(save_path))
            loaded_params1 = np.load(str(save_path / "model_params.npy"))
            
            # Save second model (should overwrite)
            params2 = vqc2.weights.copy()
            save_model(vqc2, mock_preprocessor, str(save_path))
            loaded_params2 = np.load(str(save_path / "model_params.npy"))
            
            # Verify params changed
            np.testing.assert_array_equal(loaded_params2, params2)


class TestLoadModelFunction:
    """Test suite for load_model function."""
    
    def test_load_restores_params(self, mock_preprocessor):
        """Test that load_model correctly restores model parameters."""
        vqc_save = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            
            original_params = vqc_save.weights.copy()
            save_model(vqc_save, mock_preprocessor, str(save_path))
            
            # Load model
            vqc_load, _, _ = load_model(str(save_path))
            
            np.testing.assert_array_equal(vqc_load.weights, original_params)
    
    def test_load_raises_error_if_directory_not_exists(self):
        """Test that load_model raises FileNotFoundError if directory doesn't exist."""
        
        with pytest.raises(FileNotFoundError, match="Model directory not found"):
            load_model("/nonexistent/path/to/model")
    
    def test_load_raises_error_if_params_file_missing(self, mock_preprocessor):
        """Test that load_model raises FileNotFoundError if params file is missing."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            save_path.mkdir(parents=True)
            
            # Create config file but not params file
            config = {
                "n_qubits": 4,
                "n_layers": 2,
                "device_name": "default.qubit"
            }
            with open(save_path / "model_config.json", 'w') as f:
                json.dump(config, f)
            
            with pytest.raises(FileNotFoundError, match="Model parameters not found"):
                load_model(str(save_path))
    
    def test_load_raises_error_if_config_file_missing(self):
        """Test that load_model raises FileNotFoundError if config file is missing."""
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            save_path.mkdir(parents=True)
            
            # Create params file but not config file
            params = np.random.uniform(-np.pi, np.pi, size=(2, 4, 3))
            np.save(str(save_path / "model_params.npy"), params)
            
            with pytest.raises(FileNotFoundError, match="Model configuration not found"):
                load_model(str(save_path))
    
    def test_load_with_various_dimensions(self, mock_preprocessor):
        """Test load_model with different circuit dimensions."""
        test_cases = [
            (2, 1),
            (4, 2),
            (8, 3),
        ]
        
        for n_qubits, n_layers in test_cases:
            vqc_save = VariationalQuantumClassifier(n_qubits=n_qubits, n_layers=n_layers)
            
            with tempfile.TemporaryDirectory() as tmpdir:
                save_path = Path(tmpdir) / "model"
                
                original_params = vqc_save.weights.copy()
                save_model(vqc_save, mock_preprocessor, str(save_path))
                
                # Load model
                vqc_load, _, _ = load_model(str(save_path))
                
                np.testing.assert_array_equal(vqc_load.weights, original_params)


class TestSaveLoadRoundTrip:
    """Test suite for save/load round-trip property (Requirement 6.5)."""
    
    def test_round_trip_preserves_params(self, mock_preprocessor):
        """Test that save then load produces identical parameters."""
        vqc_original = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            
            original_params = vqc_original.weights.copy()
            
            # Save
            save_model(vqc_original, mock_preprocessor, str(save_path))
            
            # Load
            vqc_restored, _, _ = load_model(str(save_path))
            
            # Verify params are identical
            np.testing.assert_array_equal(vqc_restored.weights, original_params)
    
    def test_round_trip_preserves_predictions(self, mock_preprocessor):
        """Test that save then load produces equivalent predictions (Requirement 6.5)."""
        vqc_original = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        
        # Test features
        features = np.array([[0.5, 1.0, 1.5, 2.0]])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            
            # Get original prediction
            original_prediction = vqc_original.predict(features)
            
            # Save and load
            save_model(vqc_original, mock_preprocessor, str(save_path))
            
            vqc_restored, _, _ = load_model(str(save_path))
            
            # Get restored prediction
            restored_prediction = vqc_restored.predict(features)
            
            # Verify predictions match
            np.testing.assert_array_equal(restored_prediction, original_prediction)
    
    def test_round_trip_with_various_dimensions(self, mock_preprocessor):
        """Test round-trip property with different circuit dimensions."""
        test_cases = [
            (2, 1),
            (4, 2),
            (8, 3),
        ]
        
        for n_qubits, n_layers in test_cases:
            vqc_original = VariationalQuantumClassifier(n_qubits=n_qubits, n_layers=n_layers)
            
            features = np.random.uniform(0, np.pi, size=(1, n_qubits))
            
            with tempfile.TemporaryDirectory() as tmpdir:
                save_path = Path(tmpdir) / "model"
                
                original_prediction = vqc_original.predict(features)
                
                save_model(vqc_original, mock_preprocessor, str(save_path))
                
                vqc_restored, _, _ = load_model(str(save_path))
                
                restored_prediction = vqc_restored.predict(features)
                
                np.testing.assert_array_equal(restored_prediction, original_prediction)
