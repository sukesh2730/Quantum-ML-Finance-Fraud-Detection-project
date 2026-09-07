"""Unit tests for model serialization (Task 3.6).

Tests for save and load methods of VariationalQuantumClassifier.
"""

import pytest
import numpy as np
import json
import tempfile
import shutil
from pathlib import Path
from quantum_fraud_detector.quantum_model.vqc import VariationalQuantumClassifier


class TestSaveMethod:
    """Test suite for save method."""
    
    def test_save_creates_directory(self):
        """Test that save creates the target directory if it doesn't exist."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.initialize_params(seed=42)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "new_dir" / "model"
            
            vqc.save(str(save_path))
            
            assert save_path.exists()
            assert save_path.is_dir()
    
    def test_save_creates_params_file(self):
        """Test that save creates model_params.npy file."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.initialize_params(seed=42)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            
            vqc.save(str(save_path))
            
            params_file = save_path / "model_params.npy"
            assert params_file.exists()
    
    def test_save_creates_config_file(self):
        """Test that save creates model_config.json file."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.initialize_params(seed=42)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            
            vqc.save(str(save_path))
            
            config_file = save_path / "model_config.json"
            assert config_file.exists()
    
    def test_save_params_file_content(self):
        """Test that saved params file contains correct parameter array."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.initialize_params(seed=42)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            
            original_params = vqc.params.copy()
            vqc.save(str(save_path))
            
            # Load saved params and verify
            params_file = save_path / "model_params.npy"
            loaded_params = np.load(str(params_file))
            
            np.testing.assert_array_equal(loaded_params, original_params)
    
    def test_save_config_file_content(self):
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
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            
            vqc.save(str(save_path))
            
            # Load and verify config
            config_file = save_path / "model_config.json"
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            assert config["n_qubits"] == n_qubits
            assert config["n_layers"] == n_layers
            assert config["device_name"] == device_name
    
    def test_save_raises_error_without_params(self):
        """Test that save raises ValueError if params not initialized."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            
            with pytest.raises(ValueError, match="Model parameters not initialized"):
                vqc.save(str(save_path))
    
    def test_save_with_various_dimensions(self):
        """Test save with different circuit dimensions."""
        test_cases = [
            (2, 1),
            (4, 2),
            (8, 3),
        ]
        
        for n_qubits, n_layers in test_cases:
            vqc = VariationalQuantumClassifier(n_qubits=n_qubits, n_layers=n_layers)
            vqc.initialize_params(seed=42)
            
            with tempfile.TemporaryDirectory() as tmpdir:
                save_path = Path(tmpdir) / "model"
                
                vqc.save(str(save_path))
                
                # Verify files exist
                assert (save_path / "model_params.npy").exists()
                assert (save_path / "model_config.json").exists()
                
                # Verify params shape
                loaded_params = np.load(str(save_path / "model_params.npy"))
                assert loaded_params.shape == (n_layers, n_qubits, 3)
    
    def test_save_overwrites_existing_files(self):
        """Test that save overwrites existing model files."""
        vqc1 = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc1.initialize_params(seed=42)
        
        vqc2 = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc2.initialize_params(seed=99)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            
            # Save first model
            vqc1.save(str(save_path))
            params1 = np.load(str(save_path / "model_params.npy"))
            
            # Save second model (should overwrite)
            vqc2.save(str(save_path))
            params2 = np.load(str(save_path / "model_params.npy"))
            
            # Verify params changed
            assert not np.array_equal(params1, params2)
            np.testing.assert_array_equal(params2, vqc2.params)


class TestLoadMethod:
    """Test suite for load method."""
    
    def test_load_restores_params(self):
        """Test that load correctly restores model parameters."""
        vqc_save = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc_save.initialize_params(seed=42)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            
            original_params = vqc_save.params.copy()
            vqc_save.save(str(save_path))
            
            # Create new model and load
            vqc_load = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
            vqc_load.load(str(save_path))
            
            np.testing.assert_array_equal(vqc_load.params, original_params)
    
    def test_load_validates_n_qubits(self):
        """Test that load raises ValueError if n_qubits mismatch."""
        vqc_save = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc_save.initialize_params(seed=42)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            vqc_save.save(str(save_path))
            
            # Try to load with different n_qubits
            vqc_load = VariationalQuantumClassifier(n_qubits=6, n_layers=2)
            
            with pytest.raises(ValueError, match="Configuration mismatch.*n_qubits"):
                vqc_load.load(str(save_path))
    
    def test_load_validates_n_layers(self):
        """Test that load raises ValueError if n_layers mismatch."""
        vqc_save = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc_save.initialize_params(seed=42)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            vqc_save.save(str(save_path))
            
            # Try to load with different n_layers
            vqc_load = VariationalQuantumClassifier(n_qubits=4, n_layers=3)
            
            with pytest.raises(ValueError, match="Configuration mismatch.*n_layers"):
                vqc_load.load(str(save_path))
    
    def test_load_raises_error_if_directory_not_exists(self):
        """Test that load raises FileNotFoundError if directory doesn't exist."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        
        with pytest.raises(FileNotFoundError, match="Load directory does not exist"):
            vqc.load("/nonexistent/path/to/model")
    
    def test_load_raises_error_if_params_file_missing(self):
        """Test that load raises FileNotFoundError if params file is missing."""
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
            
            with pytest.raises(FileNotFoundError, match="Parameters file not found"):
                vqc.load(str(save_path))
    
    def test_load_raises_error_if_config_file_missing(self):
        """Test that load raises FileNotFoundError if config file is missing."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            save_path.mkdir(parents=True)
            
            # Create params file but not config file
            params = np.random.uniform(-np.pi, np.pi, size=(2, 4, 3))
            np.save(str(save_path / "model_params.npy"), params)
            
            with pytest.raises(FileNotFoundError, match="Configuration file not found"):
                vqc.load(str(save_path))
    
    def test_load_validates_param_shape(self):
        """Test that load validates parameter shape matches expected dimensions."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            save_path.mkdir(parents=True)
            
            # Create config
            config = {
                "n_qubits": 4,
                "n_layers": 2,
                "device_name": "default.qubit"
            }
            with open(save_path / "model_config.json", 'w') as f:
                json.dump(config, f)
            
            # Create params with wrong shape
            wrong_params = np.random.uniform(-np.pi, np.pi, size=(2, 4, 2))  # Wrong last dim
            np.save(str(save_path / "model_params.npy"), wrong_params)
            
            with pytest.raises(ValueError, match="Parameter shape mismatch"):
                vqc.load(str(save_path))
    
    def test_load_with_various_dimensions(self):
        """Test load with different circuit dimensions."""
        test_cases = [
            (2, 1),
            (4, 2),
            (8, 3),
        ]
        
        for n_qubits, n_layers in test_cases:
            vqc_save = VariationalQuantumClassifier(n_qubits=n_qubits, n_layers=n_layers)
            vqc_save.initialize_params(seed=42)
            
            with tempfile.TemporaryDirectory() as tmpdir:
                save_path = Path(tmpdir) / "model"
                
                original_params = vqc_save.params.copy()
                vqc_save.save(str(save_path))
                
                # Load into new model
                vqc_load = VariationalQuantumClassifier(n_qubits=n_qubits, n_layers=n_layers)
                vqc_load.load(str(save_path))
                
                np.testing.assert_array_equal(vqc_load.params, original_params)
    
    def test_load_with_different_device(self):
        """Test that load works when loading model was saved with different device."""
        vqc_save = VariationalQuantumClassifier(
            n_qubits=4, 
            n_layers=2,
            device_name="default.qubit"
        )
        vqc_save.initialize_params(seed=42)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            
            original_params = vqc_save.params.copy()
            vqc_save.save(str(save_path))
            
            # Load with different device (should work)
            vqc_load = VariationalQuantumClassifier(
                n_qubits=4, 
                n_layers=2,
                device_name="lightning.qubit"
            )
            vqc_load.load(str(save_path))
            
            # Params should match even though device is different
            np.testing.assert_array_equal(vqc_load.params, original_params)


class TestSaveLoadRoundTrip:
    """Test suite for save/load round-trip property (Requirement 6.5)."""
    
    def test_round_trip_preserves_params(self):
        """Test that save then load produces identical parameters."""
        vqc_original = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc_original.initialize_params(seed=42)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            
            original_params = vqc_original.params.copy()
            
            # Save
            vqc_original.save(str(save_path))
            
            # Load
            vqc_restored = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
            vqc_restored.load(str(save_path))
            
            # Verify params are identical
            np.testing.assert_array_equal(vqc_restored.params, original_params)
    
    def test_round_trip_preserves_predictions(self):
        """Test that save then load produces equivalent predictions (Requirement 6.5)."""
        vqc_original = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc_original.initialize_params(seed=42)
        
        # Test features
        features = np.array([0.5, 1.0, 1.5, 2.0])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            
            # Get original prediction
            original_prediction = vqc_original.predict(features)
            
            # Save and load
            vqc_original.save(str(save_path))
            
            vqc_restored = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
            vqc_restored.load(str(save_path))
            
            # Get restored prediction
            restored_prediction = vqc_restored.predict(features)
            
            # Verify predictions match within numerical tolerance
            np.testing.assert_allclose(
                restored_prediction, 
                original_prediction, 
                rtol=1e-10,
                atol=1e-12
            )
    
    def test_round_trip_preserves_batch_predictions(self):
        """Test that save then load produces equivalent batch predictions."""
        vqc_original = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc_original.initialize_params(seed=42)
        
        # Test batch
        features_batch = np.array([
            [0.5, 1.0, 1.5, 2.0],
            [1.0, 1.5, 2.0, 2.5],
            [0.3, 0.8, 1.2, 1.8],
            [2.0, 2.5, 3.0, 0.5]
        ])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "model"
            
            # Get original predictions
            original_predictions = vqc_original.predict_batch(features_batch)
            
            # Save and load
            vqc_original.save(str(save_path))
            
            vqc_restored = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
            vqc_restored.load(str(save_path))
            
            # Get restored predictions
            restored_predictions = vqc_restored.predict_batch(features_batch)
            
            # Verify predictions match within numerical tolerance
            np.testing.assert_allclose(
                restored_predictions, 
                original_predictions, 
                rtol=1e-10,
                atol=1e-12
            )
    
    def test_multiple_save_load_cycles(self):
        """Test that multiple save/load cycles preserve model state."""
        vqc = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        vqc.initialize_params(seed=42)
        
        features = np.array([0.5, 1.0, 1.5, 2.0])
        original_prediction = vqc.predict(features)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Perform 3 save/load cycles
            for i in range(3):
                save_path = Path(tmpdir) / f"model_{i}"
                
                vqc.save(str(save_path))
                
                vqc_new = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
                vqc_new.load(str(save_path))
                
                prediction = vqc_new.predict(features)
                np.testing.assert_allclose(
                    prediction, 
                    original_prediction, 
                    rtol=1e-10,
                    atol=1e-12
                )
                
                vqc = vqc_new
    
    def test_round_trip_with_various_dimensions(self):
        """Test round-trip property with different circuit dimensions."""
        test_cases = [
            (2, 1),
            (4, 2),
            (8, 3),
        ]
        
        for n_qubits, n_layers in test_cases:
            vqc_original = VariationalQuantumClassifier(n_qubits=n_qubits, n_layers=n_layers)
            vqc_original.initialize_params(seed=42)
            
            features = np.random.uniform(0, np.pi, size=n_qubits)
            
            with tempfile.TemporaryDirectory() as tmpdir:
                save_path = Path(tmpdir) / "model"
                
                original_prediction = vqc_original.predict(features)
                
                vqc_original.save(str(save_path))
                
                vqc_restored = VariationalQuantumClassifier(n_qubits=n_qubits, n_layers=n_layers)
                vqc_restored.load(str(save_path))
                
                restored_prediction = vqc_restored.predict(features)
                
                np.testing.assert_allclose(
                    restored_prediction, 
                    original_prediction, 
                    rtol=1e-10,
                    atol=1e-12
                )
    
    def test_round_trip_with_different_seeds(self):
        """Test that round-trip preserves models initialized with different seeds."""
        seeds = [42, 99, 123, 456]
        
        for seed in seeds:
            vqc_original = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
            vqc_original.initialize_params(seed=seed)
            
            features = np.array([0.5, 1.0, 1.5, 2.0])
            
            with tempfile.TemporaryDirectory() as tmpdir:
                save_path = Path(tmpdir) / "model"
                
                original_prediction = vqc_original.predict(features)
                
                vqc_original.save(str(save_path))
                
                vqc_restored = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
                vqc_restored.load(str(save_path))
                
                restored_prediction = vqc_restored.predict(features)
                
                np.testing.assert_allclose(
                    restored_prediction, 
                    original_prediction, 
                    rtol=1e-10,
                    atol=1e-12
                )
