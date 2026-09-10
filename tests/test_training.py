"""Unit tests for training module."""

import numpy as np
import pytest
from quantum_fraud_detector.training.trainer import compute_metrics


class TestComputeMetrics:
    """Test cases for compute_metrics function."""
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 1])  # Binary predictions, not probabilities
        
        metrics = compute_metrics(y_true, y_pred)
        
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0
    
    def test_all_wrong_predictions(self):
        """Test metrics with completely wrong predictions."""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([1, 0, 0, 1])  # Binary predictions
        
        metrics = compute_metrics(y_true, y_pred)
        
        assert metrics['accuracy'] == 0.0
        assert metrics['precision'] == 0.0
        assert metrics['recall'] == 0.0
        assert metrics['f1_score'] == 0.0
    
    def test_mixed_predictions(self):
        """Test metrics with mixed correct and incorrect predictions."""
        y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 0, 1, 1, 0])  # Binary predictions
        
        metrics = compute_metrics(y_true, y_pred)
        
        # Correct: [T, T, F, T, T, F, T, T] = 6/8 = 0.75
        assert metrics['accuracy'] == 0.75
        
        # All metrics should be between 0 and 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
    
    def test_all_positive_class(self):
        """Test when all true labels are positive."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1])  # Binary predictions
        
        metrics = compute_metrics(y_true, y_pred)
        
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0
    
    def test_all_negative_class(self):
        """Test when all true labels are negative."""
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([0, 0, 0, 0])  # Binary predictions
        
        metrics = compute_metrics(y_true, y_pred)
        
        assert metrics['accuracy'] == 1.0
        # With zero_division=0, precision/recall/f1 should handle this gracefully
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1_score'] <= 1
    
    def test_edge_case_single_sample(self):
        """Test with single sample."""
        y_true = np.array([1])
        y_pred = np.array([1])  # Binary prediction
        
        metrics = compute_metrics(y_true, y_pred)
        
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0
    
    def test_return_types(self):
        """Test that returned metrics are float types."""
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])  # Binary predictions
        
        metrics = compute_metrics(y_true, y_pred)
        
        assert isinstance(metrics, dict)
        assert isinstance(metrics['accuracy'], float)
        assert isinstance(metrics['precision'], float)
        assert isinstance(metrics['recall'], float)
        assert isinstance(metrics['f1_score'], float)


@pytest.mark.skip(reason="Current API uses Trainer class, not train_model function")
class TestTrainModel:
    """Test cases for train_model function."""
    
    def test_train_model_initialization(self):
        """Test that train_model initializes correctly with valid inputs."""
        from quantum_fraud_detector.quantum_model.vqc import VariationalQuantumClassifier
        from quantum_fraud_detector.training.trainer import train_model
        
        # Create and initialize model
        model = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        model.initialize_params(seed=42)
        
        # Create dummy training data
        X_train = np.random.uniform(0, np.pi, (50, 4))
        y_train = np.random.randint(0, 2, 50)
        X_val = np.random.uniform(0, np.pi, (10, 4))
        y_val = np.random.randint(0, 2, 10)
        
        # Call train_model
        history = train_model(X_train, y_train, X_val, y_val, model, epochs=5)
        
        # Verify history structure
        assert isinstance(history, dict)
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert 'val_accuracy' in history
        assert 'val_precision' in history
        assert 'val_recall' in history
        assert 'val_f1' in history
        
        # Verify all are lists (empty since training loop not implemented yet)
        assert isinstance(history['train_loss'], list)
        assert isinstance(history['val_loss'], list)
        assert isinstance(history['val_accuracy'], list)
        assert isinstance(history['val_precision'], list)
        assert isinstance(history['val_recall'], list)
        assert isinstance(history['val_f1'], list)
    
    def test_train_model_with_adam_optimizer(self):
        """Test train_model with Adam optimizer."""
        from quantum_fraud_detector.quantum_model.vqc import VariationalQuantumClassifier
        from quantum_fraud_detector.training.trainer import train_model
        
        model = VariationalQuantumClassifier(n_qubits=3, n_layers=1)
        model.initialize_params(seed=123)
        
        X_train = np.random.uniform(0, np.pi, (20, 3))
        y_train = np.random.randint(0, 2, 20)
        X_val = np.random.uniform(0, np.pi, (5, 3))
        y_val = np.random.randint(0, 2, 5)
        
        history = train_model(X_train, y_train, X_val, y_val, model, optimizer="adam")
        
        assert isinstance(history, dict)
        assert len(history) == 6
    
    def test_train_model_with_sgd_optimizer(self):
        """Test train_model with SGD optimizer."""
        from quantum_fraud_detector.quantum_model.vqc import VariationalQuantumClassifier
        from quantum_fraud_detector.training.trainer import train_model
        
        model = VariationalQuantumClassifier(n_qubits=3, n_layers=1)
        model.initialize_params(seed=123)
        
        X_train = np.random.uniform(0, np.pi, (20, 3))
        y_train = np.random.randint(0, 2, 20)
        X_val = np.random.uniform(0, np.pi, (5, 3))
        y_val = np.random.randint(0, 2, 5)
        
        history = train_model(X_train, y_train, X_val, y_val, model, optimizer="sgd")
        
        assert isinstance(history, dict)
        assert len(history) == 6
    
    def test_train_model_with_nesterov_optimizer(self):
        """Test train_model with Nesterov optimizer."""
        from quantum_fraud_detector.quantum_model.vqc import VariationalQuantumClassifier
        from quantum_fraud_detector.training.trainer import train_model
        
        model = VariationalQuantumClassifier(n_qubits=3, n_layers=1)
        model.initialize_params(seed=123)
        
        X_train = np.random.uniform(0, np.pi, (20, 3))
        y_train = np.random.randint(0, 2, 20)
        X_val = np.random.uniform(0, np.pi, (5, 3))
        y_val = np.random.randint(0, 2, 5)
        
        history = train_model(X_train, y_train, X_val, y_val, model, optimizer="nesterov")
        
        assert isinstance(history, dict)
        assert len(history) == 6
    
    def test_train_model_uninitialized_params(self):
        """Test that train_model raises error with uninitialized model parameters."""
        from quantum_fraud_detector.quantum_model.vqc import VariationalQuantumClassifier
        from quantum_fraud_detector.training.trainer import train_model
        
        model = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        # Do NOT initialize params
        
        X_train = np.random.uniform(0, np.pi, (20, 4))
        y_train = np.random.randint(0, 2, 20)
        X_val = np.random.uniform(0, np.pi, (5, 4))
        y_val = np.random.randint(0, 2, 5)
        
        with pytest.raises(ValueError, match="Model parameters must be initialized"):
            train_model(X_train, y_train, X_val, y_val, model)
    
    def test_train_model_invalid_optimizer(self):
        """Test that train_model raises error with invalid optimizer."""
        from quantum_fraud_detector.quantum_model.vqc import VariationalQuantumClassifier
        from quantum_fraud_detector.training.trainer import train_model
        
        model = VariationalQuantumClassifier(n_qubits=3, n_layers=1)
        model.initialize_params(seed=42)
        
        X_train = np.random.uniform(0, np.pi, (20, 3))
        y_train = np.random.randint(0, 2, 20)
        X_val = np.random.uniform(0, np.pi, (5, 3))
        y_val = np.random.randint(0, 2, 5)
        
        with pytest.raises(ValueError, match="Unsupported optimizer"):
            train_model(X_train, y_train, X_val, y_val, model, optimizer="invalid_opt")
    
    def test_train_model_dimension_mismatch_train(self):
        """Test that train_model raises error when training data dimensions don't match model."""
        from quantum_fraud_detector.quantum_model.vqc import VariationalQuantumClassifier
        from quantum_fraud_detector.training.trainer import train_model
        
        model = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        model.initialize_params(seed=42)
        
        # Create training data with wrong dimensions (3 instead of 4)
        X_train = np.random.uniform(0, np.pi, (20, 3))
        y_train = np.random.randint(0, 2, 20)
        X_val = np.random.uniform(0, np.pi, (5, 4))
        y_val = np.random.randint(0, 2, 5)
        
        with pytest.raises(ValueError, match="Training features have 3 dimensions"):
            train_model(X_train, y_train, X_val, y_val, model)
    
    def test_train_model_dimension_mismatch_val(self):
        """Test that train_model raises error when validation data dimensions don't match model."""
        from quantum_fraud_detector.quantum_model.vqc import VariationalQuantumClassifier
        from quantum_fraud_detector.training.trainer import train_model
        
        model = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
        model.initialize_params(seed=42)
        
        # Create validation data with wrong dimensions (5 instead of 4)
        X_train = np.random.uniform(0, np.pi, (20, 4))
        y_train = np.random.randint(0, 2, 20)
        X_val = np.random.uniform(0, np.pi, (5, 5))
        y_val = np.random.randint(0, 2, 5)
        
        with pytest.raises(ValueError, match="Validation features have 5 dimensions"):
            train_model(X_train, y_train, X_val, y_val, model)
    
    def test_train_model_custom_hyperparameters(self):
        """Test train_model with custom hyperparameters."""
        from quantum_fraud_detector.quantum_model.vqc import VariationalQuantumClassifier
        from quantum_fraud_detector.training.trainer import train_model
        
        model = VariationalQuantumClassifier(n_qubits=3, n_layers=1)
        model.initialize_params(seed=42)
        
        X_train = np.random.uniform(0, np.pi, (20, 3))
        y_train = np.random.randint(0, 2, 20)
        X_val = np.random.uniform(0, np.pi, (5, 3))
        y_val = np.random.randint(0, 2, 5)
        
        history = train_model(
            X_train, y_train, X_val, y_val, model,
            epochs=100,
            learning_rate=0.001,
            optimizer="adam"
        )
        
        assert isinstance(history, dict)
        assert len(history) == 6
