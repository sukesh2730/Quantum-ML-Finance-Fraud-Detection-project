
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp  # PennyLane's numpy for autograd compatibility
from typing import Dict, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes a dictionary of common classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }
    return metrics


class Trainer:
    """Trains a VariationalQuantumClassifier model."""

    def __init__(self, model, learning_rate: float, epochs: int):
        self.model = model
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.optimizer = qml.GradientDescentOptimizer(stepsize=learning_rate)

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, list]:
        """Trains the VQC model using gradient descent."""
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

        for epoch in range(self.epochs):
            # Step and cost calculation for VQC
            self.model.weights, current_cost = self.optimizer.step_and_cost(lambda p: self.model.cost_fn(X_train, y_train, p), self.model.weights)

            # Evaluate training accuracy
            train_preds = self.model.predict(X_train)
            train_accuracy = accuracy_score(y_train, train_preds)

            history['loss'].append(current_cost)
            history['accuracy'].append(train_accuracy)

            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {current_cost:.4f}, Train Acc: {train_accuracy:.4f}", end="")

            if X_val is not None and y_val is not None:
                val_metrics = self.evaluate(X_val, y_val)
                # Placeholder for now, actual val_loss could be computed using model.cost_fn on X_val
                history['val_loss'].append(0.0)
                history['val_accuracy'].append(val_metrics.get('accuracy', 0.0))
                print(f", Val Acc: {val_metrics.get('accuracy', 0.0):.4f}")
            else:
                print()

        return history

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluates the model on the given dataset."""
        predictions = self.model.predict(X)
        metrics = compute_metrics(y, predictions)
        return metrics
