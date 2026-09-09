
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np # For np.pi

class VariationalQuantumClassifier:
    """
    A Variational Quantum Classifier (VQC) using PennyLane.
    """
    def __init__(self, n_qubits: int, n_layers: int):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Initialize trainable weights for the variational circuit
        # Using a common shape for StronglyEntanglingLayers: (n_layers, n_qubits, 3)
        self.weights = pnp.random.uniform(
            low=-np.pi/2, high=np.pi/2,
            size=(n_layers, n_qubits, 3), # Shape determined by VQC architecture
            requires_grad=True
        )

        # Build the QNode using the device and variational circuit
        self.qnode = self._build_qnode()

    def _variational_circuit(self, features, weights):
        """
        The quantum circuit ansatz.
        Assumes features are angle-encoded and then followed by strongly entangling layers.
        """
        # Angle embedding of features
        qml.AngleEmbedding(features, wires=range(self.n_qubits))

        # Variational layers
        qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))

        # Measurement
        return qml.expval(qml.PauliZ(0)) # Measuring the first qubit for classification

    def _build_qnode(self):
        """Constructs the PennyLane QNode."""
        @qml.qnode(self.dev, interface="autograd")
        def circuit(features, weights):
            return self._variational_circuit(features, weights)
        return circuit

    def cost_fn(self, features, labels, weights):
        """
        Computes the cost function (e.g., mean squared error) for VQC training.
        Assumes labels are 0 or 1.
        """
        # Get predictions from the quantum circuit
        predictions = pnp.array([self.qnode(f, weights) for f in features])

        # Map expectation values (-1 to 1) to the range [0, 1]
        mapped_predictions = (predictions + 1) / 2

        # Compute mean squared error
        cost = pnp.mean((mapped_predictions - labels)**2)
        return cost

    def predict(self, features):
        """
        Makes predictions on new data using the trained weights.
        """
        # Use the stored self.weights for prediction
        predictions = pnp.array([self.qnode(f, self.weights) for f in features])

        # Convert expectation values to binary class labels (0 or 1)
        return pnp.where(predictions > 0, 1, 0)
