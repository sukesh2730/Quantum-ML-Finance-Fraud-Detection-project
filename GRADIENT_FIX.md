# Gradient Computation Fix - Training Now Works!

## Problem
The training script was timing out and the model wasn't learning due to PennyLane gradient computation issues:

**Symptoms:**
- Warning: "Attempted to differentiate a function with no trainable parameters"
- Loss staying constant (not decreasing)
- Training extremely slow (>5 minutes for 30 epochs)

## Root Cause
The VQC parameters were initialized as plain NumPy arrays without the `requires_grad=True` flag that PennyLane needs for automatic differentiation.

## Solution

### 1. Make Parameters Trainable (vqc.py)
**Changed in `src/quantum_fraud_detector/quantum_model/vqc.py`:**

```python
# BEFORE:
def initialize_params(self, seed: int = 42) -> None:
    rng = np.random.RandomState(seed)
    self.params = rng.uniform(
        low=-np.pi,
        high=np.pi,
        size=(self.n_layers, self.n_qubits, 3)
    )

# AFTER:
def initialize_params(self, seed: int = 42) -> None:
    rng = np.random.RandomState(seed)
    params_array = rng.uniform(
        low=-np.pi,
        high=np.pi,
        size=(self.n_layers, self.n_qubits, 3)
    )
    # Convert to PennyLane array with requires_grad=True for autodiff
    self.params = qml.numpy.array(params_array, requires_grad=True)
```

### 2. Fix Gradient Computation (trainer.py)
**Changed in `src/quantum_fraud_detector/training/trainer.py`:**

- Added PennyLane numpy import: `from pennylane import numpy as pnp`
- Simplified cost function to iterate over samples (avoids autograd ArrayBox issues)
- Use `pnp.log()` instead of `np.log()` for autodiff compatibility

```python
def cost_function(params):
    model.params = params
    
    # Compute loss by iterating over training samples
    epsilon = 1e-7
    total_loss = 0.0
    
    for i in range(len(X_train)):
        features = X_train[i]
        label = y_train[i]
        pred = model.predict(features)
        
        # Use PennyLane's numpy for autograd compatibility
        sample_loss = -(label * pnp.log(pred + epsilon) + 
                       (1 - label) * pnp.log(1 - pred + epsilon))
        total_loss += sample_loss
    
    return total_loss / len(X_train)
```

### 3. Fix Serialization Bug (serialization.py)
**Changed in `src/quantum_fraud_detector/utils/serialization.py`:**

```python
# BEFORE:
"device_name": model.device_name  # AttributeError!

# AFTER:
"device_name": model.device.name  # Access from device object
```

### 4. Fix Return Type (vqc.py)
**Changed in `predict()` method:**

```python
# BEFORE:
return float(fraud_score)  # TypeError with ArrayBox during gradient computation

# AFTER:
return fraud_score  # Return native type (handles both floats and ArrayBox)
```

### 5. Handle Saved Parameters (vqc.py)
**Updated `load()` and `save()` methods:**

```python
# In save():
params_to_save = np.array(self.params) if hasattr(self.params, 'requires_grad') else self.params
np.save(str(params_file), params_to_save)

# In load():
params_array = np.load(str(params_file))
self.params = qml.numpy.array(params_array, requires_grad=True)
```

## Results

✅ **Training Now Works!**

**Before Fix:**
- Loss: 0.8827 (constant, not learning)
- Time: >5 minutes, never completed
- Gradient warnings on every step

**After Fix:**
- Loss decreases: 0.8670 → 0.7370 → 0.6791
- Accuracy improves: 22% → 39% → 51%
- Training completes in ~3-4 minutes for 15 epochs
- No gradient warnings!

## Training Output Example

```
Epoch 1/15 - Train Loss: 0.8670, Val Loss: 0.8545, Val Accuracy: 0.2200, Val F1: 0.2642
Epoch 10/15 - Train Loss: 0.7370, Val Loss: 0.7322, Val Accuracy: 0.3900, Val F1: 0.3146
  ✓ Training complete!
  - Final train loss: 0.6791
  - Final val loss: 0.6771
  - Final val accuracy: 0.5100
  - Final val F1: 0.3288
```

## Configuration Change
Reduced epochs from 30 to 15 in `train_local.py` for faster training during development:

```python
EPOCHS = 15  # Reduced for faster training on quantum simulator
```

## Files Modified

1. `src/quantum_fraud_detector/quantum_model/vqc.py`
   - `initialize_params()`: Add requires_grad=True
   - `predict()`: Remove float() conversion
   - `save()`: Handle PennyLane arrays
   - `load()`: Restore with requires_grad=True

2. `src/quantum_fraud_detector/training/trainer.py`
   - Import PennyLane numpy as pnp
   - Refactor cost_function() for autograd compatibility
   - Use pnp.log() instead of np.log()

3. `src/quantum_fraud_detector/utils/serialization.py`
   - Fix device_name access: `model.device.name`

4. `train_local.py`
   - Reduce epochs: 30 → 15

## Next Steps

- ✅ Training script working
- ✅ Model learning (loss decreasing)
- ✅ Serialization working
- 🔄 Training in progress (background process)
- ⏳ Once complete, model will be saved to `./saved_model/`
- ⏳ Web interface at http://localhost:8501 will be fully functional

## Performance Notes

- Quantum simulation is computationally expensive
- 15 epochs with 400 samples takes ~3-4 minutes
- Each epoch processes 400 transactions through 4-qubit quantum circuit
- Acceptable for development; consider using `lightning.qubit` device for faster simulation
- For production, consider reducing training set size or using hardware acceleration

## Technical Details

**PennyLane Autodiff Requirements:**
- Parameters must be `qml.numpy.array` with `requires_grad=True`
- Operations must use PennyLane/autograd-compatible numpy functions
- Avoid type conversions (float(), int()) during gradient tracing
- ArrayBox objects are used internally during gradient computation

**Quantum Circuit:**
- 4 qubits, 2 layers
- 24 trainable parameters (2 × 4 × 3 for Rot gates)
- RY angle encoding
- Ring topology CNOT entanglement
- Pauli-Z measurement on qubit 0

**Loss Function:**
- Binary cross-entropy with epsilon=1e-7
- Computed per-sample to avoid autograd issues
- Mean aggregation over training set

---

**Status**: ✅ **FIXED - Training Working**
**Date**: 2025-01-XX
**Tested**: Python 3.13, PennyLane, 400 training samples
