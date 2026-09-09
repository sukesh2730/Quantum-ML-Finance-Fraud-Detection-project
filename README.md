# ⚛️ Quantum ML Financial Fraud Detector

A Variational Quantum Classifier (VQC) built with PennyLane that detects credit card fraud using real transaction data from the OpenML Credit Card Fraud dataset.

---

## Results

| Metric | Value |
|---|---|
| F1 Score (Fraud) | **0.78** |
| Precision | **93%** |
| Recall | **67%** |
| Accuracy | **81%** |
| Fraud caught (test set) | **66 / 98** |
| False alarms | **5 / 99** |

Trained on 984 real transactions (492 fraud + 492 legitimate), balanced from 284,807 real credit card transactions.

---

## How It Works

### Quantum Circuit

```
Feature (V17) → RY gate → Qubit 0 ─┐
Feature (V14) → RY gate → Qubit 1 ─┤  CNOT ring entanglement
Feature (V12) → RY gate → Qubit 2 ─┤  + Rot gates (RZ→RY→RZ)
Feature (V10) → RY gate → Qubit 3 ─┘
                                    ↓
                         Pauli-Z measurement → fraud score [0,1]
```

1. **Angle Encoding** — each of the 4 input features is encoded into a qubit via an RY rotation gate
2. **Variational Layers (×2)** — parameterized Rot gates (24 trainable parameters total) learn the fraud pattern
3. **CNOT Entanglement** — ring topology (0→1→2→3→0) creates quantum correlations between features
4. **Measurement** — Pauli-Z expectation value on qubit 0, mapped to [0,1] as fraud probability

### Feature Selection

The dataset has 28 PCA-anonymized features (V1–V28) + Amount. We selected the top 4 by absolute correlation with the fraud label:

| Feature | Description | Fraud correlation |
|---|---|---|
| V17 | Transaction pattern component | Highest |
| V14 | Merchant behaviour component | High |
| V12 | Cardholder activity component | High |
| V10 | Location & time component | High |

Fraud transactions cluster strongly in the negative range of these components.

---

## Architecture

```
src/
└── quantum_fraud_detector/
    ├── quantum_model/
    │   └── vqc.py          # VQC circuit definition
    ├── preprocessing/
    │   └── preprocessor.py # Feature scaling (MinMaxScaler → [0.1, 3.04])
    ├── training/
    │   └── trainer.py      # Trainer class with Adam optimizer
    ├── utils/
    │   └── serialization.py
    └── config/
        └── config.py

saved_model/
├── model_params.npy    # 24 trained quantum parameters (shape: 2,4,3)
├── model_config.json   # Architecture + feature + threshold config
├── scaler.pkl          # Fitted MinMaxScaler
└── metadata.json       # Full training metrics and dataset info

app.py                  # Streamlit web interface
```

---

## Training Setup

| Parameter | Value |
|---|---|
| Dataset | OpenML Credit Card Fraud (ID 1597) |
| Total dataset size | 284,807 transactions |
| Fraud cases | 492 (0.17%) |
| Balanced sample used | 984 (492 fraud + 492 legit) |
| Train / Test split | 80% / 20% |
| Qubits | 4 |
| Variational layers | 2 |
| Trainable parameters | 24 |
| Optimizer | Adam |
| Learning rate | 0.05 |
| Epochs | 50 |
| Best seed | 7 |
| Decision threshold | 0.50 |
| Training environment | Google Colab |

---

## Confusion Matrix (Test Set — 197 transactions)

|  | Predicted Legitimate | Predicted Fraud |
|---|---|---|
| **Actually Legitimate** | 94 ✅ | 5 ❌ |
| **Actually Fraud** | 32 ❌ | 66 ✅ |

- **93% precision** — when it flags fraud, it's almost always right. Minimal false alarms.
- **67% recall** — catches 2 out of 3 fraud cases. Known limitation of 4-qubit VQCs on complex data.

---

## Running the App

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit interface
streamlit run app.py
```

The app loads the saved model and lets you input V17, V14, V12, V10 values to get a real-time fraud risk score with a gauge chart.

---

## Known Limitations

- **4 qubits = limited expressibility.** A classical model like XGBoost achieves F1 > 0.85 on this dataset. The VQC trades some performance for quantum novelty.
- **Balanced training set is small (984 samples).** The full dataset has 284,807 rows — scaling up training would improve recall.
- **PCA features are anonymized.** V1–V28 have no interpretable financial meaning, which limits explainability.
- **Simulated quantum hardware.** Uses PennyLane's `default.qubit` simulator, not real quantum hardware.

---

## Tech Stack

- [PennyLane](https://pennylane.ai/) — quantum circuit definition and autodiff training
- [Streamlit](https://streamlit.io/) — web interface
- [Plotly](https://plotly.com/) — gauge chart visualization
- [scikit-learn](https://scikit-learn.org/) — preprocessing and metrics
- [OpenML](https://www.openml.org/d/1597) — real credit card fraud dataset
- [Google Colab](https://colab.research.google.com/) — training environment
- [Kiro](https://kiro.dev/) — development environment

---

## Repository

[github.com/sukesh2730/Quantum-ML-Finance-Fraud-Detection-project](https://github.com/sukesh2730/Quantum-ML-Finance-Fraud-Detection-project)
