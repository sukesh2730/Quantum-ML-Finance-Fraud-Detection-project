"""Local training script for Quantum ML Fraud Detector.

This script demonstrates end-to-end training workflow:
1. Generate synthetic transaction data
2. Preprocess features
3. Train quantum classifier
4. Save trained model
5. Evaluate performance
"""

import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from quantum_fraud_detector.preprocessing.preprocessor import TransactionPreprocessor
from quantum_fraud_detector.quantum_model.vqc import VariationalQuantumClassifier
from quantum_fraud_detector.training.trainer import train_model
from quantum_fraud_detector.utils.serialization import save_model


def generate_synthetic_data(n_samples: int = 500, fraud_ratio: float = 0.3, seed: int = 42):
    """Generate synthetic transaction data for training.
    
    Args:
        n_samples: Number of samples to generate
        fraud_ratio: Proportion of fraudulent transactions
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic transaction features and labels
    """
    np.random.seed(seed)
    
    n_fraud = int(n_samples * fraud_ratio)
    n_legitimate = n_samples - n_fraud
    
    # Generate legitimate transactions
    legitimate_data = {
        'amount': np.random.lognormal(mean=4.0, sigma=1.0, size=n_legitimate),
        'time_of_day': np.random.normal(loc=14, scale=4, size=n_legitimate),  # Peak at 2 PM
        'distance_from_home': np.random.exponential(scale=10, size=n_legitimate),
        'merchant_category': np.random.choice(['retail', 'grocery', 'gas', 'restaurant'], size=n_legitimate),
        'is_fraud': 0
    }
    
    # Generate fraudulent transactions (different patterns)
    fraud_data = {
        'amount': np.random.lognormal(mean=5.5, sigma=1.5, size=n_fraud),  # Higher amounts
        'time_of_day': np.random.choice([2, 3, 4, 22, 23], size=n_fraud),  # Unusual hours
        'distance_from_home': np.random.exponential(scale=50, size=n_fraud),  # Far from home
        'merchant_category': np.random.choice(['online', 'electronics', 'jewelry', 'travel'], size=n_fraud),
        'is_fraud': 1
    }
    
    # Combine and shuffle
    df_legit = pd.DataFrame(legitimate_data)
    df_fraud = pd.DataFrame(fraud_data)
    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Clip time to [0, 24] and distance to positive values
    df['time_of_day'] = df['time_of_day'].clip(0, 24)
    df['distance_from_home'] = df['distance_from_home'].clip(0, None)
    
    return df


def main():
    """Main training workflow."""
    
    print("=" * 60)
    print("Quantum ML Fraud Detector - Local Training")
    print("=" * 60)
    
    # Configuration
    N_QUBITS = 4
    N_LAYERS = 2
    EPOCHS = 15  # Reduced for faster training on quantum simulator
    LEARNING_RATE = 0.01
    TEST_SPLIT = 0.2
    MODEL_SAVE_DIR = "./saved_model"
    
    print(f"\n📋 Configuration:")
    print(f"  - Qubits: {N_QUBITS}")
    print(f"  - Layers: {N_LAYERS}")
    print(f"  - Epochs: {EPOCHS}")
    print(f"  - Learning Rate: {LEARNING_RATE}")
    print(f"  - Test Split: {TEST_SPLIT}")
    
    # Step 1: Load or generate transaction data
    print("\n🔧 Step 1: Loading transaction data...")
    
    # Option 1: Load real data (if available)
    # Uncomment and modify the path below to use your real transaction data:
    df = pd.read_csv("./data/real_transactions.csv")
    # Expected columns: amount, time_of_day, distance_from_home, merchant_category, is_fraud
    
    # Option 2: Generate synthetic data (default)
    # df = generate_synthetic_data(n_samples=500, fraud_ratio=0.3, seed=42)
    
    print(f"  ✓ Loaded {len(df)} transactions")
    print(f"    - Legitimate: {(df['is_fraud'] == 0).sum()}")
    print(f"    - Fraudulent: {(df['is_fraud'] == 1).sum()}")
    
    # Step 2: Preprocess data
    print("\n🔧 Step 2: Preprocessing data...")
    
    # Define columns
    categorical_cols = ['merchant_category']
    numerical_cols = ['amount', 'time_of_day', 'distance_from_home']
    
    # Initialize preprocessor
    preprocessor = TransactionPreprocessor(
        categorical_columns=categorical_cols,
        numerical_columns=numerical_cols
    )
    
    # Split data
    split_idx = int(len(df) * (1 - TEST_SPLIT))
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()
    
    # Fit preprocessor on training data
    y_train = df_train['is_fraud'].values
    X_train_raw = df_train.drop('is_fraud', axis=1)
    preprocessor.fit(X_train_raw)
    X_train = preprocessor.transform(X_train_raw)
    
    # Transform test data
    y_test = df_test['is_fraud'].values
    X_test_raw = df_test.drop('is_fraud', axis=1)
    X_test = preprocessor.transform(X_test_raw)
    
    print(f"  ✓ Training set: {X_train.shape[0]} samples")
    print(f"  ✓ Test set: {X_test.shape[0]} samples")
    print(f"  ✓ Features scaled to [0, π]: shape {X_train.shape}")
    
    # Step 3: Initialize quantum model
    print("\n🔧 Step 3: Initializing Quantum Classifier...")
    model = VariationalQuantumClassifier(
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        device_name="default.qubit"
    )
    model.initialize_params(seed=42)
    print(f"  ✓ VQC initialized with {model.params.size} parameters")
    
    # Step 4: Train model
    print("\n🔧 Step 4: Training model...")
    print("  (This may take a few minutes on quantum simulator)\n")
    
    history = train_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        model=model,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        optimizer="adam"
    )
    
    print(f"\n  ✓ Training complete!")
    print(f"  - Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  - Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"  - Final val accuracy: {history['val_accuracy'][-1]:.4f}")
    print(f"  - Final val F1: {history['val_f1'][-1]:.4f}")
    
    # Step 5: Evaluate on test set
    print("\n🔧 Step 5: Evaluating on test set...")
    test_predictions = model.predict_batch(X_test)
    test_pred_binary = (test_predictions >= 0.5).astype(int)
    
    # Calculate metrics manually
    correct = (test_pred_binary == y_test).sum()
    accuracy = correct / len(y_test)
    
    print(f"  ✓ Test Accuracy: {accuracy:.4f}")
    print(f"  ✓ Correct predictions: {correct}/{len(y_test)}")
    
    # Step 6: Save model
    print(f"\n🔧 Step 6: Saving model to {MODEL_SAVE_DIR}...")
    
    metadata = {
        "training_date": datetime.now().isoformat(),
        "n_qubits": N_QUBITS,
        "n_layers": N_LAYERS,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "final_train_loss": float(history['train_loss'][-1]),
        "final_val_loss": float(history['val_loss'][-1]),
        "final_val_accuracy": float(history['val_accuracy'][-1]),
        "final_val_f1": float(history['val_f1'][-1]),
        "test_accuracy": float(accuracy),
        "n_training_samples": len(X_train),
        "n_test_samples": len(X_test),
        "categorical_columns": categorical_cols,
        "numerical_columns": numerical_cols
    }
    
    save_model(model, preprocessor, MODEL_SAVE_DIR, metadata)
    
    print(f"\n{'=' * 60}")
    print("✨ Training Complete!")
    print(f"{'=' * 60}")
    print(f"\nModel saved to: {Path(MODEL_SAVE_DIR).absolute()}")
    print("\nYou can now:")
    print("  1. Run 'python inference_local.py' for command-line inference")
    print("  2. Run 'streamlit run app.py' for web interface")
    print()


if __name__ == "__main__":
    main()
