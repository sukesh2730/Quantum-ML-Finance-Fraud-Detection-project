"""Local inference script for Quantum ML Fraud Detector.

This script demonstrates loading a trained model and making predictions
on new transaction data.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.quantum_fraud_detector.utils.serialization import load_model


def create_sample_transaction():
    """Create a sample transaction for testing."""
    return pd.DataFrame([{
        'amount': 850.50,
        'time_of_day': 23.5,  # 11:30 PM - suspicious
        'distance_from_home': 150.0,  # Far from home
        'merchant_category': 'online'
    }])


def main():
    """Main inference workflow."""
    
    print("=" * 60)
    print("Quantum ML Fraud Detector - Local Inference")
    print("=" * 60)
    
    MODEL_DIR = "./saved_model"
    
    # Check if model exists
    if not Path(MODEL_DIR).exists():
        print(f"\n❌ Error: Model directory not found: {MODEL_DIR}")
        print("Please run 'python train_local.py' first to train a model.")
        return
    
    # Load model
    print(f"\n🔧 Loading model from {MODEL_DIR}...")
    model, preprocessor, metadata = load_model(MODEL_DIR)
    
    print(f"  ✓ Model loaded successfully")
    print(f"  - Qubits: {model.n_qubits}")
    print(f"  - Layers: {model.n_layers}")
    
    if metadata:
        print(f"  - Training date: {metadata.get('training_date', 'N/A')}")
        print(f"  - Test accuracy: {metadata.get('test_accuracy', 'N/A'):.4f}")
    
    # Create sample transaction
    print("\n🔧 Creating sample transaction...")
    transaction = create_sample_transaction()
    
    print("\n📊 Transaction Details:")
    for col, value in transaction.iloc[0].items():
        print(f"  - {col}: {value}")
    
    # Preprocess
    print("\n🔧 Preprocessing transaction...")
    X = preprocessor.transform(transaction)
    print(f"  ✓ Features scaled to [0, π]: {X[0]}")
    
    # Make prediction
    print("\n🔧 Running quantum inference...")
    fraud_score = model.predict(X[0])
    
    print(f"\n{'=' * 60}")
    print("✨ Prediction Result:")
    print(f"{'=' * 60}")
    print(f"\n  Fraud Score: {fraud_score:.4f}")
    print(f"  Classification: {'🚨 FRAUD' if fraud_score >= 0.5 else '✅ LEGITIMATE'}")
    print(f"  Confidence: {abs(fraud_score - 0.5) * 200:.1f}%")
    
    # Try multiple transactions
    print(f"\n{'=' * 60}")
    print("Testing Multiple Scenarios:")
    print(f"{'=' * 60}\n")
    
    test_cases = [
        {
            'name': 'Legitimate: Small local purchase',
            'amount': 45.00,
            'time_of_day': 14.0,  # 2 PM
            'distance_from_home': 5.0,
            'merchant_category': 'grocery'
        },
        {
            'name': 'Suspicious: Large late-night online purchase',
            'amount': 2500.00,
            'time_of_day': 3.0,  # 3 AM
            'distance_from_home': 500.0,
            'merchant_category': 'electronics'
        },
        {
            'name': 'Legitimate: Regular restaurant',
            'amount': 75.00,
            'time_of_day': 19.0,  # 7 PM
            'distance_from_home': 8.0,
            'merchant_category': 'restaurant'
        },
        {
            'name': 'Suspicious: Jewelry purchase far away',
            'amount': 1800.00,
            'time_of_day': 22.0,  # 10 PM
            'distance_from_home': 300.0,
            'merchant_category': 'jewelry'
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        name = test_case.pop('name')
        df = pd.DataFrame([test_case])
        X = preprocessor.transform(df)
        score = model.predict(X[0])
        
        print(f"{i}. {name}")
        print(f"   Score: {score:.4f} - {'🚨 FRAUD' if score >= 0.5 else '✅ LEGITIMATE'}")
        print()
    
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
