# 🚀 Quick Start Guide

Get up and running with the Quantum ML Fraud Detector in 5 minutes!

## Step 1: Install Dependencies (1 minute)

```bash
pip install -r requirements.txt
pip install -e .
```

## Step 2: Train the Model (2-3 minutes)

```bash
python train_local.py
```

**What this does:**
- Generates 500 synthetic transactions
- Trains a 4-qubit quantum classifier
- Saves the model to `./saved_model/`

**Expected output:**
```
🔧 Step 4: Training model...
Epoch 10/30 - Train Loss: 0.5234, Val Loss: 0.5421, Val Accuracy: 0.7500, Val F1: 0.7234
Epoch 20/30 - Train Loss: 0.4567, Val Loss: 0.4789, Val Accuracy: 0.8100, Val F1: 0.7856
Epoch 30/30 - Train Loss: 0.4123, Val Loss: 0.4567, Val Accuracy: 0.8200, Val F1: 0.7945

✓ Training complete!
✨ Training Complete!
Model saved to: ./saved_model
```

## Step 3: Test with Command Line (30 seconds)

```bash
python inference_local.py
```

This runs predictions on sample transactions and shows you how the classifier works.

## Step 4: Launch Web Interface (30 seconds)

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## 🎯 Try These in the Web Interface:

### Example 1: Legitimate Transaction
- **Amount**: $45
- **Time**: 14:00 (2 PM)
- **Distance**: 5 km
- **Category**: Grocery
- **Expected**: ✅ Low fraud score (~0.2-0.3)

### Example 2: Suspicious Transaction
- **Amount**: $2500
- **Time**: 03:00 (3 AM)
- **Distance**: 500 km
- **Category**: Electronics
- **Expected**: 🚨 High fraud score (~0.7-0.9)

## 📊 Understanding the Results

- **Fraud Score < 0.25**: Very Low Risk ✅
- **Fraud Score 0.25-0.5**: Low-Medium Risk ℹ️
- **Fraud Score 0.5-0.75**: Medium-High Risk ⚠️
- **Fraud Score > 0.75**: High Risk 🚨

## 🔧 Customization

### Change Model Size

Edit `train_local.py`:

```python
N_QUBITS = 6  # More qubits = more capacity (but slower)
N_LAYERS = 3  # More layers = deeper circuit
EPOCHS = 50   # More epochs = better training (but longer)
```

### Use Your Own Data

Replace the `generate_synthetic_data()` function in `train_local.py` with:

```python
df = pd.read_csv('your_data.csv')
# Ensure columns: amount, time_of_day, distance_from_home, merchant_category, is_fraud
```

## 🆘 Common Issues

### "Model not found" error
**Solution**: Run `python train_local.py` first

### Import errors
**Solution**: Run `pip install -e .`

### Streamlit won't start
**Solution**: Try a different port: `streamlit run app.py --server.port 8502`

## 🎓 Next Steps

1. **Experiment** with different transaction patterns
2. **Modify** hyperparameters to improve accuracy
3. **Train** on your own fraud detection dataset
4. **Explore** the quantum circuit design in `src/quantum_fraud_detector/quantum_model/vqc.py`

## 📚 Learn More

- Read the full [README.md](README.md) for detailed documentation
- Check the [design document](.kiro/specs/quantum-ml-fraud-detector/design.md) for architecture details
- Run tests: `pytest tests/ -v`

---

**Have fun exploring quantum machine learning! 🔮**
