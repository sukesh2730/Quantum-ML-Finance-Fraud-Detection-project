# 🌐 Web Interface Guide

## Launching the Interface

```bash
streamlit run app.py
```

Then open your browser to: **http://localhost:8501**

## 🎨 Interface Layout

### Header Section
- **Title**: "🔮 Quantum ML Fraud Detector"
- **Description**: Overview of the quantum classifier

### Left Panel: Transaction Input

#### Transaction Details Form
1. **Transaction Amount ($)**
   - Slider: $0.01 to $10,000
   - Default: $150
   - Higher amounts may indicate fraud

2. **Time of Day (Hour)**
   - Slider: 0-23 (24-hour format)
   - Default: 14 (2 PM)
   - Late night (2-5 AM) or early morning transactions are suspicious

3. **Distance from Home (km)**
   - Input: 0 to 1,000 km
   - Default: 10 km
   - Transactions far from home (>100km) raise suspicion

4. **Merchant Category**
   - Dropdown options:
     - `retail` ✅ Low risk
     - `grocery` ✅ Low risk
     - `gas` ✅ Low risk
     - `restaurant` ✅ Low risk
     - `online` ⚠️ Medium risk
     - `electronics` 🚨 High risk
     - `jewelry` 🚨 High risk
     - `travel` ⚠️ Medium risk

5. **Analyze Transaction Button**
   - Click to run quantum inference

### Right Panel: Fraud Risk Assessment

#### Gauge Chart
- **Visual representation** of fraud score (0 to 1)
- **Color coding**:
  - Green: Score < 0.5 (Legitimate)
  - Red: Score ≥ 0.5 (Fraud)
- **Threshold line** at 0.5

#### Risk Level Indicators
- 🚨 **HIGH RISK** (Score ≥ 0.75)
  - Red alert banner
  - Recommendation: Block transaction
  
- ⚠️ **MEDIUM RISK** (Score 0.5-0.75)
  - Yellow warning banner
  - Recommendation: Require additional verification
  
- ℹ️ **LOW RISK** (Score 0.25-0.5)
  - Blue info banner
  - Recommendation: Approve with monitoring
  
- ✅ **VERY LOW RISK** (Score < 0.25)
  - Green success banner
  - Recommendation: Approve

#### Confidence Level
- Percentage showing model confidence
- Formula: `|score - 0.5| × 200%`
- Higher confidence = clearer classification

### Bottom Section: Transaction Analysis

#### Feature Analysis Chart
- Bar chart showing transaction feature values
- Visual comparison of:
  - Amount
  - Time (Hour)
  - Distance (km)

#### Risk Factors Panel
Automatically detected risk factors:
- 💰 **High transaction amount** (>$1000)
- 🌙 **Unusual time of day** (<6 AM or >10 PM)
- 📍 **Far from home location** (>100 km)
- 🏪 **High-risk merchant category**

### Sidebar: Model Information

#### Performance Metrics
- **Test Accuracy**: Model accuracy on test set
- **Qubits**: Number of qubits in circuit
- **Layers**: Depth of quantum circuit
- **Training Samples**: Size of training dataset

#### Training Details (Expandable)
- Final Validation Accuracy
- Final Validation F1 Score
- Number of Epochs
- Learning Rate

#### How It Works Section
Explains the quantum ML workflow:
1. Quantum Encoding
2. Variational Circuit
3. Measurement
4. Classification

### Example Transaction Buttons

Four quick-test buttons at the bottom:
1. **✅ Legitimate Example**
   - Small local grocery purchase
   - Expected: Low fraud score

2. **🚨 Fraud Example 1**
   - Large late-night electronics purchase
   - Expected: High fraud score

3. **🚨 Fraud Example 2**
   - Expensive jewelry far from home
   - Expected: High fraud score

4. **✅ Legitimate Example 2**
   - Regular restaurant visit
   - Expected: Low fraud score

## 💡 Usage Tips

### Testing Legitimate Transactions
Try these combinations:
- **Small amount** ($20-$100)
- **Normal hours** (9 AM - 9 PM)
- **Close to home** (<50 km)
- **Low-risk category** (grocery, restaurant)

### Testing Fraudulent Transactions
Try these combinations:
- **Large amount** ($1000+)
- **Unusual hours** (2-5 AM)
- **Far from home** (200+ km)
- **High-risk category** (electronics, jewelry, online)

### Understanding Results

#### High Confidence + High Score
- Clear fraud signal
- Multiple risk factors detected
- Strong recommendation to block

#### High Confidence + Low Score
- Clear legitimate signal
- No major risk factors
- Safe to approve

#### Low Confidence (score near 0.5)
- Ambiguous transaction
- Consider manual review
- May need more features for better classification

## 🎯 Real-World Scenarios

### Scenario 1: Late Night Online Shopping
**Input:**
- Amount: $2500
- Time: 3 AM
- Distance: 450 km
- Category: electronics

**Expected Result:**
- Fraud Score: ~0.75-0.85
- Risk: HIGH
- Recommendation: Block + Contact cardholder

### Scenario 2: Morning Coffee
**Input:**
- Amount: $4.50
- Time: 8 AM
- Distance: 2 km
- Category: restaurant

**Expected Result:**
- Fraud Score: ~0.15-0.25
- Risk: VERY LOW
- Recommendation: Approve

### Scenario 3: Weekend Shopping
**Input:**
- Amount: $150
- Time: 2 PM
- Distance: 15 km
- Category: retail

**Expected Result:**
- Fraud Score: ~0.30-0.45
- Risk: LOW
- Recommendation: Approve with monitoring

## 🔧 Customization

### Changing the Threshold
Edit `app.py` line ~260:
```python
threshold = 0.5  # Change to 0.6 for stricter fraud detection
```

### Adding More Risk Factors
Edit the risk factor detection in `app.py`:
```python
if amount > 500:  # Lower threshold
    risk_factors.append("💰 Moderate-high amount")
```

### Styling
Modify Streamlit theme in `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
```

## 🐛 Troubleshooting

### Web interface won't load
1. Check if model is trained: `ls saved_model/`
2. Verify Streamlit is installed: `pip list | grep streamlit`
3. Try different port: `streamlit run app.py --server.port 8502`

### "Model not found" error
Run training first:
```bash
python train_local.py
```

### Predictions seem random
- Model may need more training epochs
- Try re-training with different hyperparameters
- Check if preprocessing is working correctly

### Slow predictions
- Normal for quantum simulator on CPU
- Each prediction takes ~100-200ms
- Consider using `lightning.qubit` device for faster simulation

## 🎓 Educational Use

### For Students
1. **Experiment** with different transaction patterns
2. **Observe** how features affect fraud scores
3. **Understand** quantum vs classical ML differences
4. **Modify** code to learn quantum computing concepts

### For Researchers
1. **Compare** with classical ML models
2. **Analyze** quantum circuit behavior
3. **Test** different quantum architectures
4. **Benchmark** performance metrics

### For Developers
1. **Integrate** with existing systems
2. **Extend** with new features
3. **Deploy** to production environments
4. **Scale** for real-world usage

## 📸 Screenshots Guide

When sharing or documenting:
1. **Main view**: Show gauge chart with prediction
2. **Risk factors**: Highlight detected warnings
3. **Model metrics**: Display performance stats
4. **Example comparison**: Show legitimate vs fraud side-by-side

## 🚀 Advanced Features

### URL Parameters (Coming Soon)
Direct links with pre-filled values:
```
http://localhost:8501/?amount=2500&time=3&distance=500&category=electronics
```

### Batch Processing (Coming Soon)
Upload CSV file for multiple predictions

### Model Comparison (Coming Soon)
Compare different trained models side-by-side

---

**Enjoy exploring quantum machine learning through this interactive interface! 🔮**
