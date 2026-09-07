# 🎮 Command Cheat Sheet

Quick reference for all project commands.

---

## 📦 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

---

## 🔧 Training

```bash
# Train model with default settings (4 qubits, 2 layers, 30 epochs)
python train_local.py

# Expected time: ~30 seconds
# Output: Model saved to ./saved_model/
```

---

## 🔍 Inference

```bash
# Run command-line inference with examples
python inference_local.py

# Shows predictions for 4 sample transactions
```

---

## 🌐 Web Interface

```bash
# Launch Streamlit web app
streamlit run app.py

# Then open: http://localhost:8501
```

### Alternative Ports

```bash
# Use different port if 8501 is busy
streamlit run app.py --server.port 8502
streamlit run app.py --server.port 8503
```

### Development Mode

```bash
# Auto-reload on file changes
streamlit run app.py --server.runOnSave true
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_training.py -v
pytest tests/test_quantum_model.py -v
pytest tests/test_preprocessing.py -v

# Run with coverage
pytest tests/ --cov=src/quantum_fraud_detector --cov-report=html

# View coverage report
start htmlcov/index.html  # Windows
open htmlcov/index.html   # Mac
xdg-open htmlcov/index.html  # Linux

# Run single test
pytest tests/test_training.py::TestComputeMetrics::test_perfect_predictions -v
```

---

## 📊 Project Management

```bash
# Check project structure
tree . /F  # Windows
tree       # Mac/Linux

# Count lines of code
find . -name "*.py" | xargs wc -l  # Mac/Linux
```

---

## 🐍 Python Interactive

```python
# Start Python shell
python

# Quick test
from src.quantum_fraud_detector.utils.serialization import load_model
model, preprocessor, metadata = load_model("./saved_model")
print(f"Model has {model.n_qubits} qubits")
```

---

## 🔬 Jupyter Notebook (Optional)

```bash
# Install Jupyter
pip install jupyter

# Create notebook
jupyter notebook

# In notebook:
from src.quantum_fraud_detector.quantum_model.vqc import VariationalQuantumClassifier
model = VariationalQuantumClassifier(n_qubits=4, n_layers=2)
model.initialize_params()
# ... experiment
```

---

## 🛠️ Development

```bash
# Format code (if using black)
black src/ tests/

# Lint code (if using flake8)
flake8 src/ tests/

# Type checking (if using mypy)
mypy src/
```

---

## 📝 Configuration

```bash
# Copy example config
cp config.example.yaml config.yaml

# Edit config
notepad config.yaml  # Windows
nano config.yaml     # Linux
vim config.yaml      # Power users
```

---

## 🔍 Debugging

```bash
# Run with Python debugger
python -m pdb train_local.py

# Verbose pytest output
pytest tests/ -vv -s

# Show print statements
pytest tests/ -s

# Stop on first failure
pytest tests/ -x
```

---

## 📦 Model Management

```bash
# Check saved model
ls saved_model/  # Mac/Linux
dir saved_model  # Windows

# Should contain:
# - model_params.npy
# - model_config.json
# - preprocessor.pkl
# - metadata.json

# Backup model
cp -r saved_model saved_model_backup  # Mac/Linux
xcopy saved_model saved_model_backup /E /I  # Windows

# Delete model (to retrain)
rm -rf saved_model  # Mac/Linux
rmdir /s saved_model  # Windows
```

---

## 🌐 Port Management

```bash
# Check if port is in use
netstat -ano | findstr :8501  # Windows
lsof -i :8501  # Mac/Linux

# Kill process on port
taskkill /PID <PID> /F  # Windows
kill -9 <PID>  # Mac/Linux
```

---

## 📊 Performance Profiling

```python
# Time training
python -m cProfile -o profile.stats train_local.py

# Analyze profile
python -m pstats profile.stats
# Then: sort cumulative, stats 20
```

---

## 🎯 Quick Workflows

### Complete Workflow (First Time)
```bash
pip install -r requirements.txt
pip install -e .
python train_local.py
streamlit run app.py
```

### Daily Development
```bash
# Edit code
# Run tests
pytest tests/ -v

# Train
python train_local.py

# Launch UI
streamlit run app.py
```

### Demo Workflow
```bash
# Start fresh
rm -rf saved_model

# Train
python train_local.py

# Show CLI inference
python inference_local.py

# Show web UI
streamlit run app.py
```

---

## 🔧 Troubleshooting Commands

### Check Installations
```bash
pip list | grep pennylane
pip list | grep streamlit
python --version
```

### Verify Project Setup
```bash
python -c "import src.quantum_fraud_detector; print('OK')"
```

### Check Model
```python
python -c "from src.quantum_fraud_detector.utils.serialization import load_model; m,p,d = load_model('./saved_model'); print('Model loaded!')"
```

### Test Streamlit
```bash
streamlit hello  # Run Streamlit demo
```

---

## 📚 Documentation

```bash
# View README
cat README.md  # Mac/Linux
type README.md  # Windows

# View in browser (if markdown viewer installed)
grip README.md  # Requires: pip install grip
```

---

## 🎨 Customization

### Change Model Parameters
Edit `train_local.py`:
```python
N_QUBITS = 6  # Instead of 4
N_LAYERS = 3  # Instead of 2
EPOCHS = 50   # Instead of 30
```

### Change Web UI Theme
Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
```

---

## 🚀 Production Commands (Future)

```bash
# Build Docker image (when Dockerfile added)
docker build -t quantum-fraud-detector .

# Run in container
docker run -p 8501:8501 quantum-fraud-detector

# Deploy to cloud (when deployment added)
# AWS, GCP, or Azure specific commands
```

---

## 📱 Shortcuts

| Command | Description |
|---------|-------------|
| `python train_local.py` | Train model |
| `python inference_local.py` | Test inference |
| `streamlit run app.py` | Launch web UI |
| `pytest tests/ -v` | Run all tests |
| `pip install -e .` | Install package |

---

## 💡 Pro Tips

1. **Always activate virtual environment first**
   ```bash
   source venv/bin/activate  # Mac/Linux
   venv\Scripts\activate     # Windows
   ```

2. **Use tmux/screen for long training**
   ```bash
   tmux new -s training
   python train_local.py
   # Ctrl+B, then D to detach
   ```

3. **Monitor GPU usage (if using)**
   ```bash
   nvidia-smi -l 1  # Update every second
   ```

4. **Save outputs to file**
   ```bash
   python train_local.py > training_log.txt 2>&1
   ```

---

**Remember**: Check `QUICKSTART.md` for step-by-step guide!
