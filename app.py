"""Streamlit Web Interface for Quantum ML Fraud Detector.

Trained on real OpenML Credit Card Fraud dataset (ID 1597).
Features: V17, V14, V12, V10 (top PCA components by correlation with fraud).

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import json
import plotly.graph_objects as go
import joblib

# ── Page config ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Quantum Fraud Detector",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    code, .mono { font-family: 'JetBrains Mono', monospace; }
    
    .metric-card {
        background: #0f1117;
        border: 1px solid #1e2530;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 600; color: #00d4ff; }
    .metric-label { font-size: 0.8rem; color: #6b7280; margin-top: 4px; letter-spacing: 0.05em; }
    
    .fraud-high   { background: #2d0a0a; border: 1px solid #7f1d1d; border-radius: 8px; padding: 16px; }
    .fraud-medium { background: #1c1a07; border: 1px solid #78350f; border-radius: 8px; padding: 16px; }
    .fraud-low    { background: #0a1a0a; border: 1px solid #14532d; border-radius: 8px; padding: 16px; }
    
    .feature-note {
        background: #0f1a2e;
        border-left: 3px solid #00d4ff;
        padding: 12px 16px;
        border-radius: 0 6px 6px 0;
        font-size: 0.85rem;
        color: #9ca3af;
        margin-bottom: 8px;
    }
    .stSlider > div > div > div { background: #00d4ff !important; }
</style>
""", unsafe_allow_html=True)

FEATURES = ['V17', 'V14', 'V12', 'V10']
THRESHOLD = 0.50
MODEL_DIR = "./saved_model"

FEATURE_INFO = {
    'V17': {
        'desc': 'Transaction pattern component 1',
        'detail': 'Strongly negative in fraud. Values below −2 are high risk.',
        'min': -10.0, 'max': 5.0, 'default': 0.0, 'step': 0.1
    },
    'V14': {
        'desc': 'Merchant behaviour component',
        'detail': 'Most predictive single feature. Fraud clusters below −3.',
        'min': -10.0, 'max': 5.0, 'default': 0.0, 'step': 0.1
    },
    'V12': {
        'desc': 'Cardholder activity component',
        'detail': 'Negative values correlate with unusual spending patterns.',
        'min': -10.0, 'max': 5.0, 'default': 0.0, 'step': 0.1
    },
    'V10': {
        'desc': 'Location & time component',
        'detail': 'Captures geographic and temporal anomalies.',
        'min': -10.0, 'max': 5.0, 'default': 0.0, 'step': 0.1
    },
}

@st.cache_resource
def load_model_artifacts():
    model_path = Path(MODEL_DIR)
    if not model_path.exists():
        return None, None, None
    
    try:
        import sys
        sys.path.append(".")
        from src.quantum_fraud_detector.quantum_model.vqc import VariationalQuantumClassifier
        import pennylane.numpy as pnp
        
        with open(model_path / "model_config.json") as f:
            config = json.load(f)
        
        model = VariationalQuantumClassifier(
            n_qubits=config["n_qubits"],
            n_layers=config["n_layers"]
        )
        params = np.load(str(model_path / "model_params.npy"))
        model.weights = pnp.array(params, requires_grad=False)
        
        scaler = joblib.load(model_path / "scaler.pkl")
        
        with open(model_path / "metadata.json") as f:
            metadata = json.load(f)
        
        return model, scaler, metadata
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None


def predict_fraud(model, scaler, feature_values: dict) -> float:
    X = np.array([[feature_values[f] for f in FEATURES]])
    X_scaled = scaler.transform(X)
    raw = float(model.qnode(X_scaled[0], model.weights))
    return (raw + 1) / 2


def gauge_chart(score: float, threshold: float = THRESHOLD):
    is_fraud = score >= threshold
    color = "#ef4444" if is_fraud else "#22c55e"
    label = "FRAUD DETECTED" if is_fraud else "LEGITIMATE"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(score * 100, 1),
        number={'suffix': '%', 'font': {'size': 36, 'color': color}},
        title={'text': label, 'font': {'size': 18, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#374151'},
            'bar': {'color': color, 'thickness': 0.25},
            'bgcolor': '#1f2937',
            'borderwidth': 0,
            'steps': [
                {'range': [0, 50],  'color': '#0f2318'},
                {'range': [50, 75], 'color': '#1c1a07'},
                {'range': [75, 100],'color': '#2d0a0a'},
            ],
            'threshold': {
                'line': {'color': '#ffffff', 'width': 2},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e5e7eb"
    )
    return fig


model, scaler, metadata = load_model_artifacts()

st.markdown("## ⚛️ Quantum Fraud Detector")
st.markdown(
    "A Variational Quantum Classifier trained on **real credit card transactions** "
    "from the OpenML Credit Card Fraud dataset (284,807 transactions, 492 confirmed fraud cases)."
)
st.divider()

if model is None:
    st.error("Model not found. Ensure `saved_model/` exists with all 4 files.")
    st.stop()

with st.sidebar:
    st.markdown("### Model Performance")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">0.78</div>
            <div class="metric-label">F1 Score</div>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">93%</div>
            <div class="metric-label">Precision</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">67%</div>
            <div class="metric-label">Recall</div>
        </div>
        """, unsafe_allow_html=True)
    with col_d:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">81%</div>
            <div class="metric-label">Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    st.markdown("### Architecture")
    st.markdown("""
    - **Qubits:** 4
      - **Layers:** 2 variational
      - **Parameters:** 24 trainable
      - **Encoding:** RY angle encoding
      - **Entanglement:** CNOT ring topology
      - **Optimizer:** Adam (lr=0.05)
      - **Epochs:** 50
      - **Framework:** PennyLane
    """)
    
    st.divider()
    st.markdown("### Training Data")
    st.markdown("""
    - **Source:** OpenML ID 1597
      - **Balanced sample:** 984 transactions
      - **Fraud:** 492 · **Legit:** 492
      - **Features:** Top 4 PCA components
        by correlation with fraud label
    """)

col_input, col_result = st.columns([1, 1], gap="large")

with col_input:
    st.markdown("### Transaction Features")
    st.markdown(
        "These are PCA-transformed components from the original transaction data. "
        "Legitimate transactions cluster near 0; fraud tends toward negative values."
    )
    
    feature_values = {}
    for feat in FEATURES:
        info = FEATURE_INFO[feat]
        st.markdown(f"""
        <div class="feature-note">
            <strong>{feat}</strong> — {info['desc']}<br>{info['detail']}
        </div>
        """, unsafe_allow_html=True)
        feature_values[feat] = st.slider(
            f"{feat}",
            min_value=info['min'],
            max_value=info['max'],
            value=info['default'],
            step=info['step'],
            label_visibility="collapsed"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        if st.button("✅ Legit example", use_container_width=True):
            st.session_state.preset = "legit"
            st.rerun()
    with col_btn2:
        if st.button("🚨 Fraud example 1", use_container_width=True):
            st.session_state.preset = "fraud1"
            st.rerun()
    with col_btn3:
        if st.button("🚨 Fraud example 2", use_container_width=True):
            st.session_state.preset = "fraud2"
            st.rerun()
    
    presets = {
        "legit":  {'V17': 0.2,  'V14': 0.3,  'V12': 0.1,  'V10': 0.2},
        "fraud1": {'V17': -4.5, 'V14': -6.2, 'V12': -3.8, 'V10': -4.1},
        "fraud2": {'V17': -2.8, 'V14': -4.1, 'V12': -5.2, 'V10': -3.3},
    }
    if "preset" in st.session_state:
        feature_values = presets[st.session_state.preset]
    
    analyze = st.button("⚛️ Analyze Transaction", use_container_width=True, type="primary")

with col_result:
    st.markdown("### Risk Assessment")
    
    if analyze or "preset" in st.session_state:
        with st.spinner("Running quantum circuit..."):
            try:
                score = predict_fraud(model, scaler, feature_values)
                is_fraud = score >= THRESHOLD
                
                st.plotly_chart(gauge_chart(score), use_container_width=True)
                
                if score >= 0.75:
                    st.markdown("""
                    <div class="fraud-high">
                        🚨 <strong>High risk</strong> — Block transaction and contact cardholder immediately.
                    </div>
                    """, unsafe_allow_html=True)
                elif score >= THRESHOLD:
                    st.markdown("""
                    <div class="fraud-medium">
                        ⚠️ <strong>Medium risk</strong> — Require additional verification before approving.
                    </div>
                    """, unsafe_allow_html=True)
                elif score >= 0.35:
                    st.markdown("""
                    <div class="fraud-low">
                        ℹ️ <strong>Low risk</strong> — Approve with standard monitoring.
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="fraud-low">
                        ✅ <strong>Very low risk</strong> — Transaction appears legitimate.
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                st.markdown("**Feature values submitted:**")
                df_display = pd.DataFrame({
                    'Feature': list(feature_values.keys()),
                    'Value': [f"{v:.2f}" for v in feature_values.values()],
                    'Risk direction': [
                        '🔴 elevated' if feature_values[f] < -2 else '🟢 normal'
                        for f in feature_values
                    ]
                })
                st.dataframe(df_display, hide_index=True, use_container_width=True)
                
                if "preset" in st.session_state:
                    del st.session_state.preset
            
            except Exception as e:
                st.error(f"Prediction error: {e}")
    else:
        st.markdown("""
        <div style="height:280px; display:flex; align-items:center; justify-content:center;
                    color:#4b5563; border: 1px dashed #1f2937; border-radius:8px;">
            Adjust the feature sliders and click Analyze Transaction
        </div>
        """, unsafe_allow_html=True)

st.divider()
with st.expander("How the quantum classifier works"):
    st.markdown("""
    **Data pipeline**
    
    1. Raw transaction data → PCA transformation (done by the dataset provider)
      2. Top 4 PCA components selected by correlation with fraud label: V17, V14, V12, V10
      3. Features scaled to [0.1, 3.04] range for quantum angle encoding
      4. Each feature encoded into a qubit via RY rotation gate
      5. Two variational layers of Rot gates (RZ→RY→RZ) + CNOT ring entanglement
      6. Pauli-Z expectation value on qubit 0, mapped to fraud score [0, 1]
    
    **Why 93% precision matters**
      When the model flags a transaction as fraud, it is correct 93% of the time.
      This minimises false alarms — legitimate customers are rarely blocked.
    
    **Why 67% recall is the tradeoff**
      The model misses about 1 in 3 fraud cases. This is a known limitation of
      4-qubit VQCs on complex real-world data. Increasing qubits and training
      samples would improve recall further.
    
    **Confusion matrix on test set (197 transactions)**
    
    | | Predicted Legit | Predicted Fraud |
    |---|---|---|
    | **Actually Legit** | 94 ✅ | 5 ❌ |
    | **Actually Fraud** | 32 ❌ | 66 ✅ |
    """)

st.markdown("""
<div style="text-align:center; color:#374151; font-size:0.8rem; margin-top:2rem;">
    ⚛️ Quantum ML Fraud Detector · PennyLane + Streamlit · OpenML Credit Card Fraud Dataset
</div>
""", unsafe_allow_html=True)
