"""Simplified Streamlit Web Interface for Quantum ML Fraud Detector (No Plotly).

This interactive web app allows users to input transaction features
and get real-time fraud predictions from the quantum classifier.

Run with: streamlit run app_simple.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from src.quantum_fraud_detector.utils.serialization import load_model


# Page configuration
st.set_page_config(
    page_title="Quantum ML Fraud Detector",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_trained_model():
    """Load the trained model (cached for performance)."""
    MODEL_DIR = "./saved_model"
    if not Path(MODEL_DIR).exists():
        return None, None, None
    try:
        model, preprocessor, metadata = load_model(MODEL_DIR)
        return model, preprocessor, metadata
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None


def get_risk_level(score):
    """Determine risk level from fraud score."""
    if score >= 0.75:
        return "🚨 HIGH RISK", "red", "Block transaction and contact cardholder immediately."
    elif score >= 0.5:
        return "⚠️ MEDIUM RISK", "orange", "Require additional verification (e.g., OTP, security questions)."
    elif score >= 0.25:
        return "ℹ️ LOW RISK", "blue", "Approve with standard monitoring."
    else:
        return "✅ VERY LOW RISK", "green", "Approve transaction."


def main():
    """Main Streamlit app."""
    
    # Header
    st.title("🔮 Quantum ML Fraud Detector")
    st.markdown("""
    This application uses a **Variational Quantum Classifier (VQC)** to detect fraudulent 
    financial transactions. Enter transaction details below to get a real-time fraud risk assessment.
    """)
    
    # Load model
    model, preprocessor, metadata = load_trained_model()
    
    if model is None:
        st.error("⚠️ **Model not found!** Please run `python train_local.py` first to train the model.")
        st.info("After training, refresh this page to load the model.")
        st.stop()
    
    # Sidebar - Model Info
    with st.sidebar:
        st.header("📊 Model Information")
        
        if metadata:
            st.metric("Test Accuracy", f"{metadata.get('test_accuracy', 0):.1%}")
            st.metric("Qubits", metadata.get('n_qubits', 'N/A'))
            st.metric("Layers", metadata.get('n_layers', 'N/A'))
            st.metric("Training Samples", metadata.get('n_training_samples', 'N/A'))
            
            with st.expander("📈 Training Details"):
                st.write(f"**Final Validation Accuracy:** {metadata.get('final_val_accuracy', 0):.4f}")
                st.write(f"**Final Validation F1:** {metadata.get('final_val_f1', 0):.4f}")
                st.write(f"**Epochs:** {metadata.get('epochs', 'N/A')}")
                st.write(f"**Learning Rate:** {metadata.get('learning_rate', 'N/A')}")
        
        st.markdown("---")
        st.markdown("""
        ### 🔬 How It Works
        
        1. **Quantum Encoding**: Transaction features are encoded as quantum states
        2. **Variational Circuit**: Parameterized quantum gates process the data
        3. **Measurement**: Quantum state is measured to produce fraud score
        4. **Classification**: Score > 0.5 indicates potential fraud
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("💳 Transaction Details")
        
        # Input form
        with st.form("transaction_form"):
            amount = st.number_input(
                "Transaction Amount ($)",
                min_value=0.01,
                max_value=10000.0,
                value=150.0,
                step=10.0,
                help="The dollar amount of the transaction"
            )
            
            time_of_day = st.slider(
                "Time of Day (Hour)",
                min_value=0,
                max_value=23,
                value=14,
                help="Hour of day when transaction occurred (0-23)"
            )
            
            distance = st.number_input(
                "Distance from Home (km)",
                min_value=0.0,
                max_value=1000.0,
                value=10.0,
                step=5.0,
                help="Distance of transaction location from cardholder's home"
            )
            
            category = st.selectbox(
                "Merchant Category",
                options=['retail', 'grocery', 'gas', 'restaurant', 'online', 
                        'electronics', 'jewelry', 'travel'],
                help="Type of merchant where transaction occurred"
            )
            
            submit = st.form_submit_button("🔍 Analyze Transaction", use_container_width=True)
    
    with col2:
        st.header("🎯 Fraud Risk Assessment")
        
        if submit:
            # Create transaction DataFrame
            transaction = pd.DataFrame([{
                'amount': amount,
                'time_of_day': float(time_of_day),
                'distance_from_home': distance,
                'merchant_category': category
            }])
            
            # Show loading spinner
            with st.spinner("🔮 Running quantum inference..."):
                try:
                    # Preprocess
                    X = preprocessor.transform(transaction)
                    
                    # Predict
                    fraud_score = float(model.predict(X[0]))
                    
                    # Display results
                    risk_level, color, recommendation = get_risk_level(fraud_score)
                    
                    # Fraud score display
                    st.markdown(f"### Fraud Score: {fraud_score:.4f}")
                    
                    # Progress bar for score
                    st.progress(fraud_score)
                    
                    # Risk level
                    if color == "red":
                        st.error(risk_level)
                    elif color == "orange":
                        st.warning(risk_level)
                    elif color == "blue":
                        st.info(risk_level)
                    else:
                        st.success(risk_level)
                    
                    st.markdown(f"**Recommendation:** {recommendation}")
                    
                    # Confidence score
                    confidence = abs(fraud_score - 0.5) * 200
                    st.metric("Confidence Level", f"{confidence:.1f}%")
                    
                    # Risk factors
                    st.markdown("---")
                    st.subheader("Risk Factors")
                    
                    risk_factors = []
                    
                    if amount > 1000:
                        risk_factors.append("💰 High transaction amount")
                    if time_of_day < 6 or time_of_day > 22:
                        risk_factors.append("🌙 Unusual time of day")
                    if distance > 100:
                        risk_factors.append("📍 Far from home location")
                    if category in ['online', 'electronics', 'jewelry', 'travel']:
                        risk_factors.append("🏪 High-risk merchant category")
                    
                    if risk_factors:
                        st.warning("**Detected Risk Factors:**")
                        for factor in risk_factors:
                            st.write(f"- {factor}")
                    else:
                        st.success("**No major risk factors detected**")
                        st.write("- Normal transaction amount")
                        st.write("- Regular business hours")
                        st.write("- Close to home")
                        st.write("- Low-risk merchant")
                    
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
        else:
            st.info("👈 Enter transaction details and click 'Analyze Transaction' to get started")
    
    # Example transactions
    st.markdown("---")
    st.header("📝 Quick Test Examples")
    
    col3, col4, col5, col6 = st.columns(4)
    
    with col3:
        st.markdown("**✅ Legitimate**")
        st.write("Amount: $45")
        st.write("Time: 2 PM")
        st.write("Distance: 5 km")
        st.write("Category: grocery")
    
    with col4:
        st.markdown("**🚨 Fraud**")
        st.write("Amount: $2500")
        st.write("Time: 3 AM")
        st.write("Distance: 500 km")
        st.write("Category: electronics")
    
    with col5:
        st.markdown("**🚨 Fraud**")
        st.write("Amount: $1800")
        st.write("Time: 10 PM")
        st.write("Distance: 300 km")
        st.write("Category: jewelry")
    
    with col6:
        st.markdown("**✅ Legitimate**")
        st.write("Amount: $75")
        st.write("Time: 7 PM")
        st.write("Distance: 8 km")
        st.write("Category: restaurant")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>🔮 Quantum ML Fraud Detector | Built with PennyLane + Streamlit</p>
        <p>⚛️ Powered by Variational Quantum Circuits</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
