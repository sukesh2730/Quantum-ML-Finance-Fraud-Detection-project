"""Streamlit Web Interface for Quantum ML Fraud Detector.

This interactive web app allows users to input transaction features
and get real-time fraud predictions from the quantum classifier.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

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


def create_gauge_chart(score, threshold=0.5):
    """Create a gauge chart for fraud score visualization."""
    
    # Determine color based on score
    if score < threshold:
        color = "green"
        status = "LEGITIMATE"
    else:
        color = "red"
        status = "FRAUD"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Fraud Score<br><span style='font-size:24px'>{status}</span>", 
               'font': {'size': 20}},
        delta={'reference': threshold, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, threshold], 'color': 'lightgreen'},
                {'range': [threshold, 1], 'color': 'lightcoral'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=80, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "black", 'family': "Arial"}
    )
    
    return fig


def create_feature_importance_chart(features_dict):
    """Create a bar chart showing transaction features."""
    
    df = pd.DataFrame({
        'Feature': list(features_dict.keys()),
        'Value': list(features_dict.values())
    })
    
    fig = px.bar(
        df, 
        x='Value', 
        y='Feature', 
        orientation='h',
        title='Transaction Features',
        color='Value',
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    
    return fig


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
                    
                    # Display gauge chart
                    gauge_fig = create_gauge_chart(fraud_score)
                    st.plotly_chart(gauge_fig, use_container_width=True)
                    
                    # Risk assessment
                    if fraud_score >= 0.75:
                        st.error("🚨 **HIGH RISK** - Transaction likely fraudulent")
                        recommendation = "**Recommendation:** Block transaction and contact cardholder immediately."
                    elif fraud_score >= 0.5:
                        st.warning("⚠️ **MEDIUM RISK** - Transaction may be fraudulent")
                        recommendation = "**Recommendation:** Require additional verification (e.g., OTP, security questions)."
                    elif fraud_score >= 0.25:
                        st.info("ℹ️ **LOW RISK** - Transaction appears mostly legitimate")
                        recommendation = "**Recommendation:** Approve with standard monitoring."
                    else:
                        st.success("✅ **VERY LOW RISK** - Transaction appears legitimate")
                        recommendation = "**Recommendation:** Approve transaction."
                    
                    st.markdown(recommendation)
                    
                    # Confidence score
                    confidence = abs(fraud_score - 0.5) * 200
                    st.metric("Confidence Level", f"{confidence:.1f}%")
                    
                except Exception as e:
                    st.error(f"Error during prediction: {e}")
        else:
            st.info("👈 Enter transaction details and click 'Analyze Transaction' to get started")
    
    # Transaction features visualization
    if submit:
        st.markdown("---")
        st.header("📊 Transaction Feature Analysis")
        
        col3, col4 = st.columns(2)
        
        with col3:
            # Feature values
            features_dict = {
                'Amount': amount,
                'Time (Hour)': time_of_day,
                'Distance (km)': distance
            }
            
            fig = create_feature_importance_chart(features_dict)
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            st.subheader("Risk Factors")
            
            # Analyze risk factors
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
    
    # Example transactions
    st.markdown("---")
    st.header("📝 Example Transactions")
    
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        if st.button("✅ Legitimate Example", use_container_width=True):
            st.session_state.example = {
                'amount': 45.0,
                'time_of_day': 14,
                'distance': 5.0,
                'category': 'grocery'
            }
            st.rerun()
    
    with col6:
        if st.button("🚨 Fraud Example 1", use_container_width=True):
            st.session_state.example = {
                'amount': 2500.0,
                'time_of_day': 3,
                'distance': 500.0,
                'category': 'electronics'
            }
            st.rerun()
    
    with col7:
        if st.button("🚨 Fraud Example 2", use_container_width=True):
            st.session_state.example = {
                'amount': 1800.0,
                'time_of_day': 22,
                'distance': 300.0,
                'category': 'jewelry'
            }
            st.rerun()
    
    with col8:
        if st.button("✅ Legitimate Example 2", use_container_width=True):
            st.session_state.example = {
                'amount': 75.0,
                'time_of_day': 19,
                'distance': 8.0,
                'category': 'restaurant'
            }
            st.rerun()
    
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
