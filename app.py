import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="GlucoShield AI | Early Diabetes Risk Assessment",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Trained Model and Scaler
@st.cache_resource
def load_ml_assets():
    model = joblib.load("diabetes_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

try:
    model, scaler = load_ml_assets()
except Exception:
    st.error("❌ Trained assets missing! Please run `python train_model.py` first.")
    st.stop()

# Application Header
st.title("🩸 GlucoShield AI")
st.caption("Clinical-Grade Machine Learning Diagnostic & Risk Analytics Platform")
st.markdown("---")

# Main Layout
tab1, tab2, tab3 = st.tabs(["🩺 Patient Assessment", "📁 Bulk CSV Screening", "📈 Clinical Metrics"])

# ================= TAB 1: INDIVIDUAL PATIENT ASSESSMENT =================
with tab1:
    st.subheader("Patient Clinical Data Input")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        glucose = st.number_input("Fasting Glucose Level (mg/dL)", min_value=50, max_value=300, value=110, help="Normal: < 100 mg/dL | Prediabetes: 100-125 mg/dL")
        blood_pressure = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=40, max_value=140, value=72)
        bmi = st.number_input("Body Mass Index (BMI)", min_value=10.0, max_value=60.0, value=24.5, format="%.1f")

    with col2:
        insulin = st.number_input("2-Hour Serum Insulin (mu U/ml)", min_value=15, max_value=840, value=80)
        skin_thickness = st.number_input("Triceps Skin Fold Thickness (mm)", min_value=7, max_value=99, value=20)
        pregnancies = st.number_input("Pregnancies Count", min_value=0, max_value=20, value=0)

    with col3:
        age = st.number_input("Patient Age (Years)", min_value=1, max_value=120, value=30)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.05, max_value=2.5, value=0.47, format="%.2f", help="Genetic risk score based on family history.")

    st.markdown("---")

    if st.button("🔬 Analyze Risk Profile"):
        features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
        scaled_features = scaler.transform(features)
        
        prediction = model.predict(scaled_features)[0]
        probability = model.predict_proba(scaled_features)[0][1] * 100

        st.markdown("### 📊 Assessment Outcome")
        r1, r2, r3 = st.columns(3)

        with r1:
            if prediction == 1:
                st.metric("Predicted Condition", "HIGH RISK (Diabetic)", delta="Positive Threat", delta_color="inverse")
            else:
                st.metric("Predicted Condition", "LOW RISK (Non-Diabetic)", delta="Healthy", delta_color="normal")

        with r2:
            st.metric("Diabetic Risk Probability", f"{probability:.1f}%")

        with r3:
            # Health Risk Level Scale
            if probability < 30:
                risk_level = "🟢 Low"
            elif probability < 65:
                risk_level = "🟡 Moderate"
            else:
                risk_level = "🔴 High"
            st.metric("Risk Category", risk_level)

        st.progress(int(probability))

        # Recommendations based on output
        if prediction == 1:
            st.error("⚠️ **Clinical Attention Recommended:** The parameters indicate high likelihood of diabetes. Consultation with an endocrinologist for HbA1c testing is advised.")
        else:
            st.success("✅ **Healthy Indicator:** Patient parameters fall within low-risk limits. Maintain a balanced diet and regular physical activity.")

# ================= TAB 2: BATCH SCREENING =================
with tab2:
    st.subheader("Batch Patient Screening via CSV")
    uploaded_file = st.file_uploader("Upload CSV containing patient metrics", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Preview Uploaded Data:", batch_df.head(3))

            required_cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
            
            if not all(col in batch_df.columns for col in required_cols):
                st.error(f"CSV must contain all 8 columns: {', '.join(required_cols)}")
            else:
                scaled_batch = scaler.transform(batch_df[required_cols])
                batch_df["Risk Prediction"] = model.predict(scaled_batch)
                batch_df["Risk Probability (%)"] = (model.predict_proba(scaled_batch)[:, 1] * 100).round(1)
                batch_df["Risk Prediction"] = batch_df["Risk Prediction"].map({0: "Low Risk", 1: "High Risk"})

                st.write("### 📋 Screened Patients Summary")
                st.dataframe(batch_df, use_container_width=True)
        except Exception as e:
            st.error(f"Error processing file: {e}")

# ================= TAB 3: CLINICAL METRICS =================
with tab3:
    st.subheader("Model Validation & Training Metadata")
    c1, c2 = st.columns(2)
    with c1:
        st.json({
            "Dataset": "PIMA Indians Diabetes Database",
            "Algorithm": "Random Forest Classifier",
            "Preprocessing": "Imputation + StandardScaler",
            "Target Metrics": "Accuracy ~80-85% | ROC-AUC ~0.88",
        })
    with c2:
        st.markdown("""
        * **Glucose & BMI**: Identified as top two decisive feature vectors.
        * **Zero-Value Treatment**: Imputed physiological zeros (e.g., Blood Pressure) with group medians to eliminate skew.
        """)

# Sidebar
st.sidebar.title("📌 Health Dashboard")
st.sidebar.info("Developed by **Himanshu Singh**\n\n*Machine Learning Medical Decision Support Project*")