import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# 1. Page Configuration (Must be the very first Streamlit call)
st.set_page_config(
    page_title="Student Burnout Predictor",
    page_icon="🎓",
    layout="wide"
)

# 2. Cache Data Loading
@st.cache_data
def load_data():
    return pd.read_csv("student_mental_health_burnout.csv")

@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

df = load_data()
model = load_model()

# 3. Fit Encoders for Categorical Features & Target
# Exclude target and non-predictive identifiers
feature_columns = [col for col in df.columns if col not in ["burnout_level", "student_id"]]

encoders = {}
for col in feature_columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        le.fit(df[col])
        encoders[col] = le

# Target encoder to decode predictions back to readable labels
target_encoder = LabelEncoder()
target_encoder.fit(df["burnout_level"])


# 4. Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", [
    "Prediction",
    "Data Explorer",
    "Model Performance"
])


# ==========================================
# PAGE 1: PREDICTION
# ==========================================
if page == "Prediction":
    st.title("🎓 Student Burnout Prediction")
    st.markdown("Enter student metrics below to assess the predicted burnout risk level.")
    st.write("---")

    user_input = {}
    
    # Split input form into 3 neat columns
    col1, col2, col3 = st.columns(3)
    columns_list = list(feature_columns)
    
    for i, col in enumerate(columns_list):
        # Distribute fields across columns
        target_col = col1 if i % 3 == 0 else (col2 if i % 3 == 1 else col3)
        
        with target_col:
            field_label = col.replace("_", " ").title()
            
            if df[col].dtype == 'object':
                selected_val = st.selectbox(field_label, options=df[col].unique(), key=col)
                user_input[col] = encoders[col].transform([selected_val])[0]
            else:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                default_val = float(df[col].median())
                user_input[col] = st.number_input(
                    field_label,
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    key=col
                )

    st.write("")
    if st.button("Predict Burnout Level", type="primary", use_container_width=True):
        # Convert input dictionary into DataFrame with proper feature names
        input_df = pd.DataFrame([user_input])
        prediction_num = model.predict(input_df)[0]
        
        # Decode prediction back to string label (e.g., 'Low', 'Medium', 'High')
        prediction_label = target_encoder.inverse_transform([prediction_num])[0]
        
        # Display colored notification based on predicted level
        if prediction_label.lower() == "high":
            st.error(f"🔥 Predicted Burnout Level: **{prediction_label}**")
        elif prediction_label.lower() == "medium":
            st.warning(f"⚠️ Predicted Burnout Level: **{prediction_label}**")
        else:
            st.success(f"✅ Predicted Burnout Level: **{prediction_label}**")


# ==========================================
# PAGE 2: DATA EXPLORER
# ==========================================
elif page == "Data Explorer":
    st.title("📈 Data Explorer")

    st.subheader("Dataset Overview")
    st.dataframe(df.head(10), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Burnout Level Distribution")
        st.bar_chart(df["burnout_level"].value_counts())

    with col2:
        st.subheader("Numeric Summary Statistics")
        st.dataframe(df.describe().T, use_container_width=True)

    st.subheader("Correlation Heatmap (Numeric Features)")
    fig, ax = plt.subplots(figsize=(10, 6))
    numeric_df = df.select_dtypes(include=np.number).drop(columns=["student_id"], errors='ignore')
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)


# ==========================================
# PAGE 3: MODEL PERFORMANCE
# ==========================================
elif page == "Model Performance":
    st.title("🤖 Model Comparison & Evaluation")

    st.markdown("""
    Multiple models were evaluated on the dataset:
    - **Logistic Regression**
    - **Decision Tree**
    - **Random Forest** (Selected Best Model)
    """)

    # Performance metrics table
    metrics_data = pd.DataFrame({
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
        "Accuracy": [0.3315, 0.3323, 0.3325],
        "F1 Score (Weighted)": [0.3208, 0.3323, 0.3323]
    })

    st.table(metrics_data)
    st.success("🏆 Best Performing Model: **Random Forest Classifier**")