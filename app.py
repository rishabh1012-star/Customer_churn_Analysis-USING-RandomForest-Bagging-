import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="centered"
)

# ── Helpers ────────────────────────────────────────────────────────────────────
MODEL_PATH = "churn_model.joblib"
TRANSFORMER_PATH = "churn_transformer.joblib"


@st.cache_resource
def load_or_train_model(csv_path="Churn_Modelling.csv"):
    """Load saved model/transformer, or train fresh from CSV if not found."""
    if os.path.exists(MODEL_PATH) and os.path.exists(TRANSFORMER_PATH):
        model = joblib.load(MODEL_PATH)
        transformer = joblib.load(TRANSFORMER_PATH)
        return model, transformer

    if not os.path.exists(csv_path):
        return None, None

    df = pd.read_csv(csv_path)
    new_df = df[['CreditScore', 'Gender', 'IsActiveMember',
                 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                 'EstimatedSalary', 'Exited']]

    x_train, _, y_train, _ = train_test_split(
        new_df.drop('Exited', axis=1),
        new_df['Exited'],
        test_size=0.2,
        shuffle=False
    )

    transformer = ColumnTransformer(transformers=[
        ('tnfr1', StandardScaler(),
         ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']),
        ('tnfr2', OneHotEncoder(drop='first', sparse_output=False), ['Gender'])
    ], remainder='passthrough')

    x_train_transformed = transformer.fit_transform(x_train)
    features = transformer.get_feature_names_out()
    x_train_transformed = pd.DataFrame(x_train_transformed, columns=features)

    model = RandomForestClassifier(
        class_weight='balanced',
        bootstrap=True,
        criterion='gini',
        min_samples_split=5,
        n_estimators=250,
        random_state=42,
        max_features='log2',
        max_depth=12
    )
    model.fit(x_train_transformed, y_train)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(transformer, TRANSFORMER_PATH)
    return model, transformer


def predict_churn(model, transformer, input_dict):
    """Run prediction and return (label, probability)."""
    input_df = pd.DataFrame([input_dict])
    features = transformer.get_feature_names_out()
    input_transformed = pd.DataFrame(
        transformer.transform(input_df), columns=features
    )
    pred = model.predict(input_transformed)[0]
    prob = model.predict_proba(input_transformed)[0][1]
    return pred, prob


# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("🔮 Customer Churn Predictor")
st.markdown("Fill in the customer details below to predict whether they are likely to churn.")

# Model loading
model, transformer = load_or_train_model()

if model is None:
    st.warning(
        "⚠️ **Model not found.**\n\n"
        "Place `Churn_Modelling.csv` in the same folder as `app.py` and restart the app. "
        "The model will train automatically on first launch and be saved for future use."
    )
    st.stop()
else:
    st.success("✅ Model loaded and ready.", icon="✅")

st.divider()

# ── Input form ─────────────────────────────────────────────────────────────────
st.subheader("📋 Customer Information")

col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input(
        "Credit Score", min_value=300, max_value=850, value=650,
        help="Customer's credit score (300–850)"
    )
    age = st.number_input(
        "Age", min_value=18, max_value=100, value=35
    )
    tenure = st.slider(
        "Tenure (years with bank)", min_value=0, max_value=10, value=5
    )
    num_products = st.selectbox(
        "Number of Products", options=[1, 2, 3, 4], index=1
    )

with col2:
    gender = st.radio("Gender", options=["Male", "Female"], horizontal=True)
    balance = st.number_input(
        "Account Balance (₹ / $)", min_value=0.0, max_value=300000.0,
        value=75000.0, step=1000.0, format="%.2f"
    )
    estimated_salary = st.number_input(
        "Estimated Salary (₹ / $)", min_value=0.0, max_value=300000.0,
        value=60000.0, step=1000.0, format="%.2f"
    )
    is_active = st.toggle("Is Active Member?", value=True)

st.divider()

# ── Predict button ─────────────────────────────────────────────────────────────
if st.button("🔍 Predict Churn", use_container_width=True, type="primary"):
    input_data = {
        "CreditScore": credit_score,
        "Gender": gender,
        "IsActiveMember": int(is_active),
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "EstimatedSalary": estimated_salary,
    }

    pred, prob = predict_churn(model, transformer, input_data)

    st.subheader("📊 Prediction Result")

    if pred == 1:
        st.error(f"⚠️ **High Risk of Churn** — Probability: {prob:.1%}")
        st.markdown(
            "> This customer is **likely to leave**. Consider proactive retention offers."
        )
    else:
        st.success(f"✅ **Low Risk of Churn** — Probability: {prob:.1%}")
        st.markdown(
            "> This customer is **likely to stay**. Keep up the good service!"
        )

    # Probability bar
    st.markdown("**Churn Probability**")
    st.progress(float(prob))

    # Feature importance (quick view)
    with st.expander("🔎 Top Feature Importances"):
        features = transformer.get_feature_names_out()
        importance_series = pd.Series(
            model.feature_importances_, index=features
        ).sort_values(ascending=False).head(8)
        st.bar_chart(importance_series)

st.caption("Model: Random Forest Classifier · Built from Churn_Modelling.csv")