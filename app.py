import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Streamlit UI Setup
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")
st.title("🚢 Titanic Survival Prediction App")
st.markdown("Enter passenger details to predict if they would have survived the Titanic disaster.")

# Input fields
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.radio("Sex", ["male", "female"])
age = st.slider("Age", 0.42, 80.0, 30.0)
sibsp = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, value=32.20)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Encoding inputs
sex_encoded = 1 if sex == "male" else 0
embarked_C = 1 if embarked == "C" else 0
embarked_Q = 1 if embarked == "Q" else 0
embarked_S = 1 if embarked == "S" else 0

# Create input DataFrame
input_df = pd.DataFrame([{
    "Pclass": pclass,
    "Sex": sex_encoded,
    "Age": age,
    "SibSp": sibsp,
    "Parch": parch,
    "Fare": fare,
    "Embarked_C": embarked_C,
    "Embarked_Q": embarked_Q,
    "Embarked_S": embarked_S
}])

# Load scaler used during training
scaler_path = "saved_models/scaler.pkl"
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# Scale the input
input_scaled = scaler.transform(input_df)

# Load all saved models
models = {}
model_dir = "saved_models"
for filename in os.listdir(model_dir):
    if filename.endswith("_model.pkl"):
        path = os.path.join(model_dir, filename)
        with open(path, "rb") as f:
            model_data = pickle.load(f)
            models[model_data["name"]] = model_data["best_estimator"]

# Prediction and display
if st.button("Predict Survival"):
    st.subheader("🧠 Predictions from Multiple Models")

    left_col, right_col = st.columns([2, 1])
    probabilities = []
    labels = []

    with left_col:
        for model_name, model in models.items():
            try:
                prediction = model.predict(input_scaled)[0]

                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(input_scaled)[0][1]
                    probabilities.append(prob)
                    labels.append(model_name)

                    if prediction == 1:
                        st.success(f"✅ {model_name}: Survived (Probability: {prob:.2%})")
                    else:
                        st.error(f"❌ {model_name}: Did Not Survive (Probability: {prob:.2%})")
                else:
                    if prediction == 1:
                        st.success(f"✅ {model_name}: Survived (No probability available)")
                    else:
                        st.error(f"❌ {model_name}: Did Not Survive (No probability available)")

            except Exception as e:
                st.warning(f"{model_name} failed to predict: {e}")

    with right_col:
        if probabilities:
            mean_prob = np.mean(probabilities)
            right_col.markdown(
                f"""
                <div style='text-align: center; font-size: 28px; font-weight: bold; color: #2c3e50;'>
                    🔢 Overall Mean<br><span style='font-size: 48px;'>{mean_prob:.2%}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

            # 📊 Big bar plot
            fig, ax = plt.subplots(figsize=(10, 6))  # Bigger figure size
            bars = ax.barh(labels, probabilities, color='cornflowerblue', edgecolor='black')

            ax.set_xlim(0, 1)
            ax.set_xlabel("Survival Probability", fontsize=14)
            ax.set_title("🔍 Survival Probabilities by Model", fontsize=18, weight='bold')

            # Add % labels next to each bar
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.02, bar.get_y() + bar.get_height() / 2,
                        f"{width:.1%}", va='center', fontsize=13)

            ax.tick_params(axis='y', labelsize=12)
            ax.tick_params(axis='x', labelsize=12)
            st.pyplot(fig)

        else:
            right_col.info("No probability available to compute average.")
