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

# Load scaler and models with version handling
try:
    # Load scaler
    scaler_path = "saved_models/scaler.pkl"
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    # Scale the input
    input_scaled = scaler.transform(input_df)

    # Load models with version check
    models = {}
    model_dir = "saved_models"
    
    for filename in os.listdir(model_dir):
        if filename.endswith("_model.pkl"):
            path = os.path.join(model_dir, filename)
            with open(path, "rb") as f:
                model_data = pickle.load(f)
                
                # Version compatibility check
                if hasattr(model_data["best_estimator"], '__sklearn_version__'):
                    model_version = model_data["best_estimator"].__sklearn_version__
                    current_version = sklearn.__version__
                    if model_version != current_version:
                        st.warning(f"Model {model_data['name']} was trained with scikit-learn {model_version} (current: {current_version})")
                
                models[model_data["name"]] = model_data["best_estimator"]

except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# Prediction and display
if st.button("Predict Survival"):
    st.subheader("🧠 Predictions from Multiple Models")

    left_col, right_col = st.columns([2, 1])
    probabilities = []
    labels = []

    with left_col:
        for model_name, model in models.items():
            try:
                # Get prediction
                prediction = model.predict(input_scaled)[0]
                
                # Get probabilities if available
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(input_scaled)[0][1]
                    probabilities.append(prob)
                    labels.append(model_name)
                    result = f"Probability: {prob:.2%}"
                else:
                    result = "No probability available"
                    prob = None

                # Display results
                if prediction == 1:
                    st.success(f"✅ {model_name}: Survived ({result})")
                else:
                    st.error(f"❌ {model_name}: Did Not Survive ({result})")

            except Exception as e:
                st.warning(f"⚠️ {model_name} failed: {str(e)}")

    with right_col:
        if probabilities:
            mean_prob = np.mean(probabilities)
            st.metric("Average Survival Chance", f"{mean_prob:.2%}")
            
            # Visualization
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(labels, probabilities, color='#1f77b4')
            ax.set_xlim(0, 1)
            ax.set_xlabel("Survival Probability")
            ax.set_title("Model Confidence Levels")
            st.pyplot(fig)
        else:
            st.info("No probabilistic models available for visualization")