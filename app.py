import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('model.pkl')

hcc = pd.read_csv("hcc_dataset.csv", na_values=["?"])

# Streamlit app
st.title("HCC Predictor")

st.write("""
### Input Patient State
""")

# User inputs for the features
PVT = st.selectbox("PVT", ("Yes", "No"))
Total_Bil = st.slider("Total_Bil", float(hcc['Total_Bil'].min()), float(hcc['Total_Bil'].max()), float(hcc['Total_Bil'].mean()))
Metastasis = st.selectbox("Metastasis", ("Yes", "No"))
Ascites = st.selectbox("Ascites", ("None", "Mild", "Moderate/Severe"))
Albumin = st.slider("Albumin", float(hcc[' Albumin'].min()), float(hcc[' Albumin'].max()), float(hcc[' Albumin'].mean()))
Hemoglobin = st.slider("Hemoglobin", float(hcc['Hemoglobin'].min()), float(hcc['Hemoglobin'].max()), float(hcc['Hemoglobin'].mean()))
ALP = st.slider("ALP", float(hcc['ALP'].min()), float(hcc['ALP'].max()), float(hcc['ALP'].mean()))
Symptoms = st.selectbox("Symptons", ("Yes", "No"))
PS = st.selectbox("PS", ("Active", "Ambulatory", "Restricted", "Selfcare", "Disabled"))

Symptoms = 1 if Symptoms == 'Yes' else 0
Metastasis = 1 if Metastasis == 'Yes' else 0
PVT = 1 if PVT == 'Yes' else 0


PS_dict = {"Active": 0, "Ambulatory": 1, "Restricted": 2, "Selfcare": 3, "Disabled": 4}
PS = PS_dict[PS]

Ascites_dict = {"None": 0, "Mild": 1, "Moderate/Severe": 2}
Ascites = Ascites_dict[Ascites]

# Create a DataFrame with the input features
input_data = pd.DataFrame([[PVT, Total_Bil, Metastasis, Ascites, Albumin, Hemoglobin, ALP, Symptoms, PS]], columns=['PVT','Total_Bil','Metastasis', 'Ascites', ' Albumin', 'Hemoglobin', 'ALP','Symptoms','PS'])

# Predict the species using the model
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

# Display the prediction results
st.write("""
### Prediction
""")

st.write(f"Predicted Result: **{prediction[0]}**")

st.write("Prediction Probability:")
st.write(f"Dies: {prediction_proba[0][0]:.5f}")
st.write(f"Lives: {prediction_proba[0][1]:.5f}")
