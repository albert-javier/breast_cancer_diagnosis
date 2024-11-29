import streamlit as st
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import joblib

# Load and preprocess data
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)


# Load the saved model and scaler
model = joblib.load('ann_model.pkl')
scaler = joblib.load('scaler.pkl')



# --- App Title ---
st.markdown('<h1 class="title">Breast Cancer Analysis</h1>', unsafe_allow_html=True)
st.markdown('<h6 class="title" style="font-weight: normal;">AI-powered tool for accurate breast cancer analysis.</h6>', unsafe_allow_html=True)

st.divider()

# Sidebar for user input
st.sidebar.header("Input Features")


# input type selection either slider or number input
input_type = st.sidebar.radio("Input Type", ["Slider", "Number Input"])

# input data based on input type
if input_type == "Slider":
    st.sidebar.subheader("Enter Values for Features")   
    input_data = [st.sidebar.slider(f, float(df[f].min()), float(df[f].max()), format="%.6f"  ) for f in data.feature_names]
else:
    st.sidebar.subheader("Enter Values for Features")
    input_data = [st.sidebar.number_input(f, float(df[f].min()), float(df[f].max()),format="%.6f" ) for f in data.feature_names]



# Make predictions
if st.sidebar.button("Predict"):
    
    input_data_scaled = scaler.transform([input_data])
    prediction = model.predict(input_data_scaled)
    prediction_prob = model.predict_proba(input_data_scaled)
    
        
    # show in bold string    
    st.markdown(f"<h3 style='font-weight: bold;'>Prediction: {'<font color=\"red\">Malignant (cancerous)</font>' if prediction[0] == 1 else '<font color=\"green\">Benign (noncancerous)</font>'}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='font-weight: bold;'>Probality (Malignant): { (prediction_prob[0][1]*100) }%</h3>", unsafe_allow_html=True)
    
    # display the input data 
    st.write("Input Features:")
    st.table(pd.DataFrame([input_data], columns=data.feature_names).T.rename(columns={0: "Values"}))    
    
    
    
            
    
