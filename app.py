import streamlit as st
import pickle
import pandas as pd
from xgboost import XGBClassifier

# Load model
model = XGBClassifier()
model.load_model('xgb_model.json')

features = pickle.load(open('features.pkl', 'rb'))

st.title("Loan Default Predictor")

# Inputs
loan_amnt = st.slider('Loan Amount', 1000, 40000, 10000)
int_rate = st.slider('Interest Rate (%)', 5.0, 30.0, 12.0)
annual_inc = st.number_input('Annual Income', 20000, 200000, 60000)
dti = st.slider('Debt-to-Income Ratio', 0.0, 40.0, 15.0)
grade = st.selectbox('Loan Grade', ['A','B','C','D','E','F'])

# Input
input_dict = {
    'loan_amnt': loan_amnt,
    'int_rate': int_rate,
    'annual_inc': annual_inc,
    'dti': dti,
    'grade': grade
}

input_df = pd.DataFrame([input_dict])
input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=features, fill_value=0)

# Predict
if st.button('Predict'):
    prob = model.predict_proba(input_df)[0][1]

    st.metric("Default Probability", f"{prob*100:.2f}%")

    if prob > 0.4:
        st.error("High Risk")
    elif prob > 0.2:
        st.warning("Medium Risk")
    else:
        st.success("Low Risk")
st.image('shap_summary.png', caption='Feature importance (SHAP values)')
st.image('confusion_matrix.png', caption='Model performance')
st.subheader("How the model works")
st.write("""
This model predicts the probability of a borrower defaulting on a loan.
It uses features like income, interest rate, debt-to-income ratio, and loan grade.
SHAP values show which features influence predictions the most.
""")
