import numpy as np
import pandas as pd
import pickle
import streamlit as st


st.title("Customer Churn Prediction App")
st.write("Enter customer details to predict if they will exit the bank.")

# Create a form to group inputs
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
        geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=3)

    with col2:
        balance = st.number_input("Balance", min_value=0.0, value=50000.0)
        num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
        has_cr_card = st.selectbox("Has Credit Card?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        is_active_member = st.selectbox("Is Active Member?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
        estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)

    submit_button = st.form_submit_button("Predict Churn")


def engg_features(X):
    X["BalanceSalaryRatio"] = X["Balance"] / X["EstimatedSalary"]
    X["TenureByAge"] = X["Tenure"] / X["Age"]
    X["CreditScoreGivenAge"] = X["CreditScore"] / X["Age"]
    X["HasBalance"] = np.where(X["Balance"] > 0, 1, 0)
    X["ActiveByAge"] = X["IsActiveMember"] * X["Age"]
    X['AgeCategory'] = pd.cut(X['Age'], bins=[0, 35, 55, np.inf], labels=['Young', 'MiddleAge', 'Senior'])
    return X

model = pickle.load(open("./models/final_model_pipeline.pkl", "rb"))

if submit_button and model is not None:
    input_data = {
        'CreditScore': [credit_score],
        'Geography': [geography],
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],       
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary] 
    }

    input_df = pd.DataFrame(input_data)


    try:
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[0][1]

        if prediction[0] == 1:
            st.error(f"**Prediction: Churn** (Probability: {probability:.2%})")
            st.write("This customer is likely to leave the bank.")
        else:
            st.success(f"**Prediction: No Churn** (Probability: {probability:.2%})")
            st.write("This customer is likely to stay.")
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")