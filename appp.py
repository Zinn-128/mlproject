import streamlit as st
import pandas as pd
import joblib

# Load the models
try:
    best_classification_model = joblib.load('gd_classification_model.pkl')
    best_regression_model = joblib.load('gd_regression_model.pkl')
    models_loaded = True
except Exception as e:
    st.write(f"Error loading the models: {e}")
    models_loaded = False

st.write("""
# Telco Customer Churn and Monthly Charges Prediction App
This app predicts whether a customer will churn and their monthly charges.
""")

st.sidebar.header('User Input Features')

# Define the complete feature list as used in the training process
feature_list_classification = [
    'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PaperlessBilling',
    'MonthlyChargesPerTenure', 'MultipleLines_Yes', 'InternetService_DSL', 
    'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_No', 
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 'OnlineBackup_No', 
    'OnlineBackup_No internet service', 'OnlineBackup_Yes', 'DeviceProtection_No', 
    'DeviceProtection_No internet service', 'DeviceProtection_Yes', 'TechSupport_No', 
    'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No', 
    'StreamingTV_No internet service', 'StreamingTV_Yes', 'StreamingMovies_No', 
    'StreamingMovies_No internet service', 'StreamingMovies_Yes', 'Contract_Month-to-month', 
    'Contract_One year', 'Contract_Two year', 'PaymentMethod_Bank transfer (automatic)', 
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 
    'PaymentMethod_Mailed check'
]

feature_list_regression = feature_list_classification + ['Churn']

# Function to get user inputs
def user_input_features():
    tenure = st.sidebar.slider('Tenure', 0, 72, 24)
    MonthlyChargesPerTenure = st.sidebar.slider('Monthly Charges Per Tenure', 0.0, 100.0, 50.0)
    Contract = st.sidebar.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    InternetService = st.sidebar.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    OnlineSecurity = st.sidebar.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
    PaymentMethod = st.sidebar.selectbox('Payment Method', ['Bank transfer (automatic)', 'Credit card (automatic)', 'Electronic check', 'Mailed check'])
    TechSupport = st.sidebar.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
    StreamingMovies = st.sidebar.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])

    data = {
        'SeniorCitizen': 0,  # Assuming default as 0
        'Partner': 0,  # Assuming default as 0
        'Dependents': 0,  # Assuming default as 0
        'tenure': tenure,
        'PaperlessBilling': 0,  # Assuming default as 0
        'MonthlyChargesPerTenure': MonthlyChargesPerTenure,
        'MultipleLines_Yes': 0,  # Assuming default as 0
        'InternetService_DSL': 1 if InternetService == 'DSL' else 0,
        'InternetService_Fiber optic': 1 if InternetService == 'Fiber optic' else 0,
        'InternetService_No': 1 if InternetService == 'No' else 0,
        'OnlineSecurity_No': 1 if OnlineSecurity == 'No' else 0,
        'OnlineSecurity_No internet service': 1 if OnlineSecurity == 'No internet service' else 0,
        'OnlineSecurity_Yes': 1 if OnlineSecurity == 'Yes' else 0,
        'OnlineBackup_No': 0,  # Assuming default as 0
        'OnlineBackup_No internet service': 0,  # Assuming default as 0
        'OnlineBackup_Yes': 0,  # Assuming default as 0
        'DeviceProtection_No': 0,  # Assuming default as 0
        'DeviceProtection_No internet service': 0,  # Assuming default as 0
        'DeviceProtection_Yes': 0,  # Assuming default as 0
        'TechSupport_No': 1 if TechSupport == 'No' else 0,
        'TechSupport_No internet service': 1 if TechSupport == 'No internet service' else 0,
        'TechSupport_Yes': 1 if TechSupport == 'Yes' else 0,
        'StreamingTV_No': 0,  # Assuming default as 0
        'StreamingTV_No internet service': 0,  # Assuming default as 0
        'StreamingTV_Yes': 0,  # Assuming default as 0
        'StreamingMovies_No': 1 if StreamingMovies == 'No' else 0,
        'StreamingMovies_No internet service': 1 if StreamingMovies == 'No internet service' else 0,
        'StreamingMovies_Yes': 1 if StreamingMovies == 'Yes' else 0,
        'Contract_Month-to-month': 1 if Contract == 'Month-to-month' else 0,
        'Contract_One year': 1 if Contract == 'One year' else 0,
        'Contract_Two year': 1 if Contract == 'Two year' else 0,
        'PaymentMethod_Bank transfer (automatic)': 1 if PaymentMethod == 'Bank transfer (automatic)' else 0,
        'PaymentMethod_Credit card (automatic)': 1 if PaymentMethod == 'Credit card (automatic)' else 0,
        'PaymentMethod_Electronic check': 1 if PaymentMethod == 'Electronic check' else 0,
        'PaymentMethod_Mailed check': 1 if PaymentMethod == 'Mailed check' else 0
    }

    return data

# Prepare the complete feature dataframe
def prepare_feature_dataframe(user_data, feature_list):
    # Define all features with default values
    all_features = {feature: 0 for feature in feature_list}
    all_features.update(user_data)
    
    # Create dataframe
    features_df = pd.DataFrame(all_features, index=[0])
    return features_df

# Upload CSV function
def upload_csv():
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        return input_df
    else:
        return None

# Get user input
input_method = st.sidebar.radio("Select input method", ("Manual Input", "Upload CSV"))

if input_method == "Manual Input":
    user_data = user_input_features()
    input_df_classification = prepare_feature_dataframe(user_data, feature_list_classification)
    input_df_regression = prepare_feature_dataframe(user_data, feature_list_regression)
else:
    input_df = upload_csv()
    if input_df is not None:
        input_df_classification = input_df[feature_list_classification]
        input_df_regression = input_df[feature_list_regression]
    else:
        st.write("Please upload a CSV file.")
        input_df_classification = pd.DataFrame(columns=feature_list_classification)
        input_df_regression = pd.DataFrame(columns=feature_list_regression)

st.subheader('User Input Features')
st.write(input_df_classification)

if models_loaded:
    try:
        # Churn Prediction
        churn_prediction = best_classification_model.predict(input_df_classification)
        churn_prediction_proba = best_classification_model.predict_proba(input_df_classification)

        # Monthly Charges Prediction
        monthly_charges_prediction = best_regression_model.predict(input_df_regression)

        st.subheader('Churn Prediction')
        st.write(f"Churn: {'Yes' if churn_prediction[0] == 1 else 'No'}")
        st.write(f"Churn Probability: {churn_prediction_proba[0][1]:.2f}")

        st.subheader('Monthly Charges Prediction')
        st.write(f"Predicted Monthly Charges: ${monthly_charges_prediction[0]:.2f}")

        st.subheader("Prediction Results")
        result_df = pd.DataFrame({
            "Churn": ["Yes" if churn_prediction[0] == 1 else "No"],
            "Churn Probability": [f"{churn_prediction_proba[0][1]:.2f}"],
            "Predicted Monthly Charges": [f"${monthly_charges_prediction[0]:.2f}"]
        })
        st.write(result_df)
    except Exception as e:
        st.write(f"Error making predictions: {e}")
else:
    st.write("Error loading the models. Please check the model files.")


