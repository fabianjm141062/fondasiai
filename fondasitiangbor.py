import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Title of the app
st.title('Prediction of qa, sa, qh, yh, and bm using Machine Learning oleh Fabian J Manoppo')

# File upload
uploaded_file = st.file_uploader("Upload your 'datasetteori.csv' file", type="csv")

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)

    # Display first few rows of the dataset
    st.write("Dataset Preview:")
    st.write(df.head())

    # Define features (X) and targets (y)
    X = df[['diameter', 'length', 'nspt1', 'nspt2', 'nspt3']]  # Input features
    y = df[['qa', 'sa', 'qh', 'yh', 'bm']]  # Target variables

    # Split data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create RandomForestRegressor model and wrap it in MultiOutputRegressor
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(X_train, y_train)

    # Create input fields for manual input in Streamlit
    st.subheader("Input values for prediction:")
    diameter = st.number_input("Diameter:", value=0.0)
    length = st.number_input("Length:", value=0.0)
    nspt1 = st.number_input("NSPT1:", value=0.0)
    nspt2 = st.number_input("NSPT2:", value=0.0)
    nspt3 = st.number_input("NSPT3:", value=0.0)

    # When the button is clicked, perform prediction
    if st.button("Predict"):
        # Create a NumPy array of the manually input features
        input_data = np.array([[diameter, length, nspt1, nspt2, nspt3]])

        # Predict the target values based on the input
        predicted_values = model.predict(input_data)

        # Display the predicted results
        predicted_qa, predicted_sa, predicted_qh, predicted_yh, predicted_bm = predicted_values[0]
        st.write("Predicted values based on your inputs:")
        st.write(f"Predicted qa: {predicted_qa}")
        st.write(f"Predicted sa: {predicted_sa}")
        st.write(f"Predicted qh: {predicted_qh}")
        st.write(f"Predicted yh: {predicted_yh}")
        st.write(f"Predicted bm: {predicted_bm}")

    # Optionally, evaluate the model on test data and show performance metrics
    y_pred_test = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred_test, multioutput='raw_values')
    mae = mean_absolute_error(y_test, y_pred_test, multioutput='raw_values')
    r2 = r2_score(y_test, y_pred_test, multioutput='raw_values')

    st.subheader("\nModel performance on test set:")
    st.write(f"Mean Squared Error (MSE) for qa, sa, qh, yh, bm: {mse}")
    st.write(f"Mean Absolute Error (MAE) for qa, sa, qh, yh, bm: {mae}")
    st.write(f"R-squared (R²) for qa, sa, qh, yh, bm: {r2}")

# Streamlit Run Command
# To run the app, use the following command in your terminal:
# streamlit run your_script_name.py
