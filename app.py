import streamlit as st
import pandas as pd
import numpy as np
import joblib  # For loading the trained model

# Load the trained model
model = joblib.load('bigmart_model')

# Streamlit app
st.title("Item Outlet Sales Prediction")
st.write("Provide the details below to predict the sales.")

# Input form for user to provide inputs
with st.form("prediction_form"):
    # Numeric input for Item MRP
    Item_MRP = st.number_input("Item MRP", min_value=0.0, step=0.01)
    
    # Categorical input for Outlet Identifier
    Outlet_Identifier = st.selectbox(
        "Outlet Identifier", options=['OUT010', 'OUT013', 'OUT017', 'OUT018', 'OUT019', 'OUT027', 'OUT035', 'OUT045', 'OUT046', 'OUT049']
    )
    
    # Categorical input for Outlet Size
    Outlet_Size = st.selectbox("Outlet Size", options=['High', 'Medium', 'Small'])
    
    # Categorical input for Outlet Type
    Outlet_Type = st.selectbox("Outlet Type", options=['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3'])
    
    # Numeric input for Outlet Age (calculated from establishment year)
    Outlet_Establishment_Year = st.number_input("Outlet Establishment Year", min_value=1900, max_value=2024, step=1)
    
    # Submit button
    submitted = st.form_submit_button("Predict")

# Prediction logic
if submitted:
    # Convert categorical inputs to corresponding numerical values
    outlet_identifier_map = {
        'OUT010': 0, 'OUT013': 1, 'OUT017': 2, 'OUT018': 3, 'OUT019': 4,
        'OUT027': 5, 'OUT035': 6, 'OUT045': 7, 'OUT046': 8, 'OUT049': 9
    }
    
    outlet_size_map = {'High': 0, 'Medium': 1, 'Small': 2}
    outlet_type_map = {'Grocery Store': 0, 'Supermarket Type1': 1, 'Supermarket Type2': 2, 'Supermarket Type3': 3}
    
    # Prepare the input data for the model
    outlet_identifier = outlet_identifier_map[Outlet_Identifier]
    outlet_size = outlet_size_map[Outlet_Size]
    outlet_type = outlet_type_map[Outlet_Type]
    outlet_age = 2024 - Outlet_Establishment_Year  # Assuming current year is 2024

    input_data = np.array([[Item_MRP, outlet_identifier, outlet_size, outlet_type, outlet_age]])
    
    # Predict using the loaded model
    prediction = model.predict(input_data)
    
    # Display the prediction
    st.write(f"Predicted Item Outlet Sales: **{prediction[0]:.2f}**")
