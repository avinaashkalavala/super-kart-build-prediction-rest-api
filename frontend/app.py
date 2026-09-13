import streamlit as st
import pandas as pd
import requests

# Base URL of the Flask backend
#BACKEND_URL = "http://backend:7860"

BACKEND_URL = "http://172.18.0.1:7860"

# Set the title of the Streamlit app
st.title("SuperKart Sales Prediction")

# Section for online prediction
st.subheader("Online Prediction")

# Collect user input for product/store features
product_id = st.text_input("Product ID", "FD001")
product_weight = st.number_input("Product Weight", min_value=0.0, value=12.5, step=0.1)
product_sugar_content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
product_allocated_area = st.number_input("Product Allocated Area", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
product_type = st.selectbox("Product Type", [
    "Fruits and Vegetables", "Snack Foods", "Frozen Foods", "Dairy", "Household",
    "Baking Goods", "Canned", "Health and Hygiene", "Meat", "Soft Drinks",
    "Breads", "Hard Drinks", "Others", "Starchy Foods", "Breakfast", "Seafood"
])
product_mrp = st.number_input("Product MRP", min_value=0.0, value=150.0, step=1.0)
store_establishment_year = st.number_input("Store Establishment Year", min_value=1900, max_value=2026, value=2005, step=1)
store_size = st.selectbox("Store Size", ["High", "Medium", "Small"])
store_location_city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
store_type = st.selectbox("Store Type", [
    "Supermarket Type1", "Supermarket Type2", "Grocery Store", "Supermarket Type3"
])
store_id = st.selectbox("Store_Id",["OUT001","OUT002","OUT003","OUT004"])

# Convert user input into a DataFrame
input_data = pd.DataFrame([{
    'Product_Id': product_id,
    'Product_Weight': product_weight,
    'Product_Sugar_Content': product_sugar_content,
    'Product_Allocated_Area': product_allocated_area,
    'Product_Type': product_type,
    'Product_MRP': product_mrp,
    'Store_Establishment_Year': store_establishment_year,
    'Store_Size': store_size,
    'Store_Location_City_Type': store_location_city_type,
    'Store_Type': store_type,
    'Store_Id': store_id
}])

# Make prediction when the "Predict" button is clicked
if st.button("Predict", type="primary"):
    response = requests.post(f"{BACKEND_URL}/v1/predict", 
           json=input_data.to_dict(orient='records')[0],
          timeout=30)                 # Send data to Flask API            
    if response.status_code == 200:
        prediction = response.json()['Predicted Sales Total (in dollars)']
        st.success(f"Predicted Sales Total (in dollars): {prediction}")
    else:
        st.error("Unable to connect to the prediction API.")

# Section for batch prediction
st.subheader("Batch Prediction")

# Allow users to upload a CSV file for batch prediction
uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=["csv"])

# Make batch prediction when the "Predict Batch" button is clicked
if uploaded_file is not None:
    if st.button("Predict Batch", type="primary"):
        response = requests.post(f"{BACKEND_URL}/v1/predictbatch", 
              files={"file": uploaded_file},timeout=60)  # Send file to Flask API
        if response.status_code == 200:
            predictions = response.json()
            st.success("Batch predictions completed!")
            st.write(predictions)  # Display the predictions
        else:
            st.error("Unable to connect to the prediction API.")
