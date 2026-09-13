
#Import necessary libraries
import numpy as np
import os
import joblib # For loading the seralized model
import pandas as pd # for data manupulation
from flask import Flask, request, jsonify  # For creating the Flask API

#Initialize the Flask application
kart_predictor_api = Flask("SuperKart Predictor")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'superkart_model.joblib')

#Load the trained machine learning model
model = joblib.load(MODEL_PATH)

#Define a route for the Home page(GET Request)
@kart_predictor_api.route('/')
def home():
    """
    This function handles GGET requests to the root url('/) of the API.
    It retruns a simple welcome message.
    """
    return "Welcome to SuperKart Sales Prediction API"


# Define an endpoint for single product/store sales prediction (POST request)
@kart_predictor_api.route('/v1/predict', methods=['POST'])
def predict_sales():
    """
    This function handles POST requests to the '/v1/predict' endpoint.
    It expects a JSON payload containing product and store details and returns
    the predicted Product_Store_Sales_Total as a JSON response.
    """

    try:
      # Get the JSON data from the request body
      product_data = request.get_json()

      #Get ProductID and create Productid prefix
      product_id = product_data.get('Product_Id')
      product_id_prefix = product_id[:2]

      # Create input data using same features during model training
      sample = {
          'Product_Weight': product_data['Product_Weight'],
          'Product_Sugar_Content': product_data['Product_Sugar_Content'],
          'Product_Allocated_Area': product_data['Product_Allocated_Area'],
          'Product_Type': product_data['Product_Type'],
          'Product_MRP': product_data['Product_MRP'],
          'Store_Establishment_Year': product_data['Store_Establishment_Year'],
          'Store_Id': product_data['Store_Id'],
          'Product_Id_Prefix': product_id_prefix,
          'Store_Size': product_data['Store_Size'],
          'Store_Location_City_Type': product_data['Store_Location_City_Type'],
          'Store_Type': product_data['Store_Type']
      }

      # Convert the extracted data into a Pandas DataFrame
      input_data = pd.DataFrame([sample])

      print("input columns:", input_data.columns.tolist())

      # Make prediction
      predicted_sales = model.predict(input_data)[0]

      # Convert predicted_sales to Python float
      # This conversion is needed because model.predict returns NumPy float32/float64 values,
      # and Flask's jsonify function encounters a datatype error if sent directly within a JSON response
      predicted_sales = round(float(predicted_sales), 2)

      # Return the predicted sales total
      return jsonify({'Predicted Sales Total (in dollars)': predicted_sales})

    except Exception as e:
      return jsonify({'error': str(e)}),400

# Define an endpoint for batch prediction (POST request)
# Define an endpoint for batch prediction (POST request)
@kart_predictor_api.post('/v1/predictbatch')
def predict_sales_batch():
    """
    This function handles POST requests to the '/v1/predictbatch' endpoint.
    It expects a CSV file containing product/store details for multiple products
    and returns the predicted sales totals as a dictionary in the JSON response.
    """
    try:
      # Get the uploaded CSV file from the request
      file = request.files['file']

      # Read the CSV file into a Pandas DataFrame
      input_data = pd.read_csv(file)

      #Create Product_id_prefix from product_id
      input_data['Product_id_Prefix'] = input_data['Product_Id_char'].astype(str).str[:2]

      #create Producttype from producttype category
      input_data['Product_Type'] =input_data['Product_Type_Category']
      # Make predictions for all products in the DataFrame (get log_sales)
      predicted_log_sales = model.predict(input_data).tolist()

      # Calculate actual sales totals
      predicted_sales = [round(float(np.exp(log_sales)), 2) for log_sales in predicted_log_sales]

      # Create a dictionary of predictions with product IDs as keys
      product_ids = input_data['Product_Id'].tolist()  # Assuming 'Product_Id' is the product ID column
      output_dict = dict(zip(product_ids, predicted_sales))  # Use actual sales totals

      # Return the predictions dictionary as a JSON response
      return output_dict

    except Exception as e:
      return jsonify({
         "error":str(e)
      }), 400

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    kart_predictor_api.run(debug=True)
