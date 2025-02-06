import pandas as pd
import streamlit as st
import numpy as np
import pickle
import base64
import re
import json

# Load the pickle file containing the model
model_location = "E:\DSDemo\env\project\env\Scripts\env\car_price_bestmodel.pkl"
with open(model_location, 'rb') as file:
    model = pickle.load(file)

# Load the dataset for dropdown options
fl = pd.read_csv('E:\DSDemo\env\project\env\Scripts\env\Car_Dheko_Datas.csv')

# Clean and preprocess dataset
fl['Kms Driven'] = fl['Kms Driven'].fillna(0).astype(str).str.replace(',', '').astype(int)
fl['Max Power'] = fl['Max Power'].str.extract(r'(\d+\.?\d*)').astype(float)

# Function to predict car prices
def predict_car_price(brand, model_name, year, kms_driven, fuel_type, transmission, engine_displacement):
    try:
        # Ensure correct category encoding
        if brand not in fl['oem'].unique():
            return {"error": f"Brand '{brand}' not found in the dataset."}
        if model_name not in fl['model'].unique():
            return {"error": f"Model '{model_name}' not found in the dataset."}

        body_type_encoded = fl['oem'].astype('category').cat.categories.get_loc(brand)
        model_name_encoded = fl['model'].astype('category').cat.categories.get_loc(model_name)
        
        input_data = np.array([[body_type_encoded, model_name_encoded, year, kms_driven, 
                                fuel_type, transmission, engine_displacement]])
        predicted_price = model.predict(input_data)[0]
        return {"predicted_price": f"₹{predicted_price:,.2f}", "brand": brand, "model": model_name, "year": year}
    except Exception as e:
        return {"error": f"Error during prediction: {str(e)}"}

# Set up background image
def set_bg_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url(data:image/png;base64,{encoded_string});
        background-size: cover;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set your background image file path
set_bg_image("E:\DSDemo\env\project\env\Scripts\env\pngtree-gray-coupe-plugin-hybrid-3d-rendering-a-concept-sports-car-with-image_3700199.jpg")

# Streamlit app
st.title("Car Price Prediction Chatbot 🚗")
st.write("Ask your car-related queries.")

# Capture user input
user_query = st.text_input("Your Query", "")

if user_query:
    # Query patterns
    price_pattern = r'predict price for (.+?) (.+?) (\d{4})'
    detail_pattern = r'show details for (.+?) (\d{4})'
    
    # Handle price prediction query
    if re.search(price_pattern, user_query.lower()):
        match = re.search(price_pattern, user_query.lower())
        brand, model_name, year = match.groups()
        response = predict_car_price(brand.capitalize(), model_name.capitalize(), int(year), 10000, 'Petrol', 'Manual', 1200)
        st.json(response)  # Display the response as JSON

    # Handle show details query
    elif re.search(detail_pattern, user_query.lower()):
        match = re.search(detail_pattern, user_query.lower())
        brand, year = match.groups()
        filtered_data = fl[(fl['oem'].str.lower() == brand.lower()) & (fl['modelYear'] == int(year))]
        if filtered_data.empty:
            st.json({"error": f"No details found for {brand} ({year})."})  # JSON response for no data
        else:
            st.json(filtered_data.head().to_dict(orient="records"))  # Display the details as JSON

    # Handle unsupported queries
    else:
        st.json({"error": "Sorry, I can only help with price predictions and car details queries right now."})


