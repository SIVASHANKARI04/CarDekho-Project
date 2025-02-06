import streamlit as st
import numpy as np
import pandas as pd
import pickle
import base64

# Load the pickle file containing the model
model_location = "model.pkl"
with open(model_location, 'rb') as file:
    model = pickle.load(file)

# Load the dataset for dropdown options
fl = pd.read_csv('car_dheko_filled.csv')

# Clean and preprocess dataset
# Remove commas from `Kms_Driven` and convert to integer
fl['Kms_Driven'] = fl['Kms_Driven'].fillna(0).astype(str).str.replace(',', '').astype(int)

# Extract numeric values from `Max Power`
fl['Max Power'] = fl['Max Power'].str.extract(r'(\d+\.?\d*)').astype(float)

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
set_bg_image("pexels-mikebirdy-120049.jpg")

# Custom CSS for Sidebar Styling with White, Black, and Blue
sidebar_style = """
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #ffffff, #4f83cc, #000000); /* Gradient from White to Blue to Black */
        color: white;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] label {
        color: white;
        font-weight: bold;
    }
    [data-testid="stSidebar"] .css-1e4r1h6 {
        background-color: transparent;
    }
    </style>
"""
st.markdown(sidebar_style, unsafe_allow_html=True)


# Streamlit app
st.title("Car Price Prediction App")
st.write("Enter the details of the car to predict its price.")

# Sidebar for filtering options
st.sidebar.header('Filter Options')

# Filter data dynamically based on user selections
city = st.sidebar.selectbox("Select the city:", options=fl['city'].unique())
filtered_data = fl[fl['city'] == city]

body_type = st.sidebar.selectbox("Select body type:", options=filtered_data['body_type'].unique())
filtered_data = filtered_data[filtered_data['body_type'] == body_type]

year = st.sidebar.selectbox("Select the year:", options=sorted(filtered_data['modelYear'].unique()))
filtered_data = filtered_data[filtered_data['modelYear'] == year]

oem = st.sidebar.selectbox("Select OEM (Manufacturer):", options=filtered_data['oem'].unique())
filtered_data = filtered_data[filtered_data['oem'] == oem]

model_name = st.sidebar.selectbox("Select Car Model:", options=filtered_data['model'].unique())
filtered_data = filtered_data[filtered_data['model'] == model_name]

Mileage = st.sidebar.selectbox("Select Mileage (in km/l):", options=sorted(filtered_data['Mileage'].unique()))
filtered_data = filtered_data[filtered_data['Mileage'] == Mileage]

engine_type = st.sidebar.selectbox("Select the engine type:", options=sorted(filtered_data['Engine Type'].unique()))
filtered_data = filtered_data[filtered_data['Engine Type'] == engine_type]

engine_displacement = st.sidebar.selectbox("Select Engine Displacement (in CC):", options=sorted(filtered_data['Engine_Displacement'].unique()))
filtered_data = filtered_data[filtered_data['Engine_Displacement'] == engine_displacement]

seating_capacity = st.sidebar.selectbox("Select seating capacity:", options=sorted(filtered_data['Seating_Capacity'].unique()))
filtered_data = filtered_data[filtered_data['Seating_Capacity'] == seating_capacity]

ownership = st.sidebar.selectbox("Select ownership type:", options=filtered_data['Ownership'].unique())
filtered_data = filtered_data[filtered_data['Ownership'] == ownership]

kms_driven = st.sidebar.number_input("Enter kilometers driven:", min_value=0, value=10000)

fuel_type = st.sidebar.selectbox("Select fuel type:", options=filtered_data['Fuel Type'].unique())
filtered_data = filtered_data[filtered_data['Fuel Type'] == fuel_type]

transmission = st.sidebar.selectbox("Select transmission type:", options=filtered_data['Transmission'].unique())
filtered_data = filtered_data[filtered_data['Transmission'] == transmission]

max_power = st.sidebar.selectbox("Select Max Power (in BHP):", options=sorted(filtered_data['Max Power'].unique()))
filtered_data = filtered_data[filtered_data['Max Power'] == max_power]

Acceleration = st.sidebar.selectbox("Select Acceleration type:", options=sorted(filtered_data['Acceleration'].unique()))
filtered_data = filtered_data[filtered_data['Acceleration'] == Acceleration]

# Predict button
if st.button("Predict Price"):
    try:
        # Encode categorical variables
        body_type_encoded = fl['body_type'].astype('category').cat.categories.get_loc(body_type)
        city_encoded = fl['city'].astype('category').cat.categories.get_loc(city)
        engine_type_encoded = fl['Engine Type'].astype('category').cat.categories.get_loc(engine_type)
        fuel_type_encoded = fl['Fuel Type'].astype('category').cat.categories.get_loc(fuel_type)
        ownership_encoded = fl['Ownership'].astype('category').cat.categories.get_loc(ownership)
        transmission_encoded = fl['Transmission'].astype('category').cat.categories.get_loc(transmission)
        oem_encoded = fl['oem'].astype('category').cat.categories.get_loc(oem)
        model_name_encoded = fl['model'].astype('category').cat.categories.get_loc(model_name)

        # Prepare input data in the correct order
        input_data = np.array([[city_encoded, body_type_encoded, oem_encoded, model_name_encoded, year, 
                                fuel_type_encoded, ownership_encoded, transmission_encoded, engine_type_encoded, 
                                seating_capacity, Mileage, max_power, engine_displacement, kms_driven, Acceleration]])

         # Predict price
        predicted_price = model.predict(input_data)[0]

       # Display the predicted price, styled and in the same order
        st.markdown(f"<h3 style='text-align: center; color: darkblue;'>The predicted price of the car is: ₹{predicted_price:,.2f}</h3>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
   
st.write("Note: This is a demo app. The predictions depend on the model's training data.")


