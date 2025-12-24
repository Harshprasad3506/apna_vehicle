import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Vehicle Price Prediction", layout="centered")

st.title("🚗 Vehicle Price Prediction")

# Load trained model
model = joblib.load("best_vehicle_price_model.pkl")

st.subheader("Enter Vehicle Details")

make = st.selectbox("Make", [
    'Toyota', 'Honda', 'Ford', 'BMW', 'Audi', 'Mercedes-Benz'
])

fuel = st.selectbox("Fuel Type", [
    'Gasoline', 'Diesel', 'Hybrid', 'Electric'
])

transmission = st.selectbox("Transmission", [
    'Automatic', 'Manual', 'CVT'
])

body = st.selectbox("Body Type", [
    'SUV', 'Sedan', 'Hatchback', 'Pickup Truck'
])

drivetrain = st.selectbox("Drivetrain", [
    'All-wheel Drive', 'Front-wheel Drive', 'Rear-wheel Drive'
])

mileage = st.number_input("Mileage", min_value=0, value=45000)
car_age = st.number_input("Car Age (years)", min_value=0, value=4)
cylinders = st.number_input("Cylinders", min_value=2, value=4)
doors = st.number_input("Doors", min_value=2, value=4)

if st.button("Predict Price"):
    input_df = pd.DataFrame({
        'make': [make],
        'fuel': [fuel],
        'transmission': [transmission],
        'body': [body],
        'drivetrain': [drivetrain],
        'mileage': [mileage],
        'Car_Age': [car_age],
        'cylinders': [cylinders],
        'doors': [doors]
    })

    price = model.predict(input_df)[0]

    st.success(f"💰 Estimated Vehicle Price: ${price:,.2f}")
