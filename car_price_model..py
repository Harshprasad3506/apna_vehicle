# ================================================
# 🚗 VEHICLE PRICE PREDICTION - PYTHON SCRIPT VERSION
# ================================================

# --- 1. SETUP ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import joblib

# ================================================
# --- 1A. LOAD DATASET ---
# ================================================
file_name = "dataset.csv"  # <-- CSV same folder me rakho

print(f"✅ Loading file: {file_name}")

try:
    df = pd.read_csv(file_name)
    print(f"✅ Data Loaded Successfully! Shape: {df.shape}")
except FileNotFoundError:
    print(f"❌ Error: File '{file_name}' not found.")
    df = None

if df is not None:

    # ================================================
    # --- 2. DATA CLEANING & FEATURE ENGINEERING ---
    # ================================================
    print("\n--- Data Cleaning ---")

    df_clean = df[df['price'] > 1000].copy()

    df_clean.drop(columns=[
        'description', 'name', 'model', 'trim',
        'exterior_color', 'interior_color', 'engine'
    ], inplace=True, errors='ignore')

    current_year = 2025
    df_clean['Car_Age'] = current_year - df_clean['year']
    df_clean.drop(columns=['year'], inplace=True)

    numerical_cols = ['cylinders', 'mileage', 'doors']
    categorical_cols = ['fuel', 'transmission', 'body']

    for col in numerical_cols:
        if col in df_clean.columns:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)

    for col in categorical_cols:
        if col in df_clean.columns:
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)

    print(f"✅ Cleaned Data Shape: {df_clean.shape}")

    # ================================================
    # --- 3. SPLITTING & PREPROCESSING ---
    # ================================================
    numerical_features = ['mileage', 'Car_Age', 'cylinders', 'doors']
    categorical_features = ['make', 'fuel', 'transmission', 'body', 'drivetrain']

    X = df_clean.drop('price', axis=1)
    y = df_clean['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

    # ================================================
    # --- 4. MODEL + GRID SEARCH ---
    # ================================================
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])

    param_grid = {
        'regressor__n_estimators': [200, 500],
        'regressor__max_depth': [10, 20],
        'regressor__min_samples_split': [5, 10]
    }

    grid_search = GridSearchCV(
        rf_pipeline,
        param_grid,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    best_rf_model = grid_search.best_estimator_

    print("✅ Best Params:", grid_search.best_params_)

    # ================================================
    # --- 5. EVALUATION ---
    # ================================================
    y_pred = best_rf_model.predict(X_test)

    print(f"MAE: ${mean_absolute_error(y_test, y_pred):,.2f}")
    print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")

    # ================================================
    # --- 6. PLOTS (OPTIONAL FOR DEPLOY) ---
    # ================================================
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted")
    plt.grid(True)
    plt.show()

    # ================================================
    # --- 7. SAVE MODEL ---
    # ================================================
    joblib.dump(best_rf_model, "best_vehicle_price_model.pkl")
    print("💾 Model saved: best_vehicle_price_model.pkl")

    # ================================================
    # --- 8. INTERACTIVE PREDICTION (CLI VERSION) ---
    # ================================================

    print("\n🚗 VEHICLE PRICE PREDICTION (TERMINAL)")
    print("-" * 50)

    make = input("Make (Toyota): ") or "Toyota"
    fuel = input("Fuel (Gasoline): ") or "Gasoline"
    transmission = input("Transmission (Automatic): ") or "Automatic"
    body = input("Body (SUV): ") or "SUV"
    drivetrain = input("Drivetrain (All-wheel Drive): ") or "All-wheel Drive"

    mileage = float(input("Mileage: "))
    car_age = int(input("Car Age (years): "))
    cylinders = int(input("Cylinders: "))
    doors = int(input("Doors: "))

    user_vehicle = pd.DataFrame({
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

    predicted_price = best_rf_model.predict(user_vehicle)[0]

    print("\n✅ Predicted Vehicle Price")
    print(f"💰 ${predicted_price:,.2f}")

    # ================================================
    # --- HTML / IPYTHON DISPLAY PART (COMMENTED) ---
    # ================================================

    """
    from IPython.display import display, HTML

    display(HTML("<h3>Predicted Price</h3>"))
    display(HTML(f"<h2>${predicted_price:,.2f}</h2>"))
    """

