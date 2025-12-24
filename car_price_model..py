# ================================================
# 🚗 VEHICLE PRICE PREDICTION - PYTHON SCRIPT
# (DEPLOY-SAFE + STREAMLIT COMPATIBLE)
# ================================================

# --- 1. SETUP ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
file_name = "dataset.csv"   # CSV same folder me hona chahiye

print(f"✅ Loading file: {file_name}")

try:
    df = pd.read_csv(file_name)
    print(f"✅ Data Loaded Successfully! Shape: {df.shape}")
except FileNotFoundError:
    raise FileNotFoundError(f"❌ '{file_name}' file nahi mili")

# ================================================
# --- 2. DATA CLEANING & FEATURE ENGINEERING ---
# ================================================
print("\n--- Data Cleaning ---")

# Price filter
df_clean = df[df["price"] > 1000].copy()

# Unused / heavy text columns drop
df_clean.drop(
    columns=[
        "description", "name", "model", "trim",
        "exterior_color", "interior_color", "engine"
    ],
    inplace=True,
    errors="ignore"
)

# Car age feature
CURRENT_YEAR = 2025
df_clean["Car_Age"] = CURRENT_YEAR - df_clean["year"]
df_clean.drop(columns=["year"], inplace=True)

# Handle missing values
numerical_cols = ["cylinders", "mileage", "doors"]
categorical_cols = ["fuel", "transmission", "body", "make", "drivetrain"]

for col in numerical_cols:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

for col in categorical_cols:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

print(f"✅ Cleaned Data Shape: {df_clean.shape}")

# ================================================
# --- 3. SPLIT FEATURES / TARGET ---
# ================================================
numerical_features = ["mileage", "Car_Age", "cylinders", "doors"]
categorical_features = ["make", "fuel", "transmission", "body", "drivetrain"]

X = df_clean.drop("price", axis=1)
y = df_clean["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================================
# --- 4. PREPROCESSOR (SAFE ONEHOTENCODER) ---
# ================================================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        # ❗ sparse=False → sklearn safe for deploy
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_features),
    ]
)

# ================================================
# --- 5. MODEL + GRID SEARCH ---
# ================================================
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(random_state=42)),
    ]
)

param_grid = {
    "regressor__n_estimators": [200, 500],
    "regressor__max_depth": [10, 20],
    "regressor__min_samples_split": [5, 10],
}

grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="r2",
    n_jobs=-1,
    verbose=2,
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("✅ Best Params:", grid_search.best_params_)

# ================================================
# --- 6. EVALUATION ---
# ================================================
y_pred = best_model.predict(X_test)

print(f"MAE: ${mean_absolute_error(y_test, y_pred):,.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")

# ================================================
# --- 7. OPTIONAL PLOT (LOCAL ONLY) ---
# ================================================
plt.figure(figsize=(6, 5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")
plt.grid(True)
plt.show()

# ================================================
# --- 8. SAVE MODEL (DEPLOY SAFE) ---
# ================================================
joblib.dump(best_model, "best_vehicle_price_model.pkl")
print("💾 Model saved: best_vehicle_price_model.pkl")

# ================================================
# --- 9. TERMINAL TEST (OPTIONAL) ---
# ================================================
print("\n🚗 VEHICLE PRICE PREDICTION (TERMINAL)")
print("-" * 50)

user_vehicle = pd.DataFrame({
    "make": ["Toyota"],
    "fuel": ["Gasoline"],
    "transmission": ["Automatic"],
    "body": ["SUV"],
    "drivetrain": ["All-wheel Drive"],
    "mileage": [45000],
    "Car_Age": [4],
    "cylinders": [4],
    "doors": [4],
})

predicted_price = best_model.predict(user_vehicle)[0]

print(f"💰 Sample Prediction: ${predicted_price:,.2f}")
