# Weather Forecasting Project - Model Training
# Author: Kanav Benipuri
# Date: July 2025

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sys
import joblib

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Print current working directory and script location
print(f"\nCurrent working directory: {os.getcwd()}")
print(f"Script location: {os.path.abspath(__file__)}")

# Use relative paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed_weather_data.csv')
REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# Create directories if they don't exist
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Verify directory creation
print("\nVerifying directory creation:")
print(f"Reports directory exists: {os.path.exists(REPORTS_DIR)}")
print(f"Models directory exists: {os.path.exists(MODELS_DIR)}")

print("Weather Forecasting - Model Training")
print("=" * 50)
print(f"\nProject root: {PROJECT_ROOT}")
print(f"Data path: {DATA_PATH}")
print(f"Reports directory: {REPORTS_DIR}")
print(f"Models directory: {MODELS_DIR}")

# Features and targets
print("\nSetting up features and targets...")
feature_cols = [
    'temperature_lag1', 'temperature_lag7',
    'rainfall_lag1', 'rainfall_lag7',
    'temperature_ma7', 'temperature_ma30',
    'rainfall_ma7', 'rainfall_ma30',
    'is_monsoon',
    'month', 'day_of_year',
    'temperature_range', 'temperature_variation'
]

target_cols = ['temperature', 'rainfall']

# Load processed data
print("\nLoading data...")
print(f"Loading data from: {DATA_PATH}")

try:
    df = pd.read_csv(DATA_PATH)
    print(f"\n Loaded processed data with shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head(3))
    
    # Check for missing values
    print("\nChecking for missing values:")
    missing_counts = df.isnull().sum()
    print(missing_counts[missing_counts > 0])
    
    # Handle missing values
    print("\nHandling missing values...")
    # Fill missing values with appropriate methods
    df['temperature'] = df['temperature'].fillna(df['temperature'].mean())
    df['rainfall'] = df['rainfall'].fillna(0)  # Rainfall is 0 when missing
    df['temperature_range'] = df['temperature_range'].fillna(df['temperature_range'].mean())
    df['temperature_variation'] = df['temperature_variation'].fillna(df['temperature_variation'].mean())
    
    # Verify no missing values remain
    print("\nAfter handling missing values:")
    print("Missing values:", df.isnull().sum().sum())
    
except Exception as e:
    print(f"❌ Error loading data: {str(e)}")
    print(f"Checking if file exists: {os.path.exists(DATA_PATH)}")
    if os.path.exists(DATA_PATH):
        print(f"File size: {os.path.getsize(DATA_PATH) / (1024*1024):.2f} MB")
    else:
        print("File does not exist!")
    raise

# Print column summary
print("\nColumns in dataset:", df.columns.tolist())
print("\nFirst few rows of data:")
print(df.head(3))

# Print feature and target columns
print("\nFeature columns:", feature_cols)
print("Target columns:", target_cols)

# Prepare data for modeling
print("\nPreparing data for modeling...")
print("\nChecking data types:")
print(df.dtypes)

print("\nChecking if all feature columns exist:")
for col in feature_cols:
    if col not in df.columns:
        print(f"Warning: Column {col} not found in dataset!")
    else:
        print(f"Found column: {col}")

target_cols = ['temperature', 'rainfall']

# Encode city as categorical
df['city'] = pd.Categorical(df['city'])
df['city_code'] = df['city'].cat.codes

# Split data
X = df[feature_cols + ['city_code']]
y_temp = df['temperature']
y_rain = df['rainfall']

# Split into train and test sets
X_train, X_test, y_temp_train, y_temp_test = train_test_split(
    X, y_temp, test_size=0.2, random_state=42, shuffle=True
)

_, _, y_rain_train, y_rain_test = train_test_split(
    X, y_rain, test_size=0.2, random_state=42, shuffle=True
)

print(f"Train set size: {len(X_train):,}")
print(f"Test set size: {len(X_test):,}")

# Handle missing values in features before scaling
print("\nHandling missing values in features...")
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Verify no NaN values in scaled features
print("\nVerifying no NaN values in scaled features:")
print("NaN values in X_train_scaled:", np.isnan(X_train_scaled).sum())
print("NaN values in X_test_scaled:", np.isnan(X_test_scaled).sum())

def evaluate_model(y_true, y_pred, model_name, target):
    """Evaluate model performance"""
    print(f"\nEvaluating {model_name} for {target}...")
    print(f"True values shape: {y_true.shape}")
    print(f"Predicted values shape: {y_pred.shape}")
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n {model_name} - {target} Forecasting Results:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")
    
    # Plot predictions vs actual
    print(f"Creating plot for {model_name} - {target}")
    plt.figure(figsize=(12, 6))
    plt.plot(y_true.values, label='Actual')
    plt.plot(y_pred, label='Predicted', alpha=0.7)
    plt.title(f'{model_name} - {target} Forecasting')
    plt.xlabel('Time')
    plt.ylabel(target)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(REPORTS_DIR, f'{model_name}_{target}_forecast.png')
    plt.savefig(plot_path)
    print(f"Saved plot to: {plot_path}")
    plt.close()

# 1. Linear Regression
print("\nTraining Linear Regression Models...")

# Temperature model
lr_temp = LinearRegression()
lr_temp.fit(X_train_scaled, y_temp_train)
y_temp_pred = lr_temp.predict(X_test_scaled)
evaluate_model(y_temp_test, y_temp_pred, 'Linear Regression', 'Temperature')

# Rainfall model
lr_rain = LinearRegression()
lr_rain.fit(X_train_scaled, y_rain_train)
y_rain_pred = lr_rain.predict(X_test_scaled)
evaluate_model(y_rain_test, y_rain_pred, 'Linear Regression', 'Rainfall')

# 2. Random Forest
print("\nTraining Random Forest Models...")

# Optimize Random Forest parameters for better performance
rf_params = {
    'n_estimators': 100,  # Reduced from default 100 to speed up
    'max_depth': 20,      # Limit depth to prevent overfitting
    'n_jobs': -1,         # Use all CPU cores
    'verbose': 1,         # Show progress
    'random_state': 42
}

# Temperature model
print("\nTraining Temperature Random Forest...")
rf_temp = RandomForestRegressor(**rf_params)
rf_temp.fit(X_train, y_temp_train)
y_temp_pred = rf_temp.predict(X_test)
evaluate_model(y_temp_test, y_temp_pred, 'Random Forest', 'Temperature')

# Rainfall model
print("\nTraining Rainfall Random Forest...")
rf_rain = RandomForestRegressor(**rf_params)
rf_rain.fit(X_train, y_rain_train)
y_rain_pred = rf_rain.predict(X_test)
evaluate_model(y_rain_test, y_rain_pred, 'Random Forest', 'Rainfall')

# 3. XGBoost
print("\nTraining XGBoost Models...")

# Temperature model
xgb_temp = xgb.XGBRegressor(random_state=42)
xgb_temp.fit(X_train, y_temp_train)
y_temp_pred = xgb_temp.predict(X_test)
evaluate_model(y_temp_test, y_temp_pred, 'XGBoost', 'Temperature')

# Rainfall model
xgb_rain = xgb.XGBRegressor(random_state=42)
xgb_rain.fit(X_train, y_rain_train)
y_rain_pred = xgb_rain.predict(X_test)
evaluate_model(y_rain_test, y_rain_pred, 'XGBoost', 'Rainfall')

# Save models
print("\nSaving models...")
import joblib

models = {
    'linear_regression': {'temp': lr_temp, 'rain': lr_rain},
    'random_forest': {'temp': rf_temp, 'rain': rf_rain},
    'xgboost': {'temp': xgb_temp, 'rain': xgb_rain}
}

for model_type, models in models.items():
    for target, model in models.items():
        model_path = os.path.join(MODELS_DIR, f'{model_type}_{target}_model.joblib')
        joblib.dump(model, model_path)
        print(f'Saved model to: {model_path}')

print("\nModel training completed!")
print("Models saved in 'models/' directory")
print("Evaluation plots saved in 'reports/' directory")
