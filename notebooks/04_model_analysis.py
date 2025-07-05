# Weather Forecasting Model Analysis
# Author: Kanav Benipuri
# Date: July 2025

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import os

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("Starting Model Analysis")
print("=" * 50)

# Load the data
df = pd.read_csv('data/processed_weather_data.csv')
print(f"\nLoaded processed data with shape: {df.shape}")

# Load the best models
xgboost_temp = joblib.load('models/xgboost_temp_model.joblib')
random_forest_rain = joblib.load('models/random_forest_rain_model.joblib')

# Prepare data for analysis
# Create city_code from city column
print("\nEncoding city as categorical...")
df['city'] = pd.Categorical(df['city'])
df['city_code'] = df['city'].cat.codes

feature_cols = [
    'temperature_lag1', 'temperature_lag7',
    'rainfall_lag1', 'rainfall_lag7',
    'temperature_ma7', 'temperature_ma30',
    'rainfall_ma7', 'rainfall_ma30',
    'is_monsoon',
    'month', 'day_of_year',
    'temperature_range', 'temperature_variation',
    'city_code'
]

# Split data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Temperature analysis
def analyze_temperature_model():
    print("\nAnalyzing Temperature Model...")
    
    # Get test data
    X_test_temp = test_df[feature_cols]
    y_test_temp = test_df['temperature']
    
    # Handle missing values
    X_test_temp = X_test_temp.fillna(X_test_temp.mean())
    
    # Make predictions
    temp_preds = xgboost_temp.predict(X_test_temp)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test_temp, temp_preds))
    mae = mean_absolute_error(y_test_temp, temp_preds)
    r2 = r2_score(y_test_temp, temp_preds)
    
    print(f"Temperature Model Metrics:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")
    
    # Create plots
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_test_temp, temp_preds, alpha=0.3)
    plt.plot([y_test_temp.min(), y_test_temp.max()], [y_test_temp.min(), y_test_temp.max()], 'r--')
    plt.xlabel('Actual Temperature')
    plt.ylabel('Predicted Temperature')
    plt.title('Temperature Prediction Scatter Plot')
    
    plt.subplot(1, 2, 2)
    plt.hist(y_test_temp - temp_preds, bins=50, alpha=0.7)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Temperature Prediction Error Distribution')
    
    plt.tight_layout()
    plt.savefig('reports/temperature_model_analysis.png')
    plt.close()
    
    # Feature importance
    plt.figure(figsize=(10, 6))
    feature_importance = pd.Series(xgboost_temp.feature_importances_, index=feature_cols)
    feature_importance.sort_values(ascending=False).plot(kind='bar')
    plt.title('Temperature Model Feature Importance')
    plt.tight_layout()
    plt.savefig('reports/temperature_feature_importance.png')
    plt.close()

def analyze_rainfall_model():
    print("\nAnalyzing Rainfall Model...")
    
    # Get test data
    X_test_rain = test_df[feature_cols]
    y_test_rain = test_df['rainfall']
    
    # Handle missing values
    X_test_rain = X_test_rain.fillna(X_test_rain.mean())
    
    # Make predictions
    rain_preds = random_forest_rain.predict(X_test_rain)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test_rain, rain_preds))
    mae = mean_absolute_error(y_test_rain, rain_preds)
    r2 = r2_score(y_test_rain, rain_preds)
    
    print(f"Rainfall Model Metrics:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")
    
    # Create plots
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_test_rain, rain_preds, alpha=0.3)
    plt.plot([y_test_rain.min(), y_test_rain.max()], [y_test_rain.min(), y_test_rain.max()], 'r--')
    plt.xlabel('Actual Rainfall')
    plt.ylabel('Predicted Rainfall')
    plt.title('Rainfall Prediction Scatter Plot')
    
    plt.subplot(1, 2, 2)
    plt.hist(y_test_rain - rain_preds, bins=50, alpha=0.7)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Rainfall Prediction Error Distribution')
    
    plt.tight_layout()
    plt.savefig('reports/rainfall_model_analysis.png')
    plt.close()
    
    # Feature importance
    plt.figure(figsize=(10, 6))
    feature_importance = pd.Series(random_forest_rain.feature_importances_, index=feature_cols)
    feature_importance.sort_values(ascending=False).plot(kind='bar')
    plt.title('Rainfall Model Feature Importance')
    plt.tight_layout()
    plt.savefig('reports/rainfall_feature_importance.png')
    plt.close()

# Run analysis
analyze_temperature_model()
analyze_rainfall_model()

print("\nAnalysis completed! All plots saved in 'reports/' directory")
