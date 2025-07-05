# Weather Forecasting Project - Best Model Prediction
# Author: Kanav Benipuri
# Date: July 2025

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("Finding Best Model for Weather Prediction")
print("=" * 50)

# Load processed data
df = pd.read_csv('data/processed_weather_data.csv')
print(f"\nLoaded processed data with shape: {df.shape}")

# Prepare data for modeling
print("\nPreparing data for modeling...")

# Features and targets
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

# Load models
print("\nLoading models...")
models = {}
for model_type in ['linear_regression', 'random_forest', 'xgboost']:
    for target in ['temp', 'rain']:
        model_path = f'models/{model_type}_{target}_model.joblib'
        models[f'{model_type}_{target}'] = joblib.load(model_path)

# Evaluate all models
def evaluate_model(y_true, y_pred, model_name, target):
    """Evaluate model performance"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return rmse, mae, r2

# Store evaluation results
results = {}

# Evaluate temperature models
print("\nEvaluating Temperature Models...")
print(f"Test data shape: {X_test.shape}")
print(f"Test target shape: {y_temp_test.shape}")

# Print feature columns
print("\nFeature columns:")
for col in feature_cols:
    print(f"- {col}")

for model_name, model in models.items():
    if 'temp' in model_name:
        print(f"\nEvaluating {model_name}...")
        try:
            print(f"Model input shape: {X_test.shape}")
            print("Sample input data:")
            print(X_test.head(2))

            # Initialize imputer
            imputer = SimpleImputer(strategy='mean')
            
            # Impute missing values
            X_test_imputed = pd.DataFrame(imputer.fit_transform(X_test),
                                        columns=X_test.columns,
                                        index=X_test.index)

            # Make predictions
            y_pred = model.predict(X_test_imputed)
            print(f"Prediction shape: {y_pred.shape}")

            # Print sample predictions
            print("\nSample predictions:")
            print("Actual vs Predicted:")
            for i in range(5):
                print(f"Actual: {y_temp_test.iloc[i]:.2f}, Predicted: {y_pred[i]:.2f}")

            # Evaluate model
            rmse, mae, r2 = evaluate_model(y_temp_test, y_pred, model_name, 'temp')
            print(f"\n{model_name} metrics:")
            print(f"RMSE: {rmse:.2f}")
            print(f"MAE: {mae:.2f}")
            print(f"R²: {r2:.4f}")
            
            results[model_name] = {'rmse': rmse, 'mae': mae, 'r2': r2}

            # Create prediction vs actual plot
            plt.figure(figsize=(12, 6))
            plt.plot(y_temp_test.values[:50], label='Actual')
            plt.plot(y_pred[:50], label='Predicted', alpha=0.7)
            plt.title(f'{model_name} - Temperature Forecasting')
            plt.xlabel('Time')
            plt.ylabel('Temperature (°C)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'reports/{model_name}_temperature_forecast.png')
            plt.close()
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")

# Initialize imputer for rainfall models
imputer_rain = SimpleImputer(strategy='mean')

# Evaluate rainfall models
print("\nEvaluating Rainfall Models...")
for model_name, model in models.items():
    if 'rain' in model_name:
        try:
            print(f"Evaluating {model_name}...")
            
            # Impute missing values
            X_test_imputed = pd.DataFrame(imputer_rain.fit_transform(X_test),
                                        columns=X_test.columns,
                                        index=X_test.index)
            
            # Make predictions
            y_pred = model.predict(X_test_imputed)
            rmse, mae, r2 = evaluate_model(y_rain_test, y_pred, model_name, 'rain')
            print(f"{model_name}: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}")
            results[model_name] = {'rmse': rmse, 'mae': mae, 'r2': r2}

            # Create prediction vs actual plot
            plt.figure(figsize=(12, 6))
            plt.plot(y_rain_test.values[:50], label='Actual')
            plt.plot(y_pred[:50], label='Predicted', alpha=0.7)
            plt.title(f'{model_name} - Rainfall Forecasting')
            plt.xlabel('Time')
            plt.ylabel('Rainfall (mm)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'reports/{model_name}_rainfall_forecast.png')
            plt.close()
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")

# Find best models
best_temp_model = min(
    [(k, v['rmse']) for k, v in results.items() if 'temp' in k],
    key=lambda x: x[1]
)[0]

best_rain_model = min(
    [(k, v['rmse']) for k, v in results.items() if 'rain' in k],
    key=lambda x: x[1]
)[0]

print("\nBest Models Selected:")
print(f"Best Temperature Model: {best_temp_model}")
print(f"Best Rainfall Model: {best_rain_model}")

# Use best models for predictions
print("\nMaking Predictions with Best Models...")

# Temperature prediction
best_temp = models[best_temp_model]

# Impute missing values for temperature prediction
X_test_imputed_temp = pd.DataFrame(imputer.fit_transform(X_test),
                                columns=X_test.columns,
                                index=X_test.index)

y_temp_pred = best_temp.predict(X_test_imputed_temp)
rmse, mae, r2 = evaluate_model(y_temp_test, y_temp_pred, best_temp_model, 'temperature')
print(f"\nBest Temperature Model Metrics:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.4f}")

# Create temperature prediction plot
plt.figure(figsize=(12, 6))
plt.plot(y_temp_test.values[:100], label='Actual')
plt.plot(y_temp_pred[:100], label='Predicted', alpha=0.7)
plt.title(f'Best Model - Temperature Forecasting')
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('reports/best_model_temperature_forecast.png')
plt.close()

# Rainfall prediction
best_rain = models[best_rain_model]

# Impute missing values for rainfall prediction
X_test_imputed_rain = pd.DataFrame(imputer_rain.fit_transform(X_test),
                                columns=X_test.columns,
                                index=X_test.index)

y_rain_pred = best_rain.predict(X_test_imputed_rain)
rmse, mae, r2 = evaluate_model(y_rain_test, y_rain_pred, best_rain_model, 'rain')
print(f"\nBest Rainfall Model Metrics:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.4f}")

# Create rainfall prediction plot
plt.figure(figsize=(12, 6))
plt.plot(y_rain_test.values[:100], label='Actual')
plt.plot(y_rain_pred[:100], label='Predicted', alpha=0.7)
plt.title(f'Best Model - Rainfall Forecasting')
plt.xlabel('Time')
plt.ylabel('Rainfall (mm)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('reports/best_model_rainfall_forecast.png')
plt.close()

# Create rainfall prediction plot
plt.figure(figsize=(12, 6))
plt.plot(y_rain_test.values[:100], label='Actual')
plt.plot(y_rain_pred[:100], label='Predicted', alpha=0.7)
plt.title(f'Best Model - Rainfall Forecasting')
plt.xlabel('Time')
plt.ylabel('Rainfall (mm)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('reports/best_model_rainfall_forecast.png')
plt.close()

print("\nBest model predictions completed!")
print("Evaluation plots saved in 'reports/' directory")
print("Best models identified:")
print(f"Temperature: {best_temp_model}")
print(f"Rainfall: {best_rain_model}")
