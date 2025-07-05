# Weather Forecasting Regional and Seasonal Analysis
# Author: Kanav Benipuri
# Date: July 2025

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import pandas as pd

# Ensure reports directory exists
os.makedirs('reports', exist_ok=True)
from datetime import datetime

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("Starting Regional and Seasonal Analysis")
print("=" * 50)

# Load the data
import os

data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
df = pd.read_csv(os.path.join(data_dir, 'processed_weather_data.csv'))
print(f"\nLoaded processed data with shape: {df.shape}")

# Encode city as categorical before splitting
df['city'] = pd.Categorical(df['city'])
df['city_code'] = df['city'].cat.codes

# Create lag features if they don't exist
for col in ['temperature', 'rainfall']:
    for lag in [1, 7]:
        lag_col = f'{col}_lag{lag}'
        if lag_col not in df.columns:
            print(f"\nCreating lag feature: {lag_col}")
            # Create lag feature with forward fill for missing values
            df[lag_col] = df.groupby('city')[col].shift(lag).fillna(method='ffill')

# Create moving averages if they don't exist
for col in ['temperature', 'rainfall']:
    for window in [7, 30]:
        ma_col = f'{col}_ma{window}'
        if ma_col not in df.columns:
            print(f"\nCreating moving average: {ma_col}")
            # Create rolling mean with forward fill
            df[ma_col] = df.groupby('city')[col].rolling(window=window, min_periods=1).mean().reset_index(0, drop=True)
            df[ma_col] = df.groupby('city')[ma_col].fillna(method='ffill')

# Verify feature creation
print("\nFeature verification:")
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

print(f"\n Feature columns:")
print(f"Total features: {len(feature_cols)}")
print(f"Features: {feature_cols}")

# Check feature availability
missing_features = [col for col in feature_cols if col not in df.columns]
if missing_features:
    print(f" Warning: Missing features: {missing_features}")
    exit(1)

# Fill missing values with appropriate methods
print("\n Filling missing values:")
for col in feature_cols:
    if col in ['temperature_lag1', 'temperature_lag7', 'temperature_ma7', 'temperature_ma30',
               'temperature_range', 'temperature_variation']:
        df[col] = df.groupby('city')[col].transform('mean')
    elif col in ['rainfall_lag1', 'rainfall_lag7', 'rainfall_ma7', 'rainfall_ma30']:
        df[col] = df[col].fillna(0)
    elif col in ['is_monsoon']:
        df[col] = df[col].fillna(0)

# Verify no missing values
print("\n Verifying no missing values:")
print(f"Missing values after processing: {df.isna().sum().sum()}")

# Load the best models
print("\n Loading models...")
models = {}

# Get project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define model paths
model_paths = {
    'xgboost_temp': os.path.join(project_root, 'models', 'xgboost_temp_model.joblib'),
    'random_forest_rain': os.path.join(project_root, 'models', 'random_forest_rain_model.joblib'),
    'xgboost_rain': os.path.join(project_root, 'models', 'xgboost_rain_model.joblib'),
    'random_forest_temp': os.path.join(project_root, 'models', 'random_forest_temp_model.joblib'),
    'linear_regression_temp': os.path.join(project_root, 'models', 'linear_regression_temp_model.joblib'),
    'linear_regression_rain': os.path.join(project_root, 'models', 'linear_regression_rain_model.joblib')
}

# Verify model paths and load models
models = {}
for model_name, path in model_paths.items():
    try:
        print(f"\n Loading model: {model_name} from {path}")
        if not os.path.exists(path):
            print(f" Warning: Model file not found: {path}")
            continue
            
        model = joblib.load(path)
        models[model_name] = model
        print(f" Loaded {model_name}")
        print(f"Model type: {type(model)}")
        print(f"Model size: {os.path.getsize(path) / (1024 * 1024):.2f} MB")
    except Exception as e:
        print(f" Error loading {model_name}: {str(e)}")
        continue

# Check if we have the required models
required_models = ['xgboost_temp', 'random_forest_rain']
missing_models = [m for m in required_models if m not in models]
if missing_models:
    print(f" Error: Missing required models: {', '.join(missing_models)}")
    exit(1)

# Get the best models
xgboost_temp = models.get('xgboost_temp')
random_forest_rain = models.get('random_forest_rain')

if not xgboost_temp or not random_forest_rain:
    print(" Error: Required models not found after loading")
    exit(1)

# Verify model paths
for model_name, path in model_paths.items():
    if not os.path.exists(path):
        print(f" Warning: Model file not found: {path}")
    else:
        print(f"Model file exists: {path}")

# Load models with error handling
for model_name, path in model_paths.items():
    try:
        print(f"\n Loading model: {model_name} from {path}")
        models[model_name] = joblib.load(path)
        print(f"Loaded {model_name}")
        print(f"Model type: {type(models[model_name])}")
    except Exception as e:
        print(f" Error loading {model_name}: {str(e)}")
        continue

# Get the best models
xgboost_temp = models.get('xgboost_temp')
random_forest_rain = models.get('random_forest_rain')

if xgboost_temp is None or random_forest_rain is None:
    print(" Error: Required models not found. Please check model paths.")
    exit(1)

# Feature columns
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

print("\n Feature columns:")
print(f"Total features: {len(feature_cols)}")
print(f"Features: {feature_cols}")

# Verify data shape
print("\n Data verification:")
print(f"Total rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"Missing values: {df[feature_cols].isna().sum().sum()}")

def analyze_regional_performance():
    print("\n Analyzing Regional Performance...")
    
    # Get unique cities
    cities = df['city'].unique()
    print(f"Analyzing {len(cities)} cities: {', '.join(cities)}")
    
    # Create dataframes to store metrics with proper data types
    temp_metrics = pd.DataFrame({
        'city': pd.Series([], dtype='str'),
        'rmse': pd.Series([], dtype='float64'),
        'mae': pd.Series([], dtype='float64'),
        'r2': pd.Series([], dtype='float64')
    })
    
    rain_metrics = pd.DataFrame({
        'city': pd.Series([], dtype='str'),
        'rmse': pd.Series([], dtype='float64'),
        'mae': pd.Series([], dtype='float64'),
        'r2': pd.Series([], dtype='float64')
    })
    
    print("\n Initial metrics dataframes:")
    print(f"Temp metrics shape: {temp_metrics.shape}")
    print(f"Rain metrics shape: {rain_metrics.shape}")
    
    # Analyze each city
    for city in cities:
        print(f"\nCity: {city}")
        
        # Filter data for city
        city_df = df[df['city'] == city]
        
        # Encode city as categorical and get code
        city_df['city'] = pd.Categorical(city_df['city'])
        city_df['city_code'] = city_df['city'].cat.codes
        
        X_city = city_df[feature_cols]
        y_temp = city_df['temperature']
        y_rain = city_df['rainfall']
        
        # Temperature predictions
        temp_preds = xgboost_temp.predict(X_city)
        temp_rmse = np.sqrt(mean_squared_error(y_temp, temp_preds))
        temp_mae = mean_absolute_error(y_temp, temp_preds)
        temp_r2 = r2_score(y_temp, temp_preds)
        
        # Rainfall predictions
        rain_preds = random_forest_rain.predict(X_city)
        rain_rmse = np.sqrt(mean_squared_error(y_rain, rain_preds))
        rain_mae = mean_absolute_error(y_rain, rain_preds)
        rain_r2 = r2_score(y_rain, rain_preds)
        
        # Store metrics
        temp_metrics = pd.concat([
            temp_metrics,
            pd.DataFrame({
                'city': [city],
                'rmse': [temp_rmse],
                'mae': [temp_mae],
                'r2': [temp_r2]
            })
        ], ignore_index=True)
        
        rain_metrics = pd.concat([
            rain_metrics,
            pd.DataFrame({
                'city': [city],
                'rmse': [rain_rmse],
                'mae': [rain_mae],
                'r2': [rain_r2]
            })
        ], ignore_index=True)
        
        print(f"\n Metrics for city {city}:")
        print(f"Temp metrics shape: {temp_metrics.shape}")
        print(f"Rain metrics shape: {rain_metrics.shape}")
        print(f"Temp metrics sample: {temp_metrics.tail(1)}")
        print(f"Rain metrics sample: {rain_metrics.tail(1)}")
        
        print(f"Temperature Model - {city}:")
        print(f"RMSE: {temp_rmse:.2f}, MAE: {temp_mae:.2f}, R²: {temp_r2:.4f}")
        print(f"Rainfall Model - {city}:")
        print(f"RMSE: {rain_rmse:.2f}, MAE: {rain_mae:.2f}, R²: {rain_r2:.4f}")
    
    # Create regional performance plots
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    sns.barplot(data=temp_metrics, x='city', y='rmse')
    plt.title('Temperature Model RMSE by City')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 2)
    sns.barplot(data=temp_metrics, x='city', y='r2')
    plt.title('Temperature Model R² by City')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 3)
    sns.barplot(data=rain_metrics, x='city', y='rmse')
    plt.title('Rainfall Model RMSE by City')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 4)
    sns.barplot(data=rain_metrics, x='city', y='r2')
    plt.title('Rainfall Model R² by City')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('../reports/regional_performance.png')
    plt.close()
    
    # Save metrics as CSV
    temp_metrics.to_csv('../reports/temperature_regional_metrics.csv', index=False)
    rain_metrics.to_csv('../reports/rainfall_regional_metrics.csv', index=False)
    
    # Print summary statistics
    print("\n Regional Performance Summary:")
    print(f"\nTemperature Model:")
    print(f"Average RMSE: {temp_metrics['rmse'].mean():.2f}")
    print(f"Average MAE: {temp_metrics['mae'].mean():.2f}")
    print(f"Average R²: {temp_metrics['r2'].mean():.4f}")
    print(f"\nRainfall Model:")
    print(f"Average RMSE: {rain_metrics['rmse'].mean():.2f}")
    print(f"Average MAE: {rain_metrics['mae'].mean():.2f}")
    print(f"Average R²: {rain_metrics['r2'].mean():.4f}")

    # Analyze seasonal performance
    print("\n Analyzing Seasonal Performance...")
    
    # Define seasons
    seasons = {
        'Winter': [12, 1, 2],
        'Spring': [3, 4, 5],
        'Summer': [6, 7, 8],
        'Monsoon': [9, 10, 11]
    }
    
    # Create dataframes to store seasonal metrics
    temp_seasonal = pd.DataFrame({
        'season': pd.Series([], dtype='str'),
        'rmse': pd.Series([], dtype='float64'),
        'mae': pd.Series([], dtype='float64'),
        'r2': pd.Series([], dtype='float64')
    })
    
    rain_seasonal = pd.DataFrame({
        'season': pd.Series([], dtype='str'),
        'rmse': pd.Series([], dtype='float64'),
        'mae': pd.Series([], dtype='float64'),
        'r2': pd.Series([], dtype='float64')
    })
    
    # Analyze each season
    for season, months in seasons.items():
        print(f"\nSeason: {season}")
        
        # Filter data for season
        season_df = df[df['month'].isin(months)]
        
        # Encode city as categorical and get code
        season_df['city'] = pd.Categorical(season_df['city'])
        season_df['city_code'] = season_df['city'].cat.codes
        
        X_season = season_df[feature_cols]
        y_temp = season_df['temperature']
        y_rain = season_df['rainfall']
        
        # Temperature predictions
        temp_preds = xgboost_temp.predict(X_season)
        temp_rmse = np.sqrt(mean_squared_error(y_temp, temp_preds))
        temp_mae = mean_absolute_error(y_temp, temp_preds)
        temp_r2 = r2_score(y_temp, temp_preds)
        
        # Rainfall predictions
        rain_preds = random_forest_rain.predict(X_season)
        rain_rmse = np.sqrt(mean_squared_error(y_rain, rain_preds))
        rain_mae = mean_absolute_error(y_rain, rain_preds)
        rain_r2 = r2_score(y_rain, rain_preds)
        
        # Store metrics
        temp_seasonal = pd.concat([
            temp_seasonal,
            pd.DataFrame({
                'season': [season],
                'rmse': [temp_rmse],
                'mae': [temp_mae],
                'r2': [temp_r2]
            })
        ], ignore_index=True)
        
        rain_seasonal = pd.concat([
            rain_seasonal,
            pd.DataFrame({
                'season': [season],
                'rmse': [rain_rmse],
                'mae': [rain_mae],
                'r2': [rain_r2]
            })
        ], ignore_index=True)
        
        print(f"\n Metrics for season {season}:")
        print(f"Temp seasonal shape: {temp_seasonal.shape}")
        print(f"Rain seasonal shape: {rain_seasonal.shape}")
        print(f"Temp seasonal sample: {temp_seasonal.tail(1)}")
        print(f"Rain seasonal sample: {rain_seasonal.tail(1)}")
        
        print(f"Temperature Model - {season}:")
        print(f"RMSE: {temp_rmse:.2f}, MAE: {temp_mae:.2f}, R²: {temp_r2:.4f}")
        print(f"Rainfall Model - {season}:")
        print(f"RMSE: {rain_rmse:.2f}, MAE: {rain_mae:.2f}, R²: {rain_r2:.4f}")
    
    # Create seasonal performance plots
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    sns.barplot(data=temp_seasonal, x='season', y='rmse')
    plt.title('Temperature Model RMSE by Season')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 2)
    sns.barplot(data=temp_seasonal, x='season', y='r2')
    plt.title('Temperature Model R² by Season')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 3)
    sns.barplot(data=rain_seasonal, x='season', y='rmse')
    plt.title('Rainfall Model RMSE by Season')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 4)
    sns.barplot(data=rain_seasonal, x='season', y='r2')
    plt.title('Rainfall Model R² by Season')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('../reports/seasonal_performance.png')
    plt.close()
    
    # Save seasonal metrics as CSV
    temp_seasonal.to_csv('../reports/temperature_seasonal_metrics.csv', index=False)
    rain_seasonal.to_csv('../reports/rainfall_seasonal_metrics.csv', index=False)
    
    # Print seasonal summary statistics
    print("\n Seasonal Performance Summary:")
    print(f"\nTemperature Model:")
    print(f"Average RMSE: {temp_seasonal['rmse'].mean():.2f}")
    print(f"Average MAE: {temp_seasonal['mae'].mean():.2f}")
    print(f"Average R²: {temp_seasonal['r2'].mean():.4f}")
    print(f"\nRainfall Model:")
    print(f"Average RMSE: {rain_seasonal['rmse'].mean():.2f}")
    print(f"Average MAE: {rain_seasonal['mae'].mean():.2f}")
    print(f"Average R²: {rain_seasonal['r2'].mean():.4f}")
    
    # Print summary statistics
    print("\n Regional Performance Summary:")
    print(f"Best Temperature Model City: {temp_metrics.loc[temp_metrics['rmse'].idxmin()]['city']}")
    print(f"Best Rainfall Model City: {rain_metrics.loc[rain_metrics['rmse'].idxmin()]['city']}")
    print(f"Worst Temperature Model City: {temp_metrics.loc[temp_metrics['rmse'].idxmax()]['city']}")
    print(f"Worst Rainfall Model City: {rain_metrics.loc[rain_metrics['rmse'].idxmax()]['city']}")
    
    return temp_metrics, rain_metrics

def analyze_seasonal_performance():
    print("\n Analyzing Seasonal Performance...")
    
    # Define seasons
    seasons = {
        'Winter': [12, 1, 2],
        'Spring': [3, 4, 5],
        'Summer': [6, 7, 8],
        'Monsoon': [9, 10, 11]
    }
    
    # Create dataframes to store metrics with proper data types
    temp_seasonal = pd.DataFrame({
        'season': pd.Series([], dtype='str'),
        'rmse': pd.Series([], dtype='float64'),
        'mae': pd.Series([], dtype='float64'),
        'r2': pd.Series([], dtype='float64')
    })
    
    rain_seasonal = pd.DataFrame({
        'season': pd.Series([], dtype='str'),
        'rmse': pd.Series([], dtype='float64'),
        'mae': pd.Series([], dtype='float64'),
        'r2': pd.Series([], dtype='float64')
    })
    
    # Analyze each season
    for season, months in seasons.items():
        print(f"\nSeason: {season}")
        
        # Filter data for season
        season_df = df[df['month'].isin(months)]
        
        # Ensure city_code is included
        X_season = season_df[feature_cols]
        y_temp = season_df['temperature']
        y_rain = season_df['rainfall']
        
        # Temperature predictions
        temp_preds = xgboost_temp.predict(X_season)
        temp_rmse = np.sqrt(mean_squared_error(y_temp, temp_preds))
        temp_mae = mean_absolute_error(y_temp, temp_preds)
        temp_r2 = r2_score(y_temp, temp_preds)
        
        # Rainfall predictions
        rain_preds = random_forest_rain.predict(X_season)
        rain_rmse = np.sqrt(mean_squared_error(y_rain, rain_preds))
        rain_mae = mean_absolute_error(y_rain, rain_preds)
        rain_r2 = r2_score(y_rain, rain_preds)
        
        # Store metrics
        temp_seasonal = pd.concat([
            temp_seasonal,
            pd.DataFrame({
                'season': [season],
                'rmse': [temp_rmse],
                'mae': [temp_mae],
                'r2': [temp_r2]
            })
        ], ignore_index=True)
        
        rain_seasonal = pd.concat([
            rain_seasonal,
            pd.DataFrame({
                'season': [season],
                'rmse': [rain_rmse],
                'mae': [rain_mae],
                'r2': [rain_r2]
            })
        ], ignore_index=True)
        
        print(f"\n Metrics for season {season}:")
        print(f"Temp seasonal shape: {temp_seasonal.shape}")
        print(f"Rain seasonal shape: {rain_seasonal.shape}")
        print(f"Temp seasonal sample: {temp_seasonal.tail(1)}")
        print(f"Rain seasonal sample: {rain_seasonal.tail(1)}")
        
        print(f"Temperature Model - {season}:")
        print(f"RMSE: {temp_rmse:.2f}, MAE: {temp_mae:.2f}, R²: {temp_r2:.4f}")
        print(f"Rainfall Model - {season}:")
        print(f"RMSE: {rain_rmse:.2f}, MAE: {rain_mae:.2f}, R²: {rain_r2:.4f}")
    
    # Create seasonal performance plots
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    sns.barplot(data=temp_seasonal, x='season', y='rmse')
    plt.title('Temperature Model RMSE by Season')
    
    plt.subplot(2, 2, 2)
    sns.barplot(data=temp_seasonal, x='season', y='r2')
    plt.title('Temperature Model R² by Season')
    
    plt.subplot(2, 2, 3)
    sns.barplot(data=rain_seasonal, x='season', y='rmse')
    plt.title('Rainfall Model RMSE by Season')
    
    plt.subplot(2, 2, 4)
    sns.barplot(data=rain_seasonal, x='season', y='r2')
    plt.title('Rainfall Model R² by Season')
    
    plt.tight_layout()
    plt.savefig('reports/seasonal_performance.png')
    plt.close()
    
    # Save metrics as CSV
    temp_seasonal.to_csv('reports/temperature_seasonal_metrics.csv', index=False)
    rain_seasonal.to_csv('reports/rainfall_seasonal_metrics.csv', index=False)
    
    # Print summary statistics
    print("\n Seasonal Performance Summary:")
    print(f"Best Temperature Model Season: {temp_seasonal.loc[temp_seasonal['rmse'].idxmin()]['season']}")
    print(f"Best Rainfall Model Season: {rain_seasonal.loc[rain_seasonal['rmse'].idxmin()]['season']}")
    print(f"Worst Temperature Model Season: {temp_seasonal.loc[temp_seasonal['rmse'].idxmax()]['season']}")
    print(f"Worst Rainfall Model Season: {rain_seasonal.loc[rain_seasonal['rmse'].idxmax()]['season']}")
    
    return temp_seasonal, rain_seasonal

# Run analysis
regional_temp_metrics, regional_rain_metrics = analyze_regional_performance()
seasonal_temp_metrics, seasonal_rain_metrics = analyze_seasonal_performance()

print("\n Analysis completed!")
print("Regional performance plots saved in 'reports/regional_performance.png'")
print("Seasonal performance plots saved in 'reports/seasonal_performance.png'")
print("Detailed metrics saved as CSV files in 'reports/' directory")
