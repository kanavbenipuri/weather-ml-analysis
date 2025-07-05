# Weather Forecasting Project - Data Preprocessing
# Author: Kanav Benipuri
# Date: July 2025

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("  Weather Forecasting Project - Data Preprocessing")
print("=" * 60)

# =============================================================================
# STEP 1: DATA LOADING
# =============================================================================

def load_weather_data():
    """Load and merge weather data from multiple cities"""
    print("\n Loading weather data...")
    
    # Get data directory
    data_dir = Path('data/Temperature_And_Precipitation_Cities_IN')
    
    # List of cities with their file names
    city_files = {
        'Bangalore': 'Bangalore_1990_2022_BangaloreCity.csv',
        'Chennai': 'Chennai_1990_2022_Madras.csv',
        'Delhi': 'Delhi_NCR_1990_2022_Safdarjung.csv',
        'Mumbai': 'Mumbai_1990_2022_Santacruz.csv',
        'Lucknow': 'Lucknow_1990_2022.csv',
        'Jodhpur': 'Rajasthan_1990_2022_Jodhpur.csv',
        'Bhubaneswar': 'weather_Bhubhneshwar_1990_2022.csv',
        'Rourkela': 'weather_Rourkela_2021_2022.csv'
    }
    
    data = []
    
    # Load each city's data
    for city, filename in city_files.items():
        try:
            file_path = data_dir / filename
            df = pd.read_csv(file_path)
            
            # Rename columns to match the actual data structure
            df.columns = ['date', 'tavg', 'tmin', 'tmax', 'prcp']
            
            # Calculate temperature from min and max if avg is missing
            df['temperature'] = df['tavg'].fillna(
                (df['tmin'] + df['tmax']) / 2
            )
            
            # Drop rows with all temperature values missing
            df = df.dropna(subset=['temperature'], how='all')
            
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
            
            # Add city column
            df['city'] = city
            
            data.append(df)
            print(f"Loaded data for {city}")
        except Exception as e:
            print(f"Error loading {city} data: {e}")
    # Combine all city data
    df = pd.concat(data, ignore_index=True)
    
    # Sort by city and date
    df.sort_values(['city', 'date'], inplace=True)
    
    # Reset index to avoid duplicate city column
    df = df.reset_index(drop=True)
    
    # Fill missing values with forward fill
    df = df.groupby('city', group_keys=False).apply(lambda x: x.ffill().bfill())
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df

# =============================================================================
# STEP 2: DATA CLEANING AND PREPARATION
# =============================================================================

def prepare_data(df):
    """Clean and prepare the data for analysis"""
    print("\nData Cleaning and Preparation...")
    
    # Basic statistics
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Add derived features
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_year'] = df['date'].dt.dayofyear
    df['season'] = df['month'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
    })
    
    # Create additional temperature features
    df['temperature_range'] = df['tmax'] - df['tmin']
    df['temperature_variation'] = df['temperature_range'] / df['tavg']
    
    # Create rainfall categories
    df['rainfall_category'] = pd.cut(
        df['rainfall'],
        bins=[0, 0.1, 1, 5, 10, 20, 50, np.inf],
        labels=['None', 'Light', 'Moderate', 'Heavy', 'Very Heavy', 'Extremely Heavy', 'Extreme']
    )
    
    return df

# =============================================================================
# STEP 3: EXPLORATORY DATA ANALYSIS
# =============================================================================

def exploratory_analysis(df):
    """Perform exploratory data analysis"""
    print("\nStarting Exploratory Data Analysis...")
    
    # Create visualisations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Weather Data Overview - India', fontsize=16, fontweight='bold')
    
    # Temperature trends by city
    axes[0, 0].set_title('Temperature Trends by City')
    for city in df['city'].unique():
        city_data = df[df['city'] == city]
        monthly_temp = city_data.groupby('month')['temperature'].mean()
        axes[0, 0].plot(monthly_temp.index, monthly_temp.values, marker='o', label=city)
    axes[0, 0].set_xlabel('Month')
    axes[0, 0].set_ylabel('Temperature (°C)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Rainfall patterns
    axes[0, 1].set_title('Monthly Rainfall Patterns')
    monthly_rainfall = df.groupby(['month', 'city'])['rainfall'].mean().unstack()
    monthly_rainfall.plot(kind='bar', ax=axes[0, 1])
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('Rainfall (mm)')
    axes[0, 1].legend(title='City', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Correlation heatmap
    axes[1, 0].set_title('Feature Correlation Matrix')
    numeric_cols = ['temperature', 'rainfall']
    corr_matrix = df[numeric_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 0])
    
    # Seasonal patterns
    axes[1, 1].set_title('Seasonal Weather Patterns')
    seasonal_data = df.groupby('season')[['temperature', 'rainfall']].mean()
    seasonal_data.plot(kind='bar', ax=axes[1, 1])
    axes[1, 1].set_xlabel('Season')
    axes[1, 1].set_ylabel('Average Values')
    axes[1, 1].legend()
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# STEP 4: FEATURE ENGINEERING
# =============================================================================

def create_features(df):
    """Create additional features for better model performance"""
    print("\nFeature Engineering...")
    
    df_features = df.copy()
    
    # Lag features (previous day's weather)
    for col in ['temperature', 'prcp']:
        df_features[f'{col}_lag1'] = df_features.groupby('city')[col].shift(1)
        df_features[f'{col}_lag7'] = df_features.groupby('city')[col].shift(7)
    
    # Rolling averages
    df_features['temperature_ma7'] = df_features.groupby('city')['temperature'].rolling(7).mean().reset_index(0, drop=True)
    df_features['temperature_ma30'] = df_features.groupby('city')['temperature'].rolling(30).mean().reset_index(0, drop=True)
    df_features['prcp_ma7'] = df_features.groupby('city')['prcp'].rolling(7).mean().reset_index(0, drop=True)
    df_features['prcp_ma30'] = df_features.groupby('city')['prcp'].rolling(30).mean().reset_index(0, drop=True)
    
    # Seasonal indicators
    df_features['is_monsoon'] = df_features['month'].isin([6, 7, 8, 9]).astype(int)
    
    # Save feature engineered data
    df_features.to_csv('data/processed_weather_data.csv', index=False)
    print(f"Saved processed data to data/processed_weather_data.csv")
    
    return df_features

if __name__ == "__main__":
    # Load the data
    df = load_weather_data()
    
    # Prepare the data
    df = prepare_data(df)
    
    # Perform exploratory analysis
    exploratory_analysis(df)
    
    # Create features
    df_features = create_features(df)
    
    # Save the processed data
    df_features.to_csv('data/processed_weather_data.csv', index=False)
    print(f"\nData preprocessing completed!")
    print(f"Final dataset shape: {df_features.shape}")
    print("\nNext steps:")
    print("1. Model training (Linear Regression, Random Forest, XGBoost)")
    print("2. Model evaluation and comparison")
    print("3. Final report generation")


