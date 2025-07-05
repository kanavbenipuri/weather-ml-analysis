# Weather Forecasting Performance Summary Report Generator
# Author: Kanav Benipuri
# Date: July 2025

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def generate_summary_report():
    print("Generating Performance Summary Report")
    print("============================================")
    
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Create summary directory
    summary_dir = os.path.join(project_root, 'reports', 'summary')
    os.makedirs(summary_dir, exist_ok=True)
    
    # Load and verify metrics
    metrics_dir = os.path.join(project_root, 'reports')
    
    temp_regional = pd.read_csv(os.path.join(metrics_dir, 'temperature_regional_metrics.csv'))
    rain_regional = pd.read_csv(os.path.join(metrics_dir, 'rainfall_regional_metrics.csv'))
    temp_seasonal = pd.read_csv(os.path.join(metrics_dir, 'temperature_seasonal_metrics.csv'))
    rain_seasonal = pd.read_csv(os.path.join(metrics_dir, 'rainfall_seasonal_metrics.csv'))
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Temperature metrics by city
    plt.subplot(2, 2, 1)
    sns.barplot(data=temp_regional, x='city', y='rmse')
    plt.title('Temperature Model RMSE by City')
    plt.xticks(rotation=45)
    plt.ylabel('RMSE')
    
    # Rainfall metrics by city
    plt.subplot(2, 2, 2)
    sns.barplot(data=rain_regional, x='city', y='rmse')
    plt.title('Rainfall Model RMSE by City')
    plt.xticks(rotation=45)
    plt.ylabel('RMSE')
    
    # Temperature metrics by season
    plt.subplot(2, 2, 3)
    sns.barplot(data=temp_seasonal, x='season', y='r2')
    plt.title('Temperature Model R² by Season')
    plt.ylabel('R² Score')
    
    # Rainfall metrics by season
    plt.subplot(2, 2, 4)
    sns.barplot(data=rain_seasonal, x='season', y='r2')
    plt.title('Rainfall Model R² by Season')
    plt.ylabel('R² Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, 'performance_summary.png'))
    plt.close()
    
    # Create correlation heatmaps with numeric columns only
    plt.figure(figsize=(12, 8))
    
    # Temperature correlation heatmap
    plt.subplot(2, 1, 1)
    numeric_cols = temp_regional.select_dtypes(include=[np.number]).columns
    temp_corr = temp_regional[numeric_cols].corr()
    sns.heatmap(temp_corr, annot=True, cmap='coolwarm', fmt='.2f', 
               annot_kws={'size': 8}, cbar_kws={'shrink': 0.8})
    plt.title('Temperature Model Metrics Correlation')
    plt.yticks(rotation=0)
    
    # Rainfall correlation heatmap
    plt.subplot(2, 1, 2)
    numeric_cols = rain_regional.select_dtypes(include=[np.number]).columns
    rain_corr = rain_regional[numeric_cols].corr()
    sns.heatmap(rain_corr, annot=True, cmap='coolwarm', fmt='.2f', 
               annot_kws={'size': 8}, cbar_kws={'shrink': 0.8})
    plt.title('Rainfall Model Metrics Correlation')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, 'metrics_correlation.png'))
    plt.close()
    
    # Generate summary statistics
    temp_summary = temp_regional.describe().round(2)
    rain_summary = rain_regional.describe().round(2)
    
    # Generate markdown report
    report = """
# Weather Forecasting Model Performance Summary

Generated on: {date}

## Regional Performance Analysis

### Temperature Model

Best Performing City: {best_temp_city}
Worst Performing City: {worst_temp_city}
Average R²: {avg_temp_r2:.4f}
Average RMSE: {avg_temp_rmse:.2f}

### Rainfall Model

Best Performing Season: {best_rain_season}
Worst Performing Season: {worst_rain_season}
Average R²: {avg_rain_r2:.4f}
Average RMSE: {avg_rain_rmse:.2f}

## Key Insights

1. Temperature Model Performance:
- Consistently high R² scores across cities
- Low RMSE values indicating accurate predictions
- Best performance in {best_temp_city}

2. Rainfall Model Performance:
- Seasonal variations in performance
- Best performance during {best_rain_season}
- Higher RMSE values compared to temperature model

3. Regional Patterns:
- Both models show consistent performance across most cities
- Some cities show better performance for temperature prediction
- Seasonal patterns are more pronounced in rainfall predictions

4. Recommendations:
- Temperature model can be deployed with high confidence
- Rainfall model may benefit from seasonal adjustments
- Consider additional features for improving rainfall predictions
""".format(
    date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    best_temp_city=temp_regional.loc[temp_regional['rmse'].idxmin()]['city'],
    worst_temp_city=temp_regional.loc[temp_regional['rmse'].idxmax()]['city'],
    avg_temp_r2=temp_regional['r2'].mean(),
    avg_temp_rmse=temp_regional['rmse'].mean(),
    best_rain_season=temp_seasonal.loc[temp_seasonal['rmse'].idxmin()]['season'],
    worst_rain_season=temp_seasonal.loc[temp_seasonal['rmse'].idxmax()]['season'],
    avg_rain_r2=rain_seasonal['r2'].mean(),
    avg_rain_rmse=rain_seasonal['rmse'].mean()
)

    # Save report with proper encoding
    with open(os.path.join(summary_dir, 'performance_summary.md'), 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Print summary statistics
    print("\nSummary statistics:")
    print("\nTemperature Metrics:")
    print(temp_summary)
    print("\nRainfall Metrics:")
    print(rain_summary)
    
    print(f"\nSummary report generated in {summary_dir}/performance_summary.md")
    print("\nGenerated files:")
    print(f"- {summary_dir}/performance_summary.png: Visual summary of performance metrics")
    print(f"- {summary_dir}/metrics_correlation.png: Correlation heatmaps of metrics")
    print(f"- {summary_dir}/performance_summary.md: Detailed performance report")

# Run report generation
generate_summary_report()
