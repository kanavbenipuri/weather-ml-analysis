# Weather Forecasting Project - India

## Overview
This project aims to build and compare machine learning models for weather forecasting in India using publicly available datasets. The goal is to analyze historical weather data, preprocess it, perform exploratory data analysis (EDA), and develop predictive models to forecast key weather parameters. The project will also compare the performance of different models and provide visualizations and reports.

## Features
- Data acquisition from multiple sources (IMD, ERA5, Kaggle)
- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Implementation of baseline and advanced machine learning models
- Model evaluation and comparison
- Final reporting with visualizations

## Quick Start
1. **Clone the repository**
   ```bash
   git clone <repo-url>
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```
4. **Open and run the notebooks in the `notebooks/` directory**

## Project Structure
```
ashoka/
├── README.md          # Project documentation
├── data/              # Raw and processed datasets
│   ├── raw/          # Original downloaded datasets
│   └── processed/     # Cleaned and preprocessed data
├── notebooks/         # Python scripts for analysis
│   ├── 01_data_preprocessing.py    # Data cleaning and preprocessing
│   ├── 02_model_training.py        # Model training and hyperparameter tuning
│   ├── 03_best_model_prediction.py # Best model selection and prediction
│   ├── 04_model_analysis.py        # Model performance analysis and visualization
│   ├── 05_regional_seasonal_analysis.py # Regional and seasonal pattern analysis
│   ├── 06_performance_summary.py   # Final performance summary and comparison
│   └── reports/                    # Notebooks-generated reports and visualizations
├── models/            # Trained model files
│   ├── xgboost_temp_model.joblib    # Temperature prediction model
│   └── random_forest_rain_model.joblib  # Rainfall prediction model
├── reports/           # Final reports and visualizations
│   ├── temperature_model_analysis.png
│   ├── temperature_feature_importance.png
│   ├── rainfall_model_analysis.png
│   └── rainfall_feature_importance.png
├── requirements.txt   # Python dependencies
├── download_dataset.py # Script to download weather datasets
├── unzip_dataset.py    # Script to unzip downloaded datasets
└── venv/              # Python virtual environment
```

## Project Workflow

### 1. Data Preparation
- **Data Acquisition:**
  - Download historical weather data from IMD, ERA5, and Kaggle
  - Store raw data in `data/raw/`

- **Data Preprocessing:**
  - Clean and preprocess data using `01_data_preprocessing.ipynb`
  - Handle missing values, outliers, and data inconsistencies
  - Create lag features and moving averages
  - Encode categorical variables
  - Save processed data in `data/processed/`

### 2. Model Development
- **Feature Engineering:**
  - Create temporal features (month, day of year)
  - Generate lag features (temperature_lag1, rainfall_lag7, etc.)
  - Calculate moving averages (temperature_ma7, rainfall_ma30)
  - Add seasonal indicators

- **Model Training:**
  - Train multiple models using `02_model_training.ipynb`
  - Implement Linear Regression, Random Forest, and XGBoost
  - Perform hyperparameter tuning
  - Save best models in `models/`

### 3. Model Analysis
- **Performance Analysis:**
  - Evaluate model performance using `04_model_analysis.ipynb`
  - Calculate metrics (RMSE, MAE, R²)
  - Generate visualizations
  - Compare model performance

- **Regional Analysis:**
  - Analyze regional weather patterns using `05_regional_seasonal_analysis.ipynb`
  - Study seasonal variations
  - Compare different regions

### 4. Final Results
- **Best Model Selection:**
  - Select best performing models using `03_best_model_prediction.ipynb`
  - Validate on unseen data
  - Generate final predictions

- **Performance Summary:**
  - Create comprehensive summary using `06_performance_summary.ipynb`
  - Document findings
  - Generate final reports and visualizations
  - Save in `reports/`

## Data Sources
- IMD (Indian Meteorological Department)
- ERA5 (Copernicus Climate Data Store)
- Kaggle weather datasets

## Models to Implement
1. **Linear Regression** (baseline)
2. **Random Forest**
3. **XGBoost**

## Getting Started

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- Required packages listed in `requirements.txt`

### Installation
1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd ashoka
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

1. **Data Preparation**
   - Place raw data files in `data/raw/`
   - Run `01_data_preprocessing.ipynb` to clean and preprocess data

2. **Model Development**
   - Run `02_model_training.ipynb` to train models
   - Run `03_best_model_prediction.ipynb` to select best models

3. **Analysis**
   - Run `04_model_analysis.ipynb` to analyze model performance
   - Run `05_regional_seasonal_analysis.ipynb` for regional analysis
   - Run `06_performance_summary.ipynb` for final results

### Expected Output
- Model performance metrics
- Visualizations of predictions vs actual values
- Feature importance plots
- Regional weather patterns analysis
- Seasonal variation analysis

---
For questions or contributions, please open an issue or submit a pull request.
