# Leave Management System - Code Explanation Document

## Overview
This document provides comprehensive explanations of all files in the Leave Management System project. The system uses machine learning to forecast employee leave patterns and provides multiple dashboards for visualization and analysis.

---

## 📋 Table of Contents
1. [Configuration & Setup Files](#configuration--setup-files)
2. [Core Application Files](#core-application-files)
3. [Model Training & Retraining Files](#model-training--retraining-files)
4. [Dashboard & Frontend Files](#dashboard--frontend-files)
5. [Jupyter Notebook](#jupyter-notebook)
6. [Data Files](#data-files)

---

## Configuration & Setup Files

### **requirements.txt**
**Location:** `requirements.txt`

**Purpose:** Defines all Python dependencies needed to run the Leave Management System.

**Contents:**
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `matplotlib` - Static plotting library
- `seaborn` - Statistical data visualization
- `scikit-learn` - Machine learning algorithms (RandomForest, GradientBoosting)
- `xgboost` - Optimized gradient boosting library
- `plotly` - Interactive visualization library
- `holidays` - Holiday calendar for India (used for feature engineering)
- `shap` - SHAP values for model explainability
- `tensorflow` - Deep learning framework
- `joblib` - Model serialization and persistence
- `streamlit` - Web app framework for data apps
- `flask` - Lightweight web framework for REST API dashboard
- `openpyxl` - Excel file handling
- `duckdb` - SQL query engine for CSV files
- `nbformat` - Jupyter Notebook format handling

**Usage:** Install all dependencies with `pip install -r requirements.txt`

---

### **start_dashboard.bat**
**Location:** `start_dashboard.bat`

**Purpose:** Batch script to start the Flask-based web dashboard on Windows systems.

**Key Functions:**
- Checks if Python is installed and available in PATH
- Verifies virtual environment exists at `.venv\Scripts\activate.bat`
- Confirms Flask and other dependencies are installed
- Displays installed versions of Flask, Plotly, and Pandas
- Launches Flask application on `http://localhost:5000`
- Shows startup information and port details

**Usage:** 
```bash
./start_dashboard.bat
```

---

## Core Application Files

### **streamlit_app.py**
**Location:** `streamlit_app.py`

**Purpose:** Main interactive Streamlit application for leave forecasting with predictive models and visualizations.

**Key Components:**

#### Constants & Configuration
- `DATE_COLUMNS`: List of date fields to parse ("From Date", "To Date", "Applied On", "Approved On")
- `DROP_COLUMNS`: Unnecessary columns to remove ("Comments", "Approver Comments")
- `TEXT_FILL_COLUMNS`: Categorical columns to fill with "Unknown"
- `TARGET_COLUMN`: The variable being predicted ("Leave_Count")
- `MASTER_FORECAST_BUFFER_DAYS`: 730 (2 years of data for training)
- `FUTURE_FORECAST_WINDOW_DAYS`: 60 (forecasting window)

#### Key Functions

**`get_project_paths(base_dir)`**
- Constructs file paths for data and model artifacts
- Returns dictionary with paths to: project directory, data CSV, employee master Excel, model pickle, metadata pickle

**`clean_leave_data(raw_frame)`**
- Removes duplicate and invalid entries
- Converts date strings to datetime objects
- Filters for only "Approved" leave records
- Removes unwanted columns and fills missing values
- Validates that dates are logical (To Date >= From Date)

**`add_calendar_features(frame, holiday_calendar)`**
- Adds time-based features: day_of_week, month, week_of_year, quarter
- Creates boolean flags: is_weekend, is_month_start, is_month_end
- Applies trigonometric encoding for cyclical features (month_sin, month_cos)
- Identifies holidays, long weekends, and festival names
- Creates special day indicators (is_monday, is_friday, is_post_holiday)

**`add_history_features(frame, target_col)`**
- Creates lag features: leave_lag_1, leave_lag_7, leave_lag_14, leave_lag_30
- Calculates rolling statistics: rolling_mean_7, rolling_std_7, rolling_mean_30
- Captures temporal patterns and historical trends

**`expand_leave_records(df)`**
- Converts date ranges (From Date → To Date) into individual daily records
- Adds weekly, monthly aggregations
- Useful for daily forecasting at the employee level

**`build_feature_dataset(frame, holiday_calendar, target_col)`**
- Orchestrates all feature engineering steps
- Returns complete feature matrix ready for model training

**Key Workflow:**
1. Loads leave data and employee master
2. Cleans and validates data
3. Expands leave records to daily granularity
4. Engineers calendar and history features
5. Loads pre-trained model and metadata
6. Generates 60-day forward forecast
7. Displays interactive Streamlit dashboard with multiple tabs

**Visualization Tabs:**
- **Forecast:** Model evaluation metrics, feature importance, actual vs predicted
- **Intelligence:** Leave trends, cost center analysis, leave type patterns
- **Special Leaves:** Special leave and comp-off analysis
- **Cost Centre:** Risk analysis by business unit
- **Leave Patterns:** Analysis by leave reason

---

### **streamlit_sql_visualization.py**
**Location:** `streamlit_sql_visualization.py`

**Purpose:** Pure SQL-based analytics dashboard using DuckDB for CSV querying without ML models.

**Key Differences from streamlit_app.py:**
- No model artifacts or forecasting
- Uses DuckDB SQL engine instead of Pandas for faster querying
- Dynamic date filtering with validation
- Real-time insights from raw data

**Key Components:**

#### SQL Query Functions
- `build_daily_summary_query()`: Aggregates daily leave with planned/unplanned breakdown
- Additional query builders for cost centre, leave reason, and employee analyses

**Date Filtering:**
- Start date must be >= today
- End date must be within 60 days from start date
- Validates date ranges before executing queries

**Usage:** 
```bash
streamlit run streamlit_sql_visualization.py
```

---

### **web_dashboard.py**
**Location:** `web_dashboard.py`

**Purpose:** Flask-based REST API web application serving interactive HTML dashboard with Plotly visualizations.

**Key Components:**

#### Configuration
- `BASE_DIR`: Project root directory
- `CSV_PATH`: Path to Combined_All_Leave_Data.csv
- `ARTIFACT_DIR`: Path to model artifacts directory
- `FUTURE_PREDICTION_DAYS`: 60-day forecast window

#### Key Functions

**`load_clean_data()`**
- Reads and validates leave data CSV
- Converts date columns to datetime
- Filters for "Approved" status only
- Normalizes numeric and text fields
- Validates date ranges

**`expand_leave_records(df)`**
- Same as streamlit_app.py
- Creates daily records from date ranges

**`load_forecast_artifacts()`**
- Finds latest model artifacts in artifacts/ directory
- Loads test metrics, predictions, feature importance
- Loads next 30-day and 60-day forecasts
- Returns dictionary of DataFrames

**`parse_artifact_ts(path)`**
- Extracts timestamp from artifact filename
- Used for sorting and versioning

**`latest_artifact(pattern)`**
- Finds most recent artifact matching glob pattern
- Returns most recent file by timestamp

#### Routes

**`@app.route("/", methods=["GET"])`** - Main Dashboard
- Loads and processes all leave data
- Generates forecast charts (model evaluation, feature importance, predictions)
- Generates intelligence charts (trends, risk analysis, patterns)
- Generates special leave and cost centre charts
- Renders HTML dashboard with multiple tabs
- Date range controls (today to 60 days ahead)

**Chart Types Generated:**
- Model performance metrics
- Feature importance comparison
- Actual vs predicted leave counts
- Future leave forecasts with confidence bands
- Daily employee leave trends
- Cost centre risk analysis
- Leave type pattern distribution
- Special leave types breakdown
- Top cost centres and reasons

**Usage:**
```bash
python web_dashboard.py
# Access at http://localhost:5000
```

---

## Model Training & Retraining Files

### **retrain_model.py**
**Location:** `retrain_model.py`

**Purpose:** Complete end-to-end model retraining pipeline. Trains multiple models, evaluates them, and deploys the best performer.

**Configuration Class:**
```python
@dataclass
class RetrainConfig:
    project_dir: Path           # Project root
    app_path: Path             # Path to streamlit_app.py
    artifacts_dir: Path        # Artifacts storage
    archive_dir: Path          # Old model archive
    production_model_path: Path # Active model
    production_metadata_path: Path # Model metadata
    as_of_date: str            # Training cutoff date
    forecast_horizon: int = 60 # Forecast days
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state: int = 42
```

**Key Functions:**

**`parse_args()`**
- Parses command-line arguments
- `--as-of-date`: Training data cutoff (YYYY-MM-DD)
- `--forecast-horizon`: Number of forward days to forecast

**`load_forecasting_namespace(app_path)`**
- Dynamically imports functions and constants from streamlit_app.py
- Reuses feature engineering logic from main app
- Avoids code duplication

**`infer_default_as_of_date(data_path)`**
- If no as_of_date provided, infers from data
- Uses min(today, max_approved_leave_date)
- Ensures training data is current

**`build_candidate_models(random_state)`**
- Creates 3 models: RandomForest, GradientBoosting, XGBoost
- Each configured with optimized hyperparameters
- Returns dictionary of model instances

**`build_sample_weights(frame, target_col)`**
- Assigns higher weights to high-leave days
- Extra weight for holidays and long weekends
- Weight range: 1.0 to 3.0 (3x for peak days)
- Prevents model from underestimating peak leave

**`build_walk_forward_splits(frame, feature_cols, target_col, min_train_rows, n_splits)`**
- Creates time-series cross-validation splits
- Prevents data leakage (train < validation < test)
- Adaptive split sizing based on data length
- Returns list of train/validation sets

**`evaluate_predictions(y_true, y_pred, model_name, ns)`**
- Calculates 6 metrics: MAE, RMSE, MAPE, R2, WAPE, SMAPE
- Returns metrics dictionary for comparison

**Model Selection Logic:**
1. Train each model on walk-forward splits
2. Evaluate on validation sets
3. Calculate weighted metrics across folds
4. Select model with best WAPE (Weighted Absolute Percentage Error)
5. Generate 30-day and 60-day forecasts
6. Calculate model balance metrics (overfitting, generalization, stability)
7. Save model, metadata, and feature importance

**Output Artifacts:**
- `leave_forecasting_model.pkl` - Production model
- `leave_forecasting_metadata.pkl` - Metadata including metrics, dates, features
- `leave_forecasting_[model]_test_metrics.csv` - Evaluation metrics by model
- `leave_forecasting_[model]_test_predictions.csv` - Predictions vs actual
- `leave_forecasting_[model]_feature_importance.csv` - Feature rankings
- `leave_forecast_next_30days_[timestamp].csv` - 30-day forecast
- `leave_forecast_next_60days_[timestamp].csv` - 60-day forecast

**Usage:**
```bash
python retrain_model.py --as-of-date 2025-12-31 --forecast-horizon 60
```

---

### **check_model.py**
**Location:** `check_model.py`

**Purpose:** Quick validation and summary report of the currently deployed production model.

**Key Functions:**

**Main Workflow:**
1. Loads metadata from `artifacts/leave_forecasting_metadata.pkl`
2. Validates metadata structure (checks all required keys)
3. Generates summary report with:
   - Best model name (RandomForest, GradientBoosting, or XGBoost)
   - Training period (start date to end date)
   - Test period (evaluation window)
   - Number of engineered features
   - Test WAPE (Weighted Absolute Percentage Error)
   - Overfitting Signal (0.0 = no overfitting)
   - Generalization Gap (difference between train and test metrics)
   - Stability Score (consistency across folds)
   - Next N-day forecast status (days generated)

**Output Example:**
```
MODEL TRAINING SUMMARY
=================================================================
Best Model: XGBoost
Training Period: 2023-01-01 -> 2025-12-31
Test Period: 2025-11-01 -> 2025-12-31
Features: 45 engineered features
Test WAPE: 12.35%
Overfitting Signal: 0.0245
Generalization Gap: 0.0320
Stability Score: 0.892
Next 60-Day Forecast: generated (60 days)
=================================================================
```

**Error Handling:**
- Checks if metadata file exists
- Validates metadata is a dictionary
- Checks for missing required keys
- Reports warnings without crashing

**Usage:**
```bash
python check_model.py
```

---

## Dashboard & Frontend Files

### **templates/dashboard.html**
**Location:** `templates/dashboard.html`

**Purpose:** HTML template for Flask web dashboard. Renders Jinja2 template with data passed from web_dashboard.py.

**Key Elements:**
- Responsive Bootstrap layout
- Plotly chart containers
- Date range filter controls
- Multiple tabs for different analyses:
  - Forecast Tab: Model metrics and predictions
  - Intelligence Tab: Trends and patterns
  - Special Leaves Tab: Special leave type analysis
  - Cost Centre Tab: Business unit analysis
  - Reasons Tab: Leave reason patterns
- Card components for key metrics:
  - Days covered
  - Average employees per day
  - Total leave days
  - Number of cost centres
- Dynamic table rendering based on data availability

**Jinja2 Template Variables:**
- `min_date`, `max_date`: Date range bounds
- `start_date`, `end_date`: Selected date range
- `context_date`: Reference date for analytics
- `cards`: Summary metrics
- `forecast_charts`: Plotly chart HTML strings
- `intelligence_charts`: Trend visualization
- `special_charts`: Special leave analysis
- `cost_charts`: Cost centre breakdowns
- `planned_charts`: Planned vs unplanned analysis
- `reason_charts`: Leave reason patterns
- `tables`: HTML table strings for summary tables

---

## Jupyter Notebook

### **End_to_End_ML_Lifecycle_Training.ipynb**
**Location:** `End_to_End_ML_Lifecycle_Training.ipynb`

**Purpose:** Comprehensive Jupyter notebook demonstrating the complete machine learning lifecycle for leave forecasting.

**Sections (9 Code Cells):**

**Cell 1-2: Data Loading & Exploration**
- Load leave data and employee master
- Initial data shape and info
- Missing value analysis

**Cell 3: Data Cleaning & Preprocessing**
- Handle missing values
- Convert data types
- Remove invalid records
- Validate date ranges

**Cell 4-5: Feature Engineering**
- Calendar features (day of week, month, holidays)
- History features (lags, rolling averages)
- Temporal encoding (sine/cosine for cyclical features)
- Interactive visualizations

**Cell 6: Exploratory Data Analysis**
- Statistical summaries
- Distribution plots
- Correlation analysis
- Seasonal patterns

**Cell 7: Model Training**
- Train RandomForest, GradientBoosting, XGBoost
- Cross-validation with walk-forward splits
- Hyperparameter tuning

**Cell 8: Model Evaluation**
- Compare metrics across models
- Feature importance visualization
- Residual analysis
- Error distribution

**Cell 9: Forecasting & Deployment**
- Generate future predictions
- Confidence intervals
- Save production artifacts
- Display forecast visualizations

**Outputs:**
- HTML plots (Plotly visualizations)
- Model performance comparisons
- Feature importance charts
- Forecast plots with confidence bands

---

## Data Files

### **Data/Combined_All_Leave_Data.csv**
**Location:** `Data/Combined_All_Leave_Data.csv`

**Purpose:** Main source data file containing all employee leave records.

**Expected Columns:**
- `EmpNo` - Employee number
- `From Date` - Leave start date
- `To Date` - Leave end date
- `Days` - Number of leave days
- `Leave Type` - Type of leave (Casual, Sick, Special, Comp-Off, etc.)
- `Department` - Employee's department
- `Cost Centre` - Business cost center
- `Leave Reason` - Reason for leave
- `Status` - Leave status (Approved, Rejected, Pending)
- `Applied On` - Application date
- `Approved On` - Approval date
- `Approver Comments` - Comments from approver
- `Type` - Planned or Un-Planned
- `Business Area` - Business area classification
- `Location` - Work location
- And other employee/organizational attributes

**Data Quality:**
- Only "Approved" records are used
- Duplicates are removed
- Invalid date ranges are filtered
- Numeric fields are validated

---

### **Employee Master - Feb 2026 Team Member.xlsx**
**Location:** `Employee Master - Feb 2026 Team Member.xlsx`

**Purpose:** Employee master data containing current employee information and organizational hierarchy.

**Sheets:**
- `Live` - Currently active employees
- `Left` - Employees who have left the organization

**Expected Columns:**
- `SAP Emp No` - SAP employee number
- `Name` - Employee name
- `Department` - Department assignment
- `Division` - Division classification
- `Current Designation` - Job title
- `Business Area` - Business area
- `Direct / Indirect` - Employment type
- `State` - Geographic state
- `Local/Non Local` - Local/Non-local status
- `Category` - Employee category
- `Sex` - Gender
- `Marrital Status` - Marital status
- `D.O.J` - Date of joining
- `D.O.L` - Date of leaving
- `Date of Retirement` - Retirement date
- `Years of Service` - Tenure
- `Age` - Employee age

**Usage:**
- Used to filter active employees
- Calculate active headcount by department
- Align forecasts with current workforce

---

### **artifacts/** Directory
**Location:** `artifacts/`

**Purpose:** Storage for trained models, metadata, and forecast outputs.

**Artifact Files:**

**Model Files:**
- `leave_forecasting_model.pkl` - Trained production model (serialized with joblib)
- `leave_forecasting_metadata.pkl` - Model metadata (training dates, metrics, features)

**Metrics Files (by timestamp):**
- `leave_forecasting_*_test_metrics.csv` - Evaluation metrics (MAE, RMSE, MAPE, R2, WAPE, SMAPE)
- `leave_forecasting_*_test_predictions.csv` - Model predictions vs actual values on test set
- `leave_forecasting_*_feature_importance.csv` - Feature importance rankings from model

**Forecast Files:**
- `leave_forecast_next_30days_*.csv` - 30-day forward forecast with confidence intervals
- `leave_forecast_next_60days_*.csv` - 60-day forward forecast

**Columns in Forecast CSVs:**
- `Date` - Forecast date
- `Predicted_Leave_Count` - Forecasted daily leave count
- `Lower_Bound` - Lower confidence interval
- `Upper_Bound` - Upper confidence interval

---

## Workflow Summary

### **End-to-End Process:**

```
1. DATA INGESTION
   └─ Combined_All_Leave_Data.csv → Clean and validate

2. FEATURE ENGINEERING
   ├─ Calendar features (holidays, weekends, festivals)
   ├─ History features (lags, rolling averages)
   └─ Employee master enrichment

3. MODEL TRAINING (retrain_model.py)
   ├─ Build walk-forward CV splits
   ├─ Train 3 models: RandomForest, GradientBoosting, XGBoost
   ├─ Evaluate on validation sets
   └─ Select best model → Save to artifacts/

4. FORECASTING
   ├─ Generate 30-day and 60-day forward forecasts
   ├─ Calculate confidence intervals
   └─ Save to artifacts/ directory

5. VISUALIZATION & DASHBOARDS
   ├─ Flask Dashboard (web_dashboard.py) → http://localhost:5000
   ├─ Streamlit App (streamlit_app.py) → Interactive predictions
   ├─ SQL Analytics (streamlit_sql_visualization.py) → Data exploration
   └─ Check Model Status (check_model.py) → Quick validation

6. MODEL MONITORING
   └─ check_model.py → Reports model health, metrics, overfitting
```

### **Key Metrics:**
- **WAPE** - Weighted Absolute Percentage Error (primary metric)
- **MAPE** - Mean Absolute Percentage Error
- **RMSE** - Root Mean Squared Error
- **MAE** - Mean Absolute Error
- **R²** - Coefficient of determination
- **Stability Score** - Cross-fold consistency
- **Generalization Gap** - Train-test difference

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost, scikit-learn Ensemble |
| Model Explainability | SHAP |
| Visualization | Plotly, Seaborn, Matplotlib |
| Web Frameworks | Flask, Streamlit |
| Data Query | DuckDB, SQL |
| Database Compatibility | CSV-based |
| File I/O | joblib, openpyxl |
| Deployment | Python Scripts, Batch Files |

---

## Project Structure

```
Leave Management System/
├── streamlit_app.py                    # Main Streamlit app
├── streamlit_sql_visualization.py      # SQL query dashboard
├── web_dashboard.py                    # Flask REST API dashboard
├── retrain_model.py                    # Model retraining pipeline
├── check_model.py                      # Model status checker
├── End_to_End_ML_Lifecycle_Training.ipynb  # Training notebook
├── requirements.txt                    # Python dependencies
├── start_dashboard.bat                 # Windows startup script
├── templates/
│   └── dashboard.html                  # Flask dashboard template
├── Data/
│   ├── Combined_All_Leave_Data.csv     # Leave records
│   └── Employee Master - Feb 2026 Team Member.xlsx  # Employee data
├── artifacts/                          # Model and forecast files
│   ├── leave_forecasting_model.pkl
│   ├── leave_forecasting_metadata.pkl
│   ├── leave_forecast_next_30days_*.csv
│   ├── leave_forecast_next_60days_*.csv
│   └── [metrics, predictions, importance files]
└── employee-leave-forecasting-system/  # Additional modules
    ├── artifacts/
    ├── config/
    └── tests/
```

---

## Version & Date Information
- **Current Date:** April 11, 2026
- **Employee Master Version:** Feb 2026
- **Training Focus:** Leave forecasting for workforce planning
- **Forecast Horizon:** 30-60 days forward

---

## Summary
This Leave Management System provides a complete ML-powered solution for:
- **Predicting** leave patterns 30-60 days in advance
- **Analyzing** leave trends by cost centre, department, and leave type
- **Planning** workforce capacity based on forecasts
- **Monitoring** special leaves and unplanned absences
- **Dashboarding** multiple views for different stakeholders

The system uses three machine learning models (RandomForest, GradientBoosting, XGBoost) and automatically selects the best performer based on WAPE metric, ensuring reliable and accurate leave predictions.
