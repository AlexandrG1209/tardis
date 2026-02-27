# Train Delay Prediction System
## Project Overview

This project analyzes train journey data to understand delay patterns and predict future delays.

- Clean and preprocess raw transportation data
- Perform exploratory data analysis (EDA) to uncover delay patterns
- Build a machine learning regression model to predict train delay duration (in minutes)
- Deploy an interactive Streamlit dashboard for visualization and real-time delay prediction
- The final system allows users to explore delay trends and predict expected delays based on journey characteristics.

---

## Objectives:

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Analyze delay details
- Train regression model to predict delay duration
- Interactive delay visualization

---

## Step 1: tardis_eda.ipynb
- The file `project_dataset.csv` is inspected
- Duplicate rows are removed
- Data types are corrected
- A new feature "month" is created from the Date column
- Missing values are handled
- Output: - `cleaned_dataset.csv`

## Step 2: tardis_eda.ipynb (again)
- Distribution of delay durations
- Average delays by station
- Identification of peak delay periods
- Boxplot of delays by day
- Bar chart of average delays by station

---

## Step 3: tardis_model.ipynb
- `tardis_model.ipynb`: Uses RandomForestRegressor for the regression
- `tardis_model2.ipynb`: Uses LinearRegression
- `tardis_model3.ipynb`: Uses HistGradientBoostingRegressor

The target variable is:

- **Average train delay in minutes**

Models are evaluated using:

- R² score  
- Mean Absolute Error (MAE)  
- Cross-validation  

---

## Step 4: Streamlit Dashboard - `tardis_dashboard.py`

The dashboard allows users to:

- View delay distribution
- View average delay by month
- Select a route and see its average delay
- Predict delay duration using:
  - Month
  - Average journey duration
  - Number of planned circulations
  - Service type
  - Departure station
  - Arrival station

---

## How to run our dashboard:

python3 -m venv venv
source venv/bin/activate
pip install streamlit pandas joblib matplotlib seaborn scikit-learn
streamlit run tardis_dashboard.py