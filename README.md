Train Delay Prediction System:
Project Overview

This project analyzes train journey data to:

-Clean and preprocess raw transportation data
-Perform exploratory data analysis (EDA) to uncover delay patterns
-Build a machine learning regression model to predict train delay duration (in minutes)
-Deploy an interactive Streamlit dashboard for visualization and real-time delay prediction
The final system allows users to explore delay trends and predict expected delays based on journey characteristics.

Objectives:

-Data Cleaning & Preprocessing
-Exploratory Data Analysis (EDA)
-Analyze delay details
-Train regression model to predict delay duration
-Interactive delay visualization

Step 1: tardis_eda.ipynb
-project_dataset.csv inspection
-Duplicate removal
-Output: cleaned_dataset.csv

Step 2: tardis_eda.ipynb (again)
-Distribution of delay durations
-Average delays by station
-Identification of peak delay periods
-Boxplot of delays by day
-Bar chart of average delays by station

Step 3: tardis_model.ipynb
tardis_model.ipynb: Uses RandomForestRegressor for the regression
tardis_model2.ipynb: Uses LinearRegression
tardis_model3.ipynb: Uses HistGradientBoostingRegressor

Step 4: tardis_dashboard.py
How to run our dashboard:
python3 -m venv venv
source venv/bin/activate
pip install streamlit pandas joblib matplotlib seaborn scikit-learn
streamlit run tardis_dashboard.py