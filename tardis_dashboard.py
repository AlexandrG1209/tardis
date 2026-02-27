import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="TARDIS Dashboard", layout="wide")

#LOAD DATA

@st.cache_data
def load_data():
    return pd.read_csv("cleaned_dataset.csv")
df = load_data()

# LOAD MODEL

@st.cache_resource
def load_model():
    return joblib.load("realhistgrad_model.pkl")

model_data = load_model()
model = model_data["model"]
model_columns = model_data["columns"]

#TITLE

st.title("Train Delay Prediction")

# DATA OVERVIEW

st.header("Cleaned Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Rows", len(df))
col2.metric(
    "Average Delay",
    round(df["Retard moyen de tous les trains à l'arrivée"].mean(), 2)
)
col3.metric(
    "Median Delay",
    round(df["Retard moyen de tous les trains à l'arrivée"].median(), 2)
)

# DELAY DISTRIBUTION

st.subheader("Delay Distribution")

fig1, ax1 = plt.subplots()
sns.histplot(
    df["Retard moyen de tous les trains à l'arrivée"],
    bins=30,
    kde=True,
    ax=ax1
)
ax1.set_xlabel("Delay (minutes)")
st.pyplot(fig1)

# AVERAGE DELAY BY MONTH

if "month" in df.columns:
    st.subheader("Average Delay by Month")

    month_delay = (
        df.groupby("month")[
            "Retard moyen de tous les trains à l'arrivée"
        ]
        .mean()
        .reset_index()
    )

    fig2, ax2 = plt.subplots()
    sns.barplot(
        data=month_delay,
        x="month",
        y="Retard moyen de tous les trains à l'arrivée",
        ax=ax2
    )

    ax2.set_ylabel("Average Delay (minutes)")
    st.pyplot(fig2)

# UNIQUE STATIONS

unique_departures = sorted(
    df["Gare de départ"].dropna().unique()
)

unique_arrivals = sorted(
    df["Gare d'arrivée"].dropna().unique()
)

# PREDICTION SECTION

st.header("Gradient Model Prediction")

st.subheader("Enter Features")

month = st.number_input(
    "month",
    min_value=1,
    max_value=12,
    value=1
)

duree = st.number_input(
    "Durée moyenne du trajet",
    min_value=0.0,
    value=1000.0
)

circulations = st.number_input(
    "Nombre de circulations prévues",
    min_value=0.0,
    value=1000.0
)

service = st.selectbox(
    "Service",
    ["Unknown", "National", "International"]
)

gare_depart = st.selectbox(
    "Gare de départ",
    unique_departures
)

gare_arrivee = st.selectbox(
    "Gare d'arrivée",
    unique_arrivals
)

# Optional: Prevent same station
if gare_depart == gare_arrivee:
    st.warning("Departure and arrival stations should be different.")

# PREDICT BUTTON

if st.button("Predict Delay"):

    input_df = pd.DataFrame([{
        "month": month,
        "Durée moyenne du trajet": duree,
        "Nombre de circulations prévues": circulations,
        "Service": service,
        "Gare de départ": gare_depart,
        "Gare d'arrivée": gare_arrivee
    }])
    # Apply same get_dummies
    input_df = pd.get_dummies(input_df)

    # Add missing columns
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Remove extra columns
    input_df = input_df[model_columns]

    # Predict
    prediction = model.predict(input_df)[0]

    st.success(f"Predicted Delay: {round(prediction, 2)} minutes")