import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load data & model
df = pd.read_csv("cleaned_dataset.csv")
model = joblib.load("model.joblib")

st.title("Train Delay Analysis Dashboard")

# Summary Statistics Panel
st.subheader("Summary Statistics")

avg_delay = df["Retard moyen de tous les trains à l'arrivée"].mean()
total_trips = df["Nombre de circulations prévues"].sum()
punctuality_rate = 100 - df["Prct retard pour causes externes"].mean()

st.metric("Average Arrival Delay (min)", round(avg_delay, 2))
st.metric("Total Planned Trips", int(total_trips))
st.metric("Approx. Punctuality Rate (%)", round(punctuality_rate, 2))

# Interactive Filter
station_filter = st.selectbox(
    "Select Departure Station",
    df["Gare de départ"].unique()
)

filtered_df = df[df["Gare de départ"] == station_filter]

# Delay Distribution Chart
st.subheader("Delay Distribution")

fig, ax = plt.subplots()
sns.histplot(filtered_df["Retard moyen de tous les trains à l'arrivée"], bins=20, ax=ax)
st.pyplot(fig)

# Prediction Interface
st.subheader("Predict Delay")

service = st.selectbox("Service", df["Service"].unique())
arrival = st.selectbox("Arrival Station", df["Gare d'arrivée"].unique())
duration = st.number_input("Durée moyenne du trajet", min_value=0.0)
circulations = st.number_input("Nombre de circulations prévues", min_value=0)
cancelled = st.number_input("Nombre de trains annulés", min_value=0)
month = st.slider("Month", 1, 12, 1)
dayofweek = st.slider("Day of Week (0=Mon)", 0, 6, 0)

if st.button("Predict"):
    input_data = pd.DataFrame([{
        "Service": service,
        "Gare de départ": station_filter,
        "Gare d'arrivée": arrival,
        "Durée moyenne du trajet": duration,
        "Nombre de circulations prévues": circulations,
        "Nombre de trains annulés": cancelled,
        "Month": month,
        "DayOfWeek": dayofweek
    }])
    prediction = model.predict(input_data)
    st.success(f"Predicted Delay: {round(prediction[0],2)} minutes")