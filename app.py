import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor

st.markdown(
    "<h1 style='text-align: center; font-size: 38px;'>⚡ Smart Energy Consumption Analyzer</h1>",
    unsafe_allow_html=True
)
df = pd.read_csv("data/cleaned_data.csv")

df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace=True)
if st.checkbox("Show Raw Data"):
    st.write(df.head())

daily = df['-Global_active_power'].resample('D').mean()

fig, ax = plt.subplots()
daily.plot(ax=ax)

st.subheader("Daily Power Consumption")
st.pyplot(fig)

df_model = df.copy()
df_model['hour'] = df_model.index.hour
df_model['day'] = df_model.index.dayofweek
df_model['month'] = df_model.index.month

df_model['lag_1'] = df_model['-Global_active_power'].shift(1)
df_model['lag_2'] = df_model['-Global_active_power'].shift(2)
df_model['lag_24'] = df_model['-Global_active_power'].shift(24)

df_model = df_model.dropna()

X = df_model[['hour','day','month','lag_1','lag_2','lag_24']]
y = df_model['-Global_active_power']

rf = RandomForestRegressor(n_estimators=10, max_depth=10, random_state=42)
rf.fit(X, y)

st.subheader("Prediction Demo")
hour = st.slider("Hour", 0, 23)
day = st.slider("Day of Week", 0, 6)
month = st.slider("Month", 1, 12)

lag_1 = st.number_input("Previous Hour Power", value=1.0)
lag_2 = st.number_input("2 Hours Ago Power", value=1.0)
lag_24 = st.number_input("Yesterday Same Time", value=1.0)

input_data = np.array([[hour, day, month, lag_1, lag_2, lag_24]])
prediction = rf.predict(input_data)
st.write("Predicted Power Consumption:", prediction[0])