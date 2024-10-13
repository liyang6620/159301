import streamlit as st
import pandas as pd
import pydeck as pdk
import xgboost as xgb
import numpy as np
from datetime import datetime as dt
from scipy.interpolate import CubicSpline

predictions = pd.read_csv('predictions.csv')
total_sentences_predictions = predictions[predictions['Target'] == 'Total Sentences'].sort_values(['Year','Region'])
total_sentences_predictions = total_sentences_predictions.loc[total_sentences_predictions.groupby(['Year', 'Region'])['MSE'].idxmin()]

total_sentences_predictions['Date'] = pd.to_datetime(total_sentences_predictions['Year'], format='%Y') + pd.offsets.YearBegin(0)
total_sentences_predictions.set_index('Date', inplace=True)
total_sentences_predictions_monthly = pd.DataFrame()

for location in total_sentences_predictions['Region'].unique():
    total_sentences_predictions_location = total_sentences_predictions[total_sentences_predictions['Region'] == location]

    years = total_sentences_predictions_location.index.year
    total_sentences_predictionse_values = total_sentences_predictions_location['Predicted']

    months = pd.date_range(start=str(years.min()), end=str(years.max() + 1), freq='MS')

    spline = CubicSpline(years, total_sentences_predictionse_values)
    interpolated_total_sentences = spline(months.year + months.month / 12)

    interpolated_data = pd.DataFrame({
        'Date': months,
        'Location': location,
        'Crime': interpolated_total_sentences
    })

    total_sentences_predictions_monthly = pd.concat([total_sentences_predictions_monthly, interpolated_data])

total_sentences_predictions_monthly.reset_index(drop=True, inplace=True)

loaded_model = xgb.Booster()
loaded_model.load_model('xgboost-model-0')

st.title("Crime and Rent")


selected_date = st.date_input("Select date", dt.today())
num2 = st.number_input("输入第二个数字", value=0.0, step=0.1)
num3 = st.number_input("输入第三个数字", value=0.0, step=0.1)

st.write(selected_date)


data = {
    'City': ['Auckland', 'Wellington', 'Christchurch', 'Hamilton', 'Tauranga'],
    'Latitude': [-36.8485, -41.2865, -43.5321, -37.7870, -37.6878],
    'Longitude': [174.7633, 174.7762, 172.6362, 175.2830, 176.1651],
    'Average Rent': [2300, 2100, 1900, 1800, 2000]
}

df = pd.DataFrame(data)

view_state = pdk.ViewState(latitude=-40.9006, longitude=174.8860, zoom=5)


layer = pdk.Layer(
    'ScatterplotLayer',
    data=df,
    get_position='[Longitude, Latitude]',
    get_color='[200, 30, 0, 160]',
    get_radius=10000,  # 半径大小，根据需要调整
    pickable=True
)

tooltip={
    "html": "<b>City:</b> {City}<br><b>Average Rent:</b> ${Average Rent}",
    "style": {
        "backgroundColor": "steelblue",
        "color": "white"
    }
}

st.pydeck_chart(pdk.Deck(
    initial_view_state=view_state,
    layers=[layer],
    tooltip=tooltip
))


st.dataframe(total_sentences_predictions_monthly)
