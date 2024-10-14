import streamlit as st
import pandas as pd
import pydeck as pdk
import xgboost as xgb
import numpy as np
from datetime import datetime as dt
from scipy.interpolate import CubicSpline
from statsmodels.tsa.arima.model import ARIMA

predictions = pd.read_csv('predictions.csv')
crime_monthly = pd.read_csv('crime_monthly.csv')
rent_crime_monthly = pd.read_csv('rent_crime_monthly.csv')


crime_monthly['Date'] = pd.to_datetime(crime_monthly['Date'])

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
total_sentences_predictions_monthly['Location'] = total_sentences_predictions_monthly['Location'].apply(lambda x: x if x == 'ALL' else x.replace(" Region", ""))
total_sentences_predictions_monthly = pd.concat([crime_monthly, total_sentences_predictions_monthly])


loaded_model = xgb.Booster()
loaded_model.load_model('xgboost-model-0')

location_data = {
    'Location Id': [0, 1, 2, 3, 4, 5 , 7, 8, 9, 13, 14, 15],
    'Location': ['ALL','Northland', 'Auckland', 'Waikato', 'Bay of Plenty', 'Gisborne', 
                  'Taranaki', 'Manawatu-Wanganui', 'Wellington', 
                  'Canterbury', 'Otago', 'Southland']
}

location_df = pd.DataFrame(location_data)
rent_crime_monthly = rent_crime_monthly[rent_crime_monthly['Location'].isin(location_df['Location'])]
ren_monthly = rent_crime_monthly[['Time Frame','Median Rent']]
ren_monthly['Time Frame'] = pd.to_datetime(ren_monthly['Time Frame'])

st.title("Crime and Rent")


years_range = list(range(2025, 2034))  
months_range = list(range(1, 13))
selected_year = st.selectbox("Year", years_range)
selected_month = st.selectbox("Month", months_range)
selected_location = st.selectbox("Location", location_df['Location'])
location_id = location_df[location_df['Location'] == selected_location]['Location Id'].values[0]

selected_date = pd.to_datetime(f'{selected_year}-{selected_month}-01')

if location == "ALL":
    crime = st.number_input("Crime", value=0.0, step=0.1, disabled=True)
else:
    crime = st.number_input("Crime", value=0.0, step=0.1)
    selected_df = total_sentences_predictions_monthly[(total_sentences_predictions_monthly['Date'] <= selected_date) & (total_sentences_predictions_monthly['Location'] == selected_location)]
    selected_df.loc[selected_df.index[-1], 'Crime'] = crime
    selected_df['Crime_Rolling_Std_3'] = selected_df['Crime'].rolling(3,1).std()
    selected_df['Crime_Rolling_Std_6'] = selected_df['Crime'].rolling(6,1).std()

if st.button("Predict"):
    if location == "ALL":
        last_date = df['Time Frame'].max()
        delta_months = (selected_date.year - last_date.year) * 12 + (selected_date.month - last_date.month)
        predictions_per_location = {}
        for location in location_df['Location']:
            location_data = ren_monthly[ren_monthly['Location'] == location].set_index('Time Frame')
            location_data = location_data.sort_index()
            y = location_data['Median Rent']
            model = ARIMA(y, order=(1, 1, 1))
            model_fit = model.fit()
            forecast = model_fit.get_forecast(steps=delta_months)
            predicted_rent = forecast.predicted_mean
            predictions_per_location[location] = predicted_rent[-1]
        st.write(predictions_per_location)
    else:
        features = selected_df.iloc[-1][['Crime_Rolling_Std_3', 'Crime_Rolling_Std_6']]
        features['Location Id'] = location_id
        dtest = xgb.DMatrix([features])
        predictions = loaded_model.predict(dtest)




