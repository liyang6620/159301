import streamlit as st
import pandas as pd
import pydeck as pdk
import xgboost as xgb
import numpy as np

loaded_model = xgb.Booster()
loaded_model.load_model('xgboost-model-0')

st.title("简单的计算器")


num1 = st.number_input("输入第一个数字", value=0.0, step=0.1)
num2 = st.number_input("输入第二个数字", value=0.0, step=0.1)
num3 = st.number_input("输入第三个数字", value=0.0, step=0.1)
input_data = np.array([[num1, num2, num3]])

# 使用模型进行预测
if st.button("进行预测"):
    dmatrix = xgb.DMatrix(input_data)
    prediction = loaded_model.predict(dmatrix)
    st.write(f"预测结果是: {prediction[0]}")


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


st.dataframe(df)
