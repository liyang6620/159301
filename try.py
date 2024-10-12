import streamlit as st
import pandas as pd
import pydeck as pdk

st.title("简单的计算器")

# 用户输入
num1 = st.number_input("输入第一个数字", value=0.0, step=0.1)
num2 = st.number_input("输入第二个数字", value=0.0, step=0.1)

# 算术运算选择
operation = st.selectbox("选择操作", ("加法", "减法", "乘法", "除法"))

# 计算结果
if operation == "加法":
    result = num1 + num2
elif operation == "减法":
    result = num1 - num2
elif operation == "乘法":
    result = num1 * num2
elif operation == "除法":
    if num2 != 0:
        result = num1 / num2
    else:
        result = "除数不能为0！"

# 显示结果
st.write("结果是: ", result)


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
