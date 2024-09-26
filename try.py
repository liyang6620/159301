import streamlit as st
import pandas as pd
import pydeck as pdk

# 示例数据，假设这是新西兰各城市的租金数据
data = {
    'City': ['Auckland', 'Wellington', 'Christchurch', 'Hamilton', 'Tauranga'],
    'Latitude': [-36.8485, -41.2865, -43.5321, -37.7870, -37.6878],
    'Longitude': [174.7633, 174.7762, 172.6362, 175.2830, 176.1651],
    'Average Rent': [2300, 2100, 1900, 1800, 2000]
}

df = pd.DataFrame(data)

# 设置地图视图
view_state = pdk.ViewState(latitude=-40.9006, longitude=174.8860, zoom=5)

# 创建地图图层
layer = pdk.Layer(
    'ScatterplotLayer',
    data=df,
    get_position='[Longitude, Latitude]',
    get_color='[200, 30, 0, 160]',
    get_radius=10000,  # 半径大小，根据需要调整
    pickable=True
)

# Tooltip 用于展示更多信息
tooltip={
    "html": "<b>City:</b> {City}<br><b>Average Rent:</b> ${Average Rent}",
    "style": {
        "backgroundColor": "steelblue",
        "color": "white"
    }
}

# 使用 pydeck_chart 渲染地图
st.pydeck_chart(pdk.Deck(
    initial_view_state=view_state,
    layers=[layer],
    tooltip=tooltip
))

# 可选：显示数据表格
st.dataframe(df)
