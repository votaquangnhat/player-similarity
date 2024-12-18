import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import plotly.graph_objects as go


file_name = "Player_Final.csv"
file_name = st.selectbox(
    'Select dataset:',
    ['Player_Final.csv', 'GK_Final.csv'],
    placeholder="Select metric...", 
)
st.write('Data: ' + file_name)
df = pd.read_csv(f"data/{file_name}")

st.write('## 1. Player statistic')

player_name = st.selectbox(
    "Choose player for analyzing:",
    df['name'].unique(),
    index=None,
    placeholder="Select player...", 
)

st.write("Player you selected:", player_name)

player_data = df[df['name'] == player_name]
st.write(f"Statistics for {player_name}:")
st.dataframe(player_data)


attributes = st.multiselect("Select attributes:",
                                  df.select_dtypes(include=['number']).columns.to_list(),
                                  default=['Summary_Gls', 'Summary_Ast', 'Summary_SoT', 'Passing_1/3'],
                                  key=1)

if attributes and player_name:
    # player_stats.plot(kind='bar', ax= ax, color='skyblue')
    # ax.set_title(f"{player_name}'s Performance Metrics")
    # ax.set_ylabel('Value')
    #st.write(player_data[attributes].values.tolist()[0])
    fig = px.line_polar(player_data[attributes],r = player_data[attributes].values.tolist()[0], theta=attributes, line_close=True)
    fig.update_traces(fill='toself')
    st.plotly_chart(fig)

st.write('## 2. Player Comparison')

player1 = st.selectbox('Select Player 1:', df['name'].unique(),index=None, placeholder="Select player...",)
player2 = st.selectbox('Select Player 2:', df['name'].unique(),index=None, placeholder="Select player...",)

attributes_comp = st.multiselect("Select attributes:",
                                  df.select_dtypes(include=['number']).columns.to_list(),
                                  default=['Summary_Gls', 'Summary_Ast', 'Summary_SoT', 'Passing_1/3'],
                                  key=2)


if player1 and player2 and attributes_comp:
    player1_data = df[df['name'] == player1]
    player2_data = df[df['name'] == player2]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=player1_data[attributes_comp].values.tolist()[0],
        theta=attributes_comp,
        fill='toself',
        name=player1
    ))
    fig.add_trace(go.Scatterpolar(
        r=player2_data[attributes_comp].values.tolist()[0],
        theta=attributes_comp,
        fill='toself',
        name=player2
    ))

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        )),
    showlegend=False
    )

    st.plotly_chart(fig)