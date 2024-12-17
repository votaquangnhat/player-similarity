import streamlit as st
import pandas as pd

df = pd.read_csv(r"data/Player_Gr.csv")

option = st.selectbox(
    "Choose player for analyzing:",
    df['Player'].unique(),
    index=None,
    placeholder="Select player...",
)

st.write("Player you selected:", option)
