import streamlit as st

pages = {
    "Navigation": [
        st.Page('eda.py', title="EDA"),
        st.Page('player_clustering.py', title="Player Clustering"),
        st.Page('player_analysis.py', title="Player analysis"),
        st.Page('similar_players.py', title="Similar player finding"),
    ],
}

pg = st.navigation(pages)
pg.run()