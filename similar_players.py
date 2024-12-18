import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean, cityblock
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity

st.write('Selected features for similarity finding: ')

file_name = "Player_Final.csv"
file_name = st.selectbox(
    'Select dataset:',
    ['Player_Final.csv', 'GK_Final'],
    placeholder="Select metric...", 
)
st.write('Data: ' + file_name)
df = pd.read_csv(f"data/{file_name}")

features = df.columns.tolist()
exclude_feature = ['player_id', 'name', '#', 'Team Name', 'Nation', 'Summary_CrdY', 'Summary_CrdR', 'Pos',
                   'Miscellaneous_Stats_2CrdY', 'Miscellaneous_Stats_Fls', 'Miscellaneous_Stats_Off',
                   'total_games', 'Penalty_Success_%', 'foot', 'image_url', 'height_in_cm',
                   'current_club_domestic_competition_id', 'current_club_name', 'market_value_in_eur', 'Age']
selected_features = [x for x in features if x not in exclude_feature]

selected_features

player_name = st.selectbox(
    "Choose player for finding:",
    df['name'].unique(),
    index=None,
    placeholder="Select player...", 
)

similarity_metric = st.selectbox(
    'Select Similarity Metric:',
    ['Euclidean', 'Manhattan', 'Cosine'],
    index=None,
    placeholder="Select metric...", 
)

st.write("Player you selected:", player_name)
st.write("Metric you selected:", similarity_metric)

if player_name and similarity_metric:
    player_data = df[df['name'] == player_name]

    #normalize
    scaler = MinMaxScaler()
    normalized_df = df.copy()
    normalized_df[selected_features] = scaler.fit_transform(df[selected_features])

    #compute similariry

    if similarity_metric == 'Euclidean':
        target_vector = normalized_df[normalized_df['name'] == player_name][selected_features].iloc[0]
        normalized_df['Similarity'] = normalized_df[selected_features].apply(
            lambda x: euclidean(target_vector, x), axis=1
        )
    elif similarity_metric == 'Manhattan':
        target_vector = normalized_df[normalized_df['name'] == player_name][selected_features].iloc[0]
        normalized_df['Similarity'] = normalized_df[selected_features].apply(
            lambda x: cityblock(target_vector, x), axis=1
        )
    elif similarity_metric == 'Cosine':
        similarity_matrix = cosine_similarity(normalized_df[selected_features])
        target_index = normalized_df[normalized_df['name'] == player_name].index[0]
        normalized_df['Similarity'] = similarity_matrix[target_index]

    similar_players = normalized_df.sort_values(by='Similarity').head(6)
    similar_players = similar_players[similar_players['name'] != player_name]
    st.write(f"Top Similar Players to {player_name} using using {similarity_metric}:")
    similar_players[['name', 'Similarity']]

    st.write('Comparison radar chart:')
    fig = go.Figure()

    for player_name in similar_players['name']:
        player_data = similar_players[similar_players['name'] == player_name]
        fig.add_trace(go.Scatterpolar(
            r=player_data[selected_features].values.tolist()[0],
            theta=selected_features,
            fill='toself',
            name=player_name
        ))

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        )),
    showlegend=False
    )

    st.plotly_chart(fig)
