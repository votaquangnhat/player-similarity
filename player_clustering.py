import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA
import altair as alt
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

st.write('## 1. Some clustering algorithms')

st.write('Selected features for clustering: ')

df = pd.read_csv(r"data/Player_Final.csv")

df = df[df['Min'] >=5]

features = df.columns.tolist()
exclude_feature = ['player_id', 'name', '#', 'Team Name', 'Nation', 'Summary_CrdY', 'Summary_CrdR', 'Pos',
                   'Miscellaneous_Stats_2CrdY', 'Miscellaneous_Stats_Fls', 'Miscellaneous_Stats_Off',
                   'total_games', 'Penalty_Success_%', 'foot', 'image_url', 'height_in_cm',
                   'current_club_domestic_competition_id', 'current_club_name', 'market_value_in_eur', 'Age']
selected_features = [x for x in features if x not in exclude_feature]

selected_features


# Clustering
scaler = StandardScaler()
normalized_data = scaler.fit_transform(df[selected_features])

clustering_algorithm = st.selectbox('Select Clustering Algorithm:', ['K-Means', 'Hierarchical'])
num_clusters = None

if clustering_algorithm == 'K-Means':
    num_clusters = st.slider('Number of Clusters:', 2, 20, 1)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(normalized_data)
    df['Cluster'] = cluster_labels
elif clustering_algorithm == 'Hierarchical':
    method = st.selectbox('Linkage Method:', ['ward', 'complete', 'average', 'single'])
    linkage_matrix = linkage(normalized_data, method=method)
    num_clusters = st.slider('Number of Clusters:', 2, 20, 1)
    cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
    df['Cluster'] = cluster_labels


#PCA
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(normalized_data)
df['PCA1'] = reduced_data[:, 0]
df['PCA2'] = reduced_data[:, 1]

st.write(f"Player Clusters ({clustering_algorithm}):")
st.altair_chart(alt.Chart(df).mark_circle(size=60).encode(
    x='PCA1',
    y='PCA2',
    color='Cluster:N',
    tooltip=['name', 'Cluster']
).interactive())

#PCA 3d
pca = PCA(n_components=3)
reduced_data = pca.fit_transform(normalized_data)
df['PCA1'] = reduced_data[:, 0]
df['PCA2'] = reduced_data[:, 1]
df['PCA3'] = reduced_data[:, 2]
df['Cluster'] = df['Cluster'].astype(str)

fig = px.scatter_3d(
    df,
    x='PCA1',
    y='PCA2',
    z='PCA3',
    color='Cluster',
    hover_name='name',
    hover_data=selected_features,
    title='3D Visualization of Player Clusters'
)

st.plotly_chart(fig)

cluster_summary = df.groupby('Cluster')[selected_features].mean().reset_index()
st.write("Cluster Summary:")
cluster_summary

selected_cluster = st.selectbox('Select Cluster to Inspect:', sorted(df['Cluster'].unique()))
cluster_players = df[df['Cluster'] == selected_cluster]
st.write(f"Players in Cluster {selected_cluster}:")
st.dataframe(cluster_players[['name', 'Nation', 'Pos'] + selected_features])

#KMean++
st.write('## 2. K-Mean++')
wcss = []
for i in range(1, 16): 
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(reduced_data) 
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 16), wcss)
plt.xlabel('Number of clusters (K)')
plt.ylabel('WCSS')
st.pyplot(plt)

st.write('Choose n_clusters = 6')
