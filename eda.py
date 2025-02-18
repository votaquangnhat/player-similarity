import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.write('# Title: Football analysis')

file_name = "Player_Final.csv"
file_name = st.selectbox(
    'Select dataset:',
    ['Player_Final.csv', 'GK_Final.csv'],
    placeholder="Select metric...", 
)
st.write('Data: ' + file_name)
df = pd.read_csv(f"data/{file_name}")

st.write('## 1. Statistics')
st.write(df.describe())

st.write('Position Distribution:')
pos_counts = df['Pos'].value_counts()
fig, ax = plt.subplots()
pos_counts.plot(kind='bar', ax=ax)
st.pyplot(fig)

st.write('Top 10 nations with the most players:')
pos_counts = df['Nation'].value_counts().head(10)
fig, ax = plt.subplots()
pos_counts.plot(kind='bar', ax=ax)
st.pyplot(fig)

st.write('Top 10 teams with the most players:')
pos_counts = df['current_club_name'].value_counts().head(10)
fig, ax = plt.subplots()
pos_counts.plot(kind='bar', ax=ax)
if file_name == 'Player_Final.csv':
    ax.set_ylim(bottom=25)
st.pyplot(fig)

if file_name == 'Player_Final.csv':
    st.write(r'Penalty_Success:')
    t = df[df['Summary_PKatt'] > 0][['name', 'Summary_PKatt', 'Penalty_Success_%']].sort_values(by='Summary_PKatt', ascending=False)
    t
else:
    st.write(r'Goalkeeping_Saves')
    t = df[df['Goalkeeping_Saves'] > 0][['name', 'Goalkeeping_Saves', 'Goalkeeping_Save%']].sort_values(by='Goalkeeping_Saves', ascending=False)
    t

column = st.selectbox('Select a column for distribution:',
                      df.select_dtypes(include='number').columns,
                      index=2) # default is Min
fig, ax = plt.subplots()
df[column].hist(bins=20, ax=ax)
ax.set_title('Distribution for ' + column)
st.pyplot(fig)

st.write(len(df.select_dtypes(include='number').columns))

st.write('## 2. Correlation')
#corr = df.select_dtypes(include='number').corr()

selected_columns = st.multiselect("Select columns to compute correlation:",
                                  df.select_dtypes(include=['number']).columns.to_list(),
                                  default=df.select_dtypes(include=['number']).columns.to_list()[2:12])

if selected_columns:
    corr = df[selected_columns].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, mask=mask)
    ax.set_title("Correlation Heatmap of selected columns", fontsize=16)
    st.pyplot(fig)


selected_columns = df.select_dtypes(include=['number']).columns.to_list()
corr = df[selected_columns].corr()
threshold = st.number_input(
"Insert a number", value=0.9, placeholder="Enter the threshold...")
st.write("The current threshold is ", threshold)
attribute = st.selectbox('Select an attribute for check correlation:',
                      df.select_dtypes(include='number').columns,
                      index=2)
high_corr_pairs = corr.where(np.abs(corr) > threshold).stack().reset_index()
high_corr_pairs.columns = ['Attribute 1', 'Attribute 2', 'Correlation']
high_corr_pairs = high_corr_pairs[high_corr_pairs['Attribute 1'] != high_corr_pairs['Attribute 2']]
st.write("High Correlations (|correlation| > threshold):")
high_corr_pairs[high_corr_pairs['Attribute 1']==attribute]
high_corr_pairs

st.write('Drop columns that have high correlation with another columns')

dropped_columns_set = set()

for _, row in high_corr_pairs.iterrows():
    attr1, attr2, corr = row["Attribute 1"], row["Attribute 2"], row["Correlation"]
    if attr1 not in dropped_columns_set and attr2 not in dropped_columns_set:
        dropped_columns_set.add(attr2)

dropped_columns = list(dropped_columns_set)
dropped_columns
# #####
# selected_columns = df.select_dtypes(include=['number']).columns.to_list()
# t = df[selected_columns].drop(columns=dropped_columns)
# corr = t.corr()
# high_corr_pairs = corr.where(np.abs(corr) > threshold).stack().reset_index()
# high_corr_pairs.columns = ['Attribute 1', 'Attribute 2', 'Correlation']
# high_corr_pairs = high_corr_pairs[high_corr_pairs['Attribute 1'] != high_corr_pairs['Attribute 2']]
# high_corr_pairs


# selected_columns_pp = st.multiselect("Select columns to pairplot:",
#                                   df.select_dtypes(include=['number']).columns.to_list(),
#                                   default=['Summary_Gls', 'Summary_xG'])

# if selected_columns_pp:
#     sns.pairplot(df[selected_columns_pp])
#     fig = plt.gcf()
#     st.pyplot(fig)