#!/usr/bin/env python
# coding: utf-8

# ===============================
# 🔹 IMPORTS
# ===============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import plotly.express as px

warnings.filterwarnings('ignore')
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# ===============================
# 🔹 LOAD DATA
# ===============================
df = pd.read_csv('mymoviedb.csv', lineterminator='\n')
print(" Data loaded successfully!")
print("Shape:", df.shape)
print(df.head())

# ===============================
# 🔹 BASIC CLEANING
# ===============================
df.info()
df['Release_Date'] = pd.to_datetime(df['Release_Date'], errors='coerce')
df['Release_Date'] = df['Release_Date'].dt.year

# Drop unnecessary columns safely
cols = ['Overview', 'Original_Language', 'Poster_Url']
df.drop(cols, axis=1, inplace=True, errors='ignore')

for col in cols:
    if col not in df.columns:
        print(f" Column not found: {col}")

# ===============================
# 🔹 CONVERT TYPES
# ===============================
df['Vote_Average'] = pd.to_numeric(df['Vote_Average'], errors='coerce')
df['Vote_Count'] = pd.to_numeric(df['Vote_Count'], errors='coerce')
df['Popularity'] = pd.to_numeric(df['Popularity'], errors='coerce')

# ===============================
# 🔹 CATEGORIZE 'Vote_Average'
# ===============================
def categorize_col(df, col, labels):
    df[col] = pd.to_numeric(df[col], errors='coerce')
    valid = df[col].dropna()
    edges = sorted(valid.quantile([0, 0.25, 0.5, 0.75, 1.0]).unique())
    
    if len(set(edges)) < 2:
        print(f"Not enough variation in {col} to create bins. Skipping categorization.")
        return df
    
    df[f"{col}_Category"] = pd.cut(
        df[col],
        bins=edges,
        labels=labels[:len(edges)-1],
        include_lowest=True,
        duplicates='drop'
    )
    return df

labels = ['not_popular', 'below_avg', 'average', 'popular']
df = categorize_col(df, 'Vote_Average', labels)

# ===============================
# 🔹 HANDLE NULLS & DUPLICATES
# ===============================
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

print("Cleaned data shape:", df.shape)

# ===============================
# 🔹 SPLIT GENRES INTO ROWS
# ===============================
if 'Genre' in df.columns:
    df['Genre'] = df['Genre'].astype(str).str.split(', ')
    df = df.explode('Genre').reset_index(drop=True)
    df['Genre'] = df['Genre'].astype('category')
else:
    print(" 'Genre' column not found!")

# ===============================
# 🔹 SAMPLE VIEW (SAFE)
# ===============================
if not df.empty:
    print(df.sample(5))
else:
    print(" DataFrame is empty — no data to display!")

# ===============================
# 🔹 VISUALIZATIONS
# ===============================
sns.set_style('whitegrid')

# Genre distribution
sns.catplot(y='Genre', data=df, kind='count',
            order=df['Genre'].value_counts().index,
            color='#4287f5')
plt.title('Genre column distribution')
plt.show()

# Votes distribution
sns.catplot(y='Vote_Average', data=df, kind='count',
            order=df['Vote_Average'].value_counts().index,
            color='#4287f5')
plt.title('Votes distribution')
plt.show()

# Most popular
print(" Highest Popularity Movie:")
print(df[df['Popularity'] == df['Popularity'].max()])

# Least popular
print(" Lowest Popularity Movie:")
print(df[df['Popularity'] == df['Popularity'].min()])

# Release year distribution
df['Release_Date'].hist()
plt.title('Release Year Distribution')
plt.show()

# Top 10 genres
genre_counts = df['Genre'].value_counts().head(10)
sns.barplot(x=genre_counts.values, y=genre_counts.index, palette='viridis')
plt.title('Top 10 Genres on Netflix')
plt.xlabel('Number of Titles')
plt.show()

# Scatter plots
sns.scatterplot(x='Popularity', y='Vote_Average', hue='Genre', data=df, alpha=0.7)
plt.title('Popularity vs Vote Average by Genre')
plt.show()

sns.scatterplot(data=df, x='Vote_Count', y='Popularity', hue='Genre', alpha=0.7)
plt.title('Popularity vs Vote Count by Genre')
plt.xlabel('Vote Count')
plt.ylabel('Popularity')
plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

sns.scatterplot(data=df, x='Vote_Count', y='Vote_Average', hue='Genre', alpha=0.7)
plt.title('Vote Count vs Vote Average by Genre')
plt.xlabel('Vote Count')
plt.ylabel('Vote Average')
plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# ===============================
# 🔹 INTERACTIVE PLOTS (PLOTLY)
# ===============================
fig = px.scatter(
    df,
    x='Popularity',
    y='Vote_Average',
    color='Genre',
    hover_data=['Title'],
    size='Vote_Count',
    title='Popularity vs Vote Average (Interactive Scatter)'
)
fig.show()

fig = px.scatter(
    df,
    x='Vote_Average',
    y='Popularity',
    color='Release_Date',
    hover_data=['Title', 'Genre'],
    size='Vote_Count',
    color_continuous_scale='Viridis',
    title='Popularity vs Vote Average Colored by Year'
)
fig.show()

# ===============================
# 🔹 PIE CHART
# ===============================
popularity_by_genre = df.groupby('Genre')['Popularity'].mean().sort_values(ascending=False)
plt.figure(figsize=(8, 8))
plt.pie(
    popularity_by_genre.values,
    labels=popularity_by_genre.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=plt.cm.Reds(np.linspace(0.3, 0.85, len(popularity_by_genre)))
)
plt.title('Average Popularity Share by Genre', fontsize=14, color='darkred')
plt.show()

# ===============================
# 🔹 SAVE CLEANED DATA
# ===============================
df.to_csv("mymoviedb_clean.csv", index=False)
print(" Cleaned file saved as mymoviedb_clean.csv")
