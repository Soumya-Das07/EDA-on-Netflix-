import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# ======================
# PAGE CONFIGURATION
# ======================
st.set_page_config(page_title="Netflix & Movies EDA Dashboard", layout="wide")
st.title(" Netflix Movies & TV Shows EDA Dashboard")

# ======================
# LOAD DATA
# ======================
@st.cache_data
def load_data():
    df = pd.read_csv("mymoviedb.csv", encoding='utf-8-sig', lineterminator='\n')
    df['Release_Date'] = pd.to_datetime(df['Release_Date'], errors='coerce')
    df['Release_Date'] = df['Release_Date'].dt.year
    df['Vote_Count'] = pd.to_numeric(df['Vote_Count'], errors='coerce')
    df['Popularity'] = pd.to_numeric(df['Popularity'], errors='coerce')

    # Split genres into multiple rows
    if 'Genre' in df.columns:
        df['Genre'] = df['Genre'].astype(str).str.split(', ')
        df = df.explode('Genre')

    return df

df = load_data()

# ======================
# SIDEBAR DASHBOARD
# =====================
st.sidebar.title(" Dashboard Menu")
st.sidebar.markdown("Click on any section to visualize data:")

options = [
    "1 - Dataset Overview",
    "2 - Genre Distribution",
    "3 - Release Date Column Distribution",
    "4 - Top 10 Genres",
    "5 - Popularity vs Vote Count",
    "6 - Popularity Share by Genre (Pie Chart)",
    "7 - Highest & Lowest Popularity Movies",
    "8 - Correlation Heatmap"
]


choice = st.sidebar.radio("Select Graph:", options)

# ======================
# VISUALIZATIONS
# ======================
sns.set_style("whitegrid")

# Dataset Overview
if choice == "1 - Dataset Overview":
    st.subheader("Dataset Overview")
    st.write("Shape:", df.shape)
    st.dataframe(df.head())
    st.write("Columns:", list(df.columns))
    st.write(df.describe())

#  Genre Distribution
elif choice == "2 - Genre Distribution":
    st.subheader("Genre Distribution")
    genre_counts = df['Genre'].value_counts()
    fig = px.bar(
        x=genre_counts.values,
        y=genre_counts.index,
        orientation="h",
        title="Genre Count",
        labels={'x': 'Count', 'y': 'Genre'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
elif choice == "3 - Release Date Column Distribution":
    st.subheader("RELEASE DATE COLUMN DISTRIBUTION")

    # Convert 'Release_Date' to datetime safely
    df['Release_Date'] = pd.to_datetime(df['Release_Date'], errors='coerce')

    # Extract release year
    df['Year'] = df['Release_Date'].dt.year

    # Drop missing years
    df = df.dropna(subset=['Year'])
    df['Year'] = df['Year'].astype(int)

    # Plot histogram (exactly like your backend)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        df['Year'],
        bins=30,
        color='steelblue',
        edgecolor='black'
    )
    ax.set_title('RELEASE DATE COLUMN DISTRIBUTION', fontsize=14, fontweight='bold')
    ax.set_xlabel('Release Year')
    ax.set_ylabel('Number of Titles')

    # Display on Streamlit
    st.pyplot(fig)


# Top 10 Genres
elif choice == "4 - Top 10 Genres":
    st.subheader("Top 10 Genres on Netflix")
    top_genres = df['Genre'].value_counts().head(10)
    fig, ax = plt.subplots()
    sns.barplot(x=top_genres.values, y=top_genres.index, palette='viridis', ax=ax)
    st.pyplot(fig)

#  Popularity vs Vote Count
elif choice == "5 - Popularity vs Vote Count":
    st.subheader("Popularity vs Vote Count by Genre")
    fig, ax = plt.subplots()
    sns.scatterplot(
        x='Vote_Count', y='Popularity',
        hue='Genre', data=df, alpha=0.7, ax=ax
    )
    st.pyplot(fig)

# Popularity Share by Genre
elif choice == "6 - Popularity Share by Genre (Pie Chart)":
    st.subheader("Average Popularity Share by Genre")
    popularity_by_genre = df.groupby('Genre')['Popularity'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 7))
    plt.pie(
        popularity_by_genre.values,
        labels=popularity_by_genre.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=plt.cm.Reds(np.linspace(0.3, 0.85, len(popularity_by_genre)))
    )
    plt.title('Popularity Share by Genre', fontsize=13)
    st.pyplot(fig)

#  Highest & Lowest Popularity Movies
elif choice == "7 - Highest & Lowest Popularity Movies":
    st.subheader("Highest & Lowest Popularity Movies")
    highest = df[df['Popularity'] == df['Popularity'].max()]
    lowest = df[df['Popularity'] == df['Popularity'].min()]
    st.write("🎯 **Most Popular Movie(s):**")
    st.dataframe(highest)
    st.write("🔻 **Least Popular Movie(s):**")
    st.dataframe(lowest)

#  Correlation Heatmap
elif choice == "8 - Correlation Heatmap":
    st.subheader("Correlation Heatmap (Numeric Columns)")
    numeric_df = df.select_dtypes(include=np.number)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

# ======================
# FOOTER
# ======================
st.markdown("""
---
**Instructions:**
- Place `app.py` and `mymoviedb.csv` in the same folder.
- Run using: `streamlit run app.py`
- Use sidebar to explore visualizations.
""")
