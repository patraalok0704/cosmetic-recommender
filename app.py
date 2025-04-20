import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Load and prepare dataset
df = pd.read_csv("D:\ML\cosmetics.csv")
df.dropna(subset=["Ingredients", "Price", "Rank"], inplace=True)
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["Rank"] = pd.to_numeric(df["Rank"], errors="coerce")
df.dropna(subset=["Price", "Rank"], inplace=True)
df.reset_index(drop=True, inplace=True)

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df["Ingredients"])

def recommend_by_ingredient(input_ingredient, top_n=10, sort_by="Price", ascending=True):
    input_vec = tfidf.transform([input_ingredient])
    cosine_sim = cosine_similarity(input_vec, tfidf_matrix).flatten()
    df['similarity'] = cosine_sim
    similar_df = df[df['similarity'] > 0.1].copy()

    if sort_by.lower() == "price":
        similar_df = similar_df.sort_values(by=["Price", "similarity"], ascending=[ascending, False])
    elif sort_by.lower() == "rank":
        similar_df = similar_df.sort_values(by=["Rank", "similarity"], ascending=[ascending, False])
    else:
        similar_df = similar_df.sort_values(by="similarity", ascending=False)

    return similar_df[["Brand", "Name", "Price", "Rank", "Ingredients"]].head(top_n)

def plot_market_insights():
    avg_price = df.groupby("Label")["Price"].mean().sort_values()
    avg_rank = df.groupby("Label")["Rank"].mean().sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    avg_price.plot(kind='barh', color='skyblue', ax=axes[0])
    axes[0].set_title("Average Price by Product Type")
    axes[0].set_xlabel("Average Price")

    avg_rank.plot(kind='barh', color='salmon', ax=axes[1])
    axes[1].set_title("Average Rating by Product Type")
    axes[1].set_xlabel("Average Rating")

    st.pyplot(fig)

# Streamlit UI

st.title("🧴 Cosmetic Product Recommender")

with st.sidebar:
    st.header("Input Preferences")
    user_input = st.text_input("Enter ingredients (comma-separated):", "glycerin")
    sort_by = st.radio("Sort by:", ["Price", "Rank"])
    order = st.radio("Order:", ["Ascending", "Descending"])
    show_chart = st.checkbox("Show Market Insights")

ascending = True if order == "Ascending" else False

if user_input:
    st.subheader(f"Top Recommendations for: *{user_input}*")
    recommendations = recommend_by_ingredient(user_input, sort_by=sort_by, ascending=ascending)
    st.dataframe(recommendations)

if show_chart:
    st.subheader("📊 Market Insights")
    plot_market_insights()
