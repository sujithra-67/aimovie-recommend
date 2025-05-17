import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import difflib
import streamlit as st
from io import StringIO

# Embedded dataset
csv_data = """
title,genres,keywords,cast,director
The Matrix,Action,Sci-Fi,Keanu Reeves,Lana Wachowski
Inception,Action,Thriller,Leonardo DiCaprio,Christopher Nolan
Interstellar,Adventure,Space,Matthew McConaughey,Christopher Nolan
The Dark Knight,Action,Hero,Christian Bale,Christopher Nolan
Titanic,Romance,Ship,Leonardo DiCaprio,James Cameron
Avatar,Adventure,Pandora,Sam Worthington,James Cameron
Gladiator,Action,Ancient,Russell Crowe,Ridley Scott
The Prestige,Drama,Magic,Hugh Jackman,Christopher Nolan
The Lion King,Animation,Animals,Matthew Broderick,Roger Allers
Frozen,Animation,Ice,Idina Menzel,Chris Buck
Black Panther,Action,Wakanda,Chadwick Boseman,Ryan Coogler
Avengers: Endgame,Action,Infinity,Robert Downey Jr.,Anthony Russo
Doctor Strange,Action,Magic,Benedict Cumberbatch,Scott Derrickson
Iron Man,Action,Armor,Robert Downey Jr.,Jon Favreau
Up,Animation,Balloon,Ed Asner,Pete Docter
Coco,Animation,Music,Anthony Gonzalez,Lee Unkrich
Shutter Island,Thriller,Mystery,Leonardo DiCaprio,Martin Scorsese
The Revenant,Adventure,Survival,Leonardo DiCaprio,Alejandro G. Iñárritu
Mad Max: Fury Road,Action,Desert,Tom Hardy,George Miller
Joker,Drama,Psychological,Joaquin Phoenix,Todd Phillips
"""

# Load DataFrame
df = pd.read_csv(StringIO(csv_data.strip()))

# Clean and combine features
def clean_data(x):
    return x.lower().strip().replace(" ", "") if isinstance(x, str) else ""

for col in ["genres", "keywords", "cast", "director", "title"]:
    df[col] = df[col].fillna("").apply(clean_data)

df["combined_features"] = df["genres"] + " " + df["keywords"] + " " + df["cast"] + " " + df["director"]

# TF-IDF and Cosine Similarity
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["combined_features"])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Indexing
df = df.reset_index()
indices = pd.Series(df.index, index=df["title"].str.lower().str.strip()).drop_duplicates()

# Match title
def get_closest_match(title):
    possible_titles = df["title"].str.lower().str.strip().tolist()
    matches = difflib.get_close_matches(title.lower().strip(), possible_titles, n=1, cutoff=0.6)
    return matches[0] if matches else None

# Recommendation
def get_recommendations(title):
    matched_title = get_closest_match(title)
    if not matched_title:
        return None, f"Movie '{title}' not found."
    idx = indices[matched_title]
    sim_scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return df["title"].iloc[movie_indices], None

# Accuracy Test
def test_accuracy():
    test_cases = {
        "Inception": "Interstellar",
        "The Dark Knight": "The Prestige",
        "Titanic": "Avatar",
        "Coco": "Up",
        "Iron Man": "Avengers: Endgame",
    }
    hits = 0
    for query, expected in test_cases.items():
        recs, _ = get_recommendations(query)
        if recs is not None and expected.lower().strip() in recs.str.lower().str.strip().tolist():
            hits += 1
    return hits / len(test_cases)

# Streamlit UI
st.title("AI Movie Recommender System")

user_input = st.text_input("Enter a movie title")

col1, col2 = st.columns(2)
search_clicked = col1.button("Get Recommendations")
test_clicked = col2.button("Run Accuracy Test")

if search_clicked:
    if user_input:
        results, error = get_recommendations(user_input)
        if error:
            st.error(error)
        else:
            st.subheader("Top 10 Recommended Movies:")
            for i, title in enumerate(results, 1):
                st.write(f"{i}. {title}")
    else:
        st.warning("Please enter a movie title.")

if test_clicked:
    accuracy = test_accuracy()
    st.success(f"Accuracy on test queries: {accuracy * 100:.2f}%")