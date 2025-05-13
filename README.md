

# 🎬 AI Movie Recommendation System

This is a simple content-based movie recommender system built using Python and scikit-learn. It uses TF-IDF vectorization to find similar movies based on features such as genre, cast, keywords, and director.

---

## 📂 Dataset

The system uses a subset of a movie metadata dataset (CSV format). The file should be named `movies1.csv` and located in the same directory as the script.

**Expected columns:**

* `title`
* `genres`
* `keywords`
* `cast`
* `director`

---

## ⚙️ Features

* Cleans and combines relevant metadata features.
* Computes cosine similarity between movies based on TF-IDF vectors.
* Uses fuzzy matching to handle minor typos in movie titles.
* Recommends top 10 similar movies to the one entered by the user.

---

## 🧪 Requirements

* Python 3.x
* pandas
* scikit-learn

Install the dependencies using:

```bash
pip install pandas scikit-learn
```

---

## 🚀 How to Run

1. Make sure your dataset (`movies1.csv`) is in the same directory.
2. Run the script:

```bash
python movie_recommender.py
```

3. Enter a movie title when prompted.
4. Receive top 10 recommended movies!

---

## 📌 Example

```text
AI Movie Recommender System
Enter a movie title: The Matrix

Top 10 Recommendations:

1. The Matrix Reloaded
2. Inception
3. Equilibrium
4. The Matrix Revolutions
5. Blade Runner
...
```

---

## 📎 Notes

* Only the first 1000 rows of the dataset are processed for performance reasons. You can adjust the `chunk_size` variable as needed.
* The similarity is computed using cosine similarity over TF-IDF vectors of combined textual features.

---

## 🛠️ Customization

To include more features or preprocess data differently, update the `clean_data` function and the `combined_features` construction accordingly.

---

## 🧑‍💻 Author

Developed as a basic AI project using Python and scikit-learn.


