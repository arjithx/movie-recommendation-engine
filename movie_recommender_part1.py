# ============================================================
# MOVIE RECOMMENDER — PART 1
# Data Loading + Content-Based Filtering (TF-IDF + Cosine Similarity)
# ============================================================

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

# Official MovieLens small dataset (ml-latest-small)
MOVIES_URL  = "movies.csv"
RATINGS_URL = "ratings.csv"


def load_data(movies_path: str = MOVIES_URL,
              ratings_path: str = RATINGS_URL) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load MovieLens movies.csv and ratings.csv.

    Falls back to local files if the URLs are unreachable.
    Returns (movies_df, ratings_df).
    """
    print("Loading MovieLens dataset …")

    try:
        movies_df  = pd.read_csv(movies_path)
        ratings_df = pd.read_csv(ratings_path)
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not load dataset.\n"
            f"Download ml-latest-small from https://grouplens.org/datasets/movielens/latest/\n"
            f"Then pass the local file paths to load_data().\n"
            f"Original error: {exc}"
        )

    # Basic validation
    for col in ('movieId', 'title', 'genres'):
        if col not in movies_df.columns:
            raise ValueError(f"Expected column '{col}' in movies file.")

    movies_df  = movies_df.dropna(subset=['title', 'genres']).reset_index(drop=True)
    ratings_df = ratings_df.dropna(subset=['userId', 'movieId', 'rating']).reset_index(drop=True)

    print(f"{len(movies_df):,} movies  |  {len(ratings_df):,} ratings  |  "
          f"{ratings_df['userId'].nunique():,} users")

    return movies_df, ratings_df


# ─────────────────────────────────────────────
# 2. CONTENT-BASED FILTERING
# ─────────────────────────────────────────────

class ContentBasedRecommender:
    """
    TF-IDF on movie genres + cosine similarity.

    Usage
    -----
    cbr = ContentBasedRecommender(movies_df)
    cbr.fit()
    recs = cbr.recommend("Toy Story", n=5)
    """

    def __init__(self, movies_df: pd.DataFrame):
        self.movies   = movies_df.copy().reset_index(drop=True)
        self._tfidf   = None
        self._sim     = None
        self._idx_map = None          # title (lower) → row index

    # ── Fit ───────────────────────────────────

    def fit(self):
        """Build TF-IDF matrix and cosine similarity matrix."""
        print("\nBuilding TF-IDF content model …")

        # Replace '|' separators with spaces so each genre is a term
        genre_corpus = (
            self.movies['genres']
            .str.replace('|', ' ', regex=False)
            .str.replace('(no genres listed)', '', regex=False)
            .str.lower()
        )

        self._tfidf = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 1),
            min_df=1,
            stop_words='english',
        )
        tfidf_matrix = self._tfidf.fit_transform(genre_corpus)

        # linear_kernel is faster than cosine_similarity for TF-IDF
        self._sim = linear_kernel(tfidf_matrix, tfidf_matrix)

        # Build a reverse index: cleaned title → position
        self._idx_map = {
            title.lower().strip(): idx
            for idx, title in enumerate(self.movies['title'])
        }

        print(f"TF-IDF matrix  {tfidf_matrix.shape}  |  "
              f"similarity matrix  {self._sim.shape}")
        return self

    # ── Search helper ─────────────────────────

    def _find_index(self, query: str) -> int | None:
        """Fuzzy match: exact → startswith → substring."""
        q = query.lower().strip()
        if q in self._idx_map:
            return self._idx_map[q]
        # startswith
        for title, idx in self._idx_map.items():
            if title.startswith(q):
                return idx
        # substring
        for title, idx in self._idx_map.items():
            if q in title:
                return idx
        return None

    # ── Recommend ─────────────────────────────

    def recommend(self, movie_title: str, n: int = 5) -> pd.DataFrame:
        """
        Return top-n movies most similar to `movie_title`.

        Returns a DataFrame with columns:
          rank, title, genres, similarity_score
        """
        if self._sim is None:
            raise RuntimeError("Call .fit() before .recommend()")

        idx = self._find_index(movie_title)
        if idx is None:
            print(f" '{movie_title}' not found in the dataset.")
            return pd.DataFrame()

        matched_title = self.movies.iloc[idx]['title']
        print(f"\n🎯 Content-based recommendations for  →  '{matched_title}'")

        sim_scores = list(enumerate(self._sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Exclude the query movie itself (score == 1.0 at position 0)
        sim_scores = [s for s in sim_scores if s[0] != idx][:n]

        rec_indices = [s[0] for s in sim_scores]
        rec_scores  = [s[1] for s in sim_scores]

        result = self.movies.iloc[rec_indices][['title', 'genres']].copy()
        result.insert(0, 'rank', range(1, len(result) + 1))
        result['similarity_score'] = [round(s, 4) for s in rec_scores]
        result = result.reset_index(drop=True)

        return result

    # ── Pretty print ──────────────────────────

    def display_recommendations(self, movie_title: str, n: int = 5):
        recs = self.recommend(movie_title, n)
        if recs.empty:
            return

        print(f"\n{'Rank':<5} {'Title':<50} {'Genres':<35} {'Score':>7}")
        print("─" * 100)
        for _, row in recs.iterrows():
            print(f"{row['rank']:<5} {row['title']:<50} "
                  f"{row['genres']:<35} {row['similarity_score']:>7.4f}")
        return recs


# ─────────────────────────────────────────────
# 3. ENRICHED CONTENT MODEL
#    (add average rating & popularity weight)
# ─────────────────────────────────────────────

def enrich_with_ratings(movies_df: pd.DataFrame,
                        ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach mean rating and vote count to each movie.
    Useful for tie-breaking or weighted ranking.
    """
    stats = (
        ratings_df
        .groupby('movieId')['rating']
        .agg(avg_rating='mean', vote_count='count')
        .reset_index()
    )
    enriched = movies_df.merge(stats, on='movieId', how='left')
    enriched['avg_rating']  = enriched['avg_rating'].fillna(0).round(2)
    enriched['vote_count']  = enriched['vote_count'].fillna(0).astype(int)
    return enriched


# ─────────────────────────────────────────────
# 4. POPULAR MOVIES OVERVIEW
# ─────────────────────────────────────────────

def top_popular_movies(movies_df: pd.DataFrame,
                       ratings_df: pd.DataFrame,
                       n: int = 10,
                       min_votes: int = 50) -> pd.DataFrame:
    """Return the n most popular well-rated movies."""
    enriched = enrich_with_ratings(movies_df, ratings_df)
    filtered = enriched[enriched['vote_count'] >= min_votes]
    top = (
        filtered
        .sort_values(['avg_rating', 'vote_count'], ascending=False)
        .head(n)
        [['title', 'genres', 'avg_rating', 'vote_count']]
        .reset_index(drop=True)
    )
    top.index += 1
    return top


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # 1. Load data
    movies_df, ratings_df = load_data()

    # 2. Show popular movies
    print("\nTop 10 Popular Movies:")
    popular = top_popular_movies(movies_df, ratings_df, n=10)
    print(popular.to_string())

    # 3. Fit content-based model
    cbr = ContentBasedRecommender(movies_df)
    cbr.fit()

    # 4. Demo recommendations
    for query in ["Toy Story", "Inception", "The Matrix"]:
        cbr.display_recommendations(query, n=5)
        print()

    print("\nPart 1 complete.")