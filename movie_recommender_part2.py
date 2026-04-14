# ============================================================
# MOVIE RECOMMENDER — PART 2
# Collaborative Filtering (SVD) + Hybrid System + Streamlit UI
# Depends on: movie_recommender_part1.py
# ============================================================

import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Re-use loaders from Part 1
from movie_recommender_part1 import (
    load_data,
    ContentBasedRecommender,
    enrich_with_ratings,
)


# ─────────────────────────────────────────────
# 1. COLLABORATIVE FILTERING  (SVD)
# ─────────────────────────────────────────────

class CollaborativeRecommender:
    """
    Matrix Factorisation via TruncatedSVD.

    Decomposes the user-movie rating matrix (filled with 0 for unrated)
    into latent factors, then reconstructs predicted ratings for every
    (user, movie) pair.

    Usage
    -----
    cfr = CollaborativeRecommender(movies_df, ratings_df)
    cfr.fit(n_components=50)
    recs = cfr.recommend(user_id=42, n=5)
    rmse = cfr.evaluate()
    """

    def __init__(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame):
        self.movies         = movies_df.copy()
        self.ratings        = ratings_df.copy()
        self._user_matrix   = None          # raw pivot table
        self._predicted     = None          # reconstructed ratings (dense)
        self._svd           = None
        self._movie_id_list = None          # ordered list of movieIds in matrix

    # ── Fit ───────────────────────────────────

    def fit(self, n_components: int = 50):
        """Build user-movie matrix and apply SVD."""
        print(f"\nFitting SVD collaborative model  (n_components={n_components}) …")

        # Pivot: rows = users, columns = movies
        pivot = self.ratings.pivot_table(
            index='userId',
            columns='movieId',
            values='rating',
            aggfunc='mean',
        ).fillna(0)

        self._user_matrix   = pivot
        self._movie_id_list = list(pivot.columns)

        # Decompose
        self._svd    = TruncatedSVD(n_components=n_components, random_state=42)
        latent       = self._svd.fit_transform(pivot.values)

        # Reconstruct full matrix
        self._predicted = np.dot(latent, self._svd.components_)

        explained = self._svd.explained_variance_ratio_.sum()
        print(f"SVD fitted  |  explained variance: {explained:.2%}  |  "
              f"matrix shape: {pivot.shape}")
        return self

    # ── Recommend ─────────────────────────────

    def recommend(self, user_id: int, n: int = 5) -> pd.DataFrame:
        """
        Return top-n movies predicted to be highest-rated
        by `user_id` that they haven't yet rated.
        """
        if self._predicted is None:
            raise RuntimeError("Call .fit() before .recommend()")

        valid_ids = list(self._user_matrix.index)
        if user_id not in valid_ids:
            print(f"user_id {user_id} not found. "
                  f"Valid range: {min(valid_ids)} – {max(valid_ids)}")
            return pd.DataFrame()

        user_row_idx   = list(self._user_matrix.index).index(user_id)
        pred_ratings   = self._predicted[user_row_idx]

        # Mask already-rated movies
        actual_ratings = self._user_matrix.iloc[user_row_idx].values
        unrated_mask   = actual_ratings == 0

        # Sort by predicted rating
        scored = [
            (self._movie_id_list[i], pred_ratings[i])
            for i in range(len(pred_ratings))
            if unrated_mask[i]
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        top_ids = [mid for mid, _ in scored[:n]]
        top_sc  = [sc  for _, sc  in scored[:n]]

        result = self.movies[self.movies['movieId'].isin(top_ids)][['movieId', 'title', 'genres']].copy()
        score_map = dict(zip(top_ids, top_sc))
        result['predicted_rating'] = result['movieId'].map(score_map).round(3)
        result = (
            result
            .sort_values('predicted_rating', ascending=False)
            .drop(columns='movieId')
            .reset_index(drop=True)
        )
        result.insert(0, 'rank', range(1, len(result) + 1))

        print(f"\nCollaborative recommendations for  →  User {user_id}")
        return result

    # ── Display ───────────────────────────────

    def display_recommendations(self, user_id: int, n: int = 5):
        recs = self.recommend(user_id, n)
        if recs.empty:
            return
        print(f"\n{'Rank':<5} {'Title':<50} {'Genres':<35} {'Pred ★':>7}")
        print("─" * 100)
        for _, row in recs.iterrows():
            print(f"{row['rank']:<5} {row['title']:<50} "
                  f"{row['genres']:<35} {row['predicted_rating']:>7.3f}")
        return recs

    # ── Evaluate ──────────────────────────────

    def evaluate(self, sample_frac: float = 0.10) -> float:
        """
        Hold out `sample_frac` of known ratings and compute RMSE
        between actual and SVD-predicted values.
        """
        known = self.ratings[['userId', 'movieId', 'rating']].copy()
        _, test_df = train_test_split(known, test_size=sample_frac,
                                      random_state=42)

        actuals    = []
        predicted  = []

        user_idx   = {uid: i for i, uid in enumerate(self._user_matrix.index)}
        movie_idx  = {mid: i for i, mid in enumerate(self._movie_id_list)}

        for _, row in test_df.iterrows():
            uid = int(row['userId'])
            mid = int(row['movieId'])
            if uid in user_idx and mid in movie_idx:
                actuals.append(row['rating'])
                predicted.append(self._predicted[user_idx[uid], movie_idx[mid]])

        rmse = np.sqrt(mean_squared_error(actuals, predicted))
        print(f"\nCollaborative model RMSE: {rmse:.4f}  "
              f"(evaluated on {len(actuals):,} held-out ratings)")
        return rmse


# ─────────────────────────────────────────────
# 2. HYBRID RECOMMENDER
# ─────────────────────────────────────────────

class HybridRecommender:
    """
    Combines content-based and collaborative filtering.

    Strategy
    --------
    • If both a movie title AND a user_id are supplied:
        Score = α × content_sim + (1-α) × normalised_predicted_rating
    • If only movie title → pure content-based.
    • If only user_id     → pure collaborative.
    """

    def __init__(self,
                 content_model: ContentBasedRecommender,
                 collab_model:  CollaborativeRecommender,
                 movies_df:     pd.DataFrame,
                 alpha: float = 0.5):
        self.content = content_model
        self.collab  = collab_model
        self.movies  = movies_df.copy()
        self.alpha   = alpha

    def recommend(self, movie_title: str = None,
                  user_id: int = None,
                  n: int = 5) -> pd.DataFrame:

        if movie_title and user_id:
            return self._hybrid(movie_title, user_id, n)
        elif movie_title:
            return self.content.recommend(movie_title, n)
        elif user_id:
            return self.collab.recommend(user_id, n)
        else:
            print("Provide at least a movie_title or user_id.")
            return pd.DataFrame()

    def _hybrid(self, movie_title: str, user_id: int, n: int) -> pd.DataFrame:
        """Weighted combination of both signals."""
        c_recs = self.content.recommend(movie_title, n * 3)
        u_recs = self.collab.recommend(user_id,      n * 3)

        if c_recs.empty or u_recs.empty:
            # Fall back to whichever is available
            return c_recs if not c_recs.empty else u_recs

        # Normalise scores to [0, 1]
        c_recs['c_score'] = (c_recs['similarity_score'] /
                             c_recs['similarity_score'].max())
        u_recs['u_score'] = (u_recs['predicted_rating'] /
                             u_recs['predicted_rating'].max())

        merged = c_recs[['title', 'genres', 'c_score']].merge(
            u_recs[['title', 'u_score']], on='title', how='outer'
        ).fillna(0)

        merged['hybrid_score'] = (
            self.alpha * merged['c_score'] +
            (1 - self.alpha) * merged['u_score']
        )

        result = (
            merged
            .sort_values('hybrid_score', ascending=False)
            .head(n)
            .reset_index(drop=True)
        )
        result.insert(0, 'rank', range(1, len(result) + 1))

        print(f"\nHybrid recommendations  →  "
              f"movie='{movie_title}'  user={user_id}  (α={self.alpha})")
        return result[['rank', 'title', 'genres', 'hybrid_score']]


# ─────────────────────────────────────────────
# 3. COMMAND-LINE INTERACTIVE MODE
# ─────────────────────────────────────────────

def run_cli(movies_df, ratings_df,
            content_model, collab_model, hybrid_model):
    """Simple REPL for terminal testing."""
    print("\n" + "=" * 55)
    print("  MOVIE RECOMMENDATION SYSTEM  ")
    print("=" * 55)

    while True:
        print("\nChoose mode:")
        print("  1  Content-based  (enter a movie name)")
        print("  2  Collaborative  (enter a user ID)")
        print("  3  Hybrid         (movie + user)")
        print("  4  Exit")

        choice = input("\n> ").strip()

        if choice == '1':
            title = input("Movie title: ").strip()
            content_model.display_recommendations(title)

        elif choice == '2':
            try:
                uid = int(input("User ID: ").strip())
                collab_model.display_recommendations(uid)
            except ValueError:
                print("Enter a numeric user ID.")

        elif choice == '3':
            title = input("Movie title: ").strip()
            try:
                uid   = int(input("User ID: ").strip())
                recs  = hybrid_model.recommend(title, uid)
                print(recs.to_string(index=False))
            except ValueError:
                print("Enter a numeric user ID.")

        elif choice == '4':
            print("\n👋 Bye!")
            break
        else:
            print("Invalid choice.")


# ─────────────────────────────────────────────
# 4. STREAMLIT UI  (run with: streamlit run movie_recommender_part2.py)
# ─────────────────────────────────────────────

def launch_streamlit():
    """
    Streamlit front-end.
    Only imported/executed when the script is run via `streamlit run`.
    """
    import streamlit as st

    st.set_page_config(page_title="🎬 Movie Recommender", layout="wide")
    st.title("Movie Recommendation System")
    st.caption("Powered by MovieLens · Content-Based + Collaborative Filtering")

    # ── Load & cache ──────────────────────────
    @st.cache_data(show_spinner="Loading dataset …")
    def get_data():
        return load_data()

    @st.cache_resource(show_spinner="Building models …")
    def get_models(movies_df, ratings_df):
        cbr = ContentBasedRecommender(movies_df).fit()
        cfr = CollaborativeRecommender(movies_df, ratings_df).fit()
        hybrid = HybridRecommender(cbr, cfr, movies_df)
        return cbr, cfr, hybrid

    try:
        movies_df, ratings_df = get_data()
        cbr, cfr, hybrid      = get_models(movies_df, ratings_df)
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        st.stop()

    # ── Sidebar controls ──────────────────────
    st.sidebar.header("Settings")
    mode = st.sidebar.radio(
        "Recommendation mode",
        ["Content-Based", "Collaborative", "Hybrid"],
    )
    n_recs = st.sidebar.slider("Number of recommendations", 3, 15, 5)

    # ── Main inputs ───────────────────────────
    col1, col2 = st.columns(2)

    movie_input = None
    user_input  = None

    if mode in ("Content-Based", "Hybrid"):
        with col1:
            movie_input = st.text_input(
                "Movie title",
                placeholder="e.g. Toy Story, Inception, The Matrix",
            )

    if mode in ("Collaborative", "Hybrid"):
        with col2:
            valid_users = sorted(ratings_df['userId'].unique())
            user_input  = st.selectbox(
                "👤 User ID",
                options=valid_users,
                index=0,
            )

    # ── Run ───────────────────────────────────
    if st.button("✨ Get Recommendations", type="primary"):
        with st.spinner("Finding your movies …"):
            if mode == "Content-Based" and movie_input:
                recs = cbr.recommend(movie_input, n=n_recs)
            elif mode == "Collaborative" and user_input:
                recs = cfr.recommend(user_input, n=n_recs)
            elif mode == "Hybrid":
                recs = hybrid.recommend(movie_input, user_input, n=n_recs)
            else:
                st.warning("Please fill in the required field(s).")
                st.stop()

        if recs.empty:
            st.error("No recommendations found. Try a different title or user.")
        else:
            st.success(f"Top {len(recs)} recommendations:")
            st.dataframe(recs, use_container_width=True, hide_index=True)

            # Popularity overlay
            if 'avg_rating' not in recs.columns:
                recs = recs.merge(
                    enrich_with_ratings(movies_df, ratings_df)[['title', 'avg_rating', 'vote_count']],
                    on='title', how='left'
                )
            if 'avg_rating' in recs.columns:
                st.bar_chart(
                    recs.set_index('title')['avg_rating'].dropna(),
                    height=220,
                )

    # ── Stats sidebar ─────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.metric("Total movies",  f"{len(movies_df):,}")
    st.sidebar.metric("Total ratings", f"{len(ratings_df):,}")
    st.sidebar.metric("Total users",   f"{ratings_df['userId'].nunique():,}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Detect if called via `streamlit run`
    if "streamlit" in sys.modules or any("streamlit" in arg for arg in sys.argv):
        launch_streamlit()
    else:
        # CLI mode
        print("Loading data and fitting models …")
        movies_df, ratings_df = load_data()

        cbr    = ContentBasedRecommender(movies_df).fit()
        cfr    = CollaborativeRecommender(movies_df, ratings_df).fit()
        hybrid = HybridRecommender(cbr, cfr, movies_df)

        # Evaluate collaborative model
        cfr.evaluate()

        # Quick demos
        print("\n" + "="*55 + "\n  SAMPLE RECOMMENDATIONS\n" + "="*55)
        cbr.display_recommendations("Toy Story", n=5)
        cfr.display_recommendations(1, n=5)
        recs = hybrid.recommend("Inception", user_id=42, n=5)
        if not recs.empty:
            print(recs.to_string(index=False))

        # Start REPL
        run_cli(movies_df, ratings_df, cbr, cfr, hybrid)