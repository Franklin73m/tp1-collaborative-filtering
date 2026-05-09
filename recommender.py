"""
TP1 - Collaborative Filtering Item-Item
Algorithme :
  1. Construire matrice user-item
  2. Similarité cosinus entre items
  3. Top-N items similaires
  4. Prédire scores pour un user
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class ItemItemCF:

    def __init__(self, top_n=10):
        self.top_n = top_n
        self.matrix = None       # matrice user-item
        self.similarity = None   # matrice similarité item-item
        self.movies = None       # infos films

    # ─── 1. Chargement des données ───────────────
    def load_data(self, ratings_path, movies_path):
        ratings = pd.read_csv(ratings_path)
        self.movies = pd.read_csv(movies_path)

        # Garder seulement les colonnes utiles
        ratings = ratings[["userId", "movieId", "rating"]]

        # Filtrer les films avec moins de 10 ratings
        counts = ratings["movieId"].value_counts()
        ratings = ratings[ratings["movieId"].isin(
            counts[counts >= 10].index
        )]

        print(f"[INFO] {ratings['userId'].nunique()} users")
        print(f"[INFO] {ratings['movieId'].nunique()} films")
        print(f"[INFO] {len(ratings)} ratings")
        return ratings

    # ─── 2. Entraînement ─────────────────────────
    def fit(self, ratings_path, movies_path):
        ratings = self.load_data(ratings_path, movies_path)

        # Matrice user x item (NaN = pas de rating)
        self.matrix = ratings.pivot_table(
            index="userId",
            columns="movieId",
            values="rating"
        )

        # Remplacer NaN par 0 pour le calcul
        filled = self.matrix.fillna(0)

        # Similarité cosinus entre films (items en lignes)
        sim = cosine_similarity(filled.T)
        self.similarity = pd.DataFrame(
            sim,
            index=self.matrix.columns,
            columns=self.matrix.columns
        )

        sparsity = 1 - self.matrix.notna().sum().sum() / self.matrix.size
        print(f"[INFO] Sparsité : {sparsity:.1%}")
        print("[INFO] Modèle prêt ✓")

    # ─── 3. Films similaires à un film ───────────
    def similar_items(self, movie_id, n=None):
        n = n or self.top_n
        sims = self.similarity[movie_id].drop(index=movie_id)
        top = sims.sort_values(ascending=False).head(n)

        # Ajouter les titres
        result = top.reset_index()
        result.columns = ["movieId", "similarite"]
        result = result.merge(self.movies[["movieId", "title"]], on="movieId")
        return result[["title", "similarite"]]

    # ─── 4. Recommandations pour un user ─────────
    def recommend(self, user_id, n=None):
        n = n or self.top_n
        user_ratings = self.matrix.loc[user_id].dropna()
        rated_ids    = user_ratings.index.tolist()
        unrated_ids  = [m for m in self.matrix.columns
                        if m not in rated_ids]

        scores = {}
        for movie in unrated_ids:
            sims  = self.similarity.loc[movie, rated_ids]
            num   = (sims * user_ratings[rated_ids]).sum()
            denom = sims.abs().sum()
            scores[movie] = num / denom if denom > 0 else 0

        result = pd.DataFrame(
            list(scores.items()),
            columns=["movieId", "score"]
        ).sort_values("score", ascending=False).head(n)

        result = result.merge(
            self.movies[["movieId", "title", "genres"]], on="movieId"
        )
        return result[["title", "genres", "score"]]

    # ─── 5. Stats ─────────────────────────────────
    def stats(self):
        return {
            "users":   len(self.matrix.index),
            "films":   len(self.matrix.columns),
            "ratings": int(self.matrix.notna().sum().sum()),
            "sparsite": f"{(1 - self.matrix.notna().sum().sum()/self.matrix.size):.1%}",
            "note_moy": round(self.matrix.stack().mean(), 2),
        }