"""
Personalization module: product embeddings + simple recommender.

This module:
1. Loads product data
2. Generates embeddings using SentenceTransformers
3. Computes similarity between products
4. Returns top-k recommendations
"""
import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class PersonalizationEngine:
    def __init__(self, product_path="../data/products.csv"):
        # Convert relative path to absolute path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.product_path = os.path.abspath(os.path.join(base_dir, "..", "data", "products.csv"))

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.products = None
        self.embeddings = None


    def load_products(self):
        """Load product catalog."""
        self.products = pd.read_csv(self.product_path)
        return self.products

    def build_embeddings(self):
        """Generate embeddings from product descriptions."""
        if self.products is None:
            self.load_products()

        descriptions = self.products["description"].tolist()
        self.embeddings = self.model.encode(descriptions)
        return self.embeddings

    def recommend(self, product_id, top_k=5):
        """
        Recommend similar products based on embedding similarity.
        """
        if self.embeddings is None:
            self.build_embeddings()

        # Find index of the product
        idx = self.products.index[self.products["product_id"] == product_id][0]

        # Compute similarity
        sims = cosine_similarity(
            [self.embeddings[idx]],
            self.embeddings
        )[0]

        # Sort and get top-k
        top_indices = sims.argsort()[::-1][1 : top_k + 1]

        return self.products.iloc[top_indices][["product_id", "name", "category", "brand", "price"]]

    # ---------------------------------------------------------
    # NEW: Simple user-level recommendation method
    # ---------------------------------------------------------
    def recommend_products(self, user_id=1, top_k=5):
        """
        Simple category-based recommendation for a user.
        Uses product embeddings only indirectly (via categories).
        """
        if self.products is None:
            self.load_products()

        # Very lightweight user preference profile
        # (You can expand this later)
        user_preferences = {
            "Boots": 0.9,
            "Jeans": 0.6,
            "Shirts": 0.3
        }

        # Score products by category preference
        self.products["score"] = self.products["category"].apply(
            lambda c: user_preferences.get(c, 0)
        )

        # Return top-k highest scoring products
        result = (
            self.products.sort_values("score", ascending=False)
            .head(top_k)
            .drop(columns=["score"])
        )

        return result