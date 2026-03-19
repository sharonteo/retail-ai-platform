"""
Lightweight RAG engine for product search and semantic retrieval.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class LightRAG:
    def __init__(self, product_path="../data/products.csv"):
        root = Path(__file__).resolve().parents[1]
        self.product_path = root / "data" / "products.csv"

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.products = None
        self.embeddings = None

    def load_products(self):
        self.products = pd.read_csv(self.product_path)
        return self.products

    def build_index(self):
        descriptions = self.products["description"].fillna("").tolist()
        self.embeddings = self.model.encode(descriptions, convert_to_numpy=True)
        return self.embeddings

    def search(self, query, top_k=5):
        query_emb = self.model.encode([query], convert_to_numpy=True)
        sims = cosine_similarity(query_emb, self.embeddings)[0]

        top_idx = sims.argsort()[::-1][:top_k]
        results = self.products.iloc[top_idx].copy()
        results["similarity"] = sims[top_idx]
        return results

    def similar_products(self, product_id, top_k=5):
        row = self.products[self.products["product_id"] == product_id]
        if row.empty:
            raise ValueError(f"Product {product_id} not found.")

        desc = row["description"].values[0]
        return self.search(desc, top_k=top_k)