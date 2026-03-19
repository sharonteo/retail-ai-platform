"""
Lightweight AI Shopping Assistant

Thin orchestration layer over:
- LightRAG (semantic product search + similar items)
- ForecastingEngine (demand forecasting)
- PersonalizationEngine (simple recommendations)
"""

import re

from src.rag_pipeline import LightRAG
from src.forecasting import ForecastingEngine
from src.personalization import PersonalizationEngine


class ShoppingAssistant:
    def __init__(self):
        # Core engines
        self.rag = LightRAG()
        self.forecaster = ForecastingEngine()
        self.personalizer = PersonalizationEngine()

        # Init data + models
        self.rag.load_products()
        self.rag.build_index()

        self.forecaster.load_sales()
        self.forecaster.train()

    # -----------------------------
    # Very simple intent routing
    # -----------------------------
    def _detect_intent(self, query: str) -> str:
        q = query.lower()

        if "similar" in q:
            return "similar"
        if "forecast" in q or "demand" in q or "predict" in q:
            return "forecast"
        if "recommend" in q or "suggest" in q:
            return "recommend"

        return "search"

    def _extract_product_id(self, query: str):
        match = re.search(r"\b(\d+)\b", query)
        return int(match.group(1)) if match else None

    # -----------------------------
    # Public entrypoint
    # -----------------------------
    def answer(self, query: str):
        intent = self._detect_intent(query)
        product_id = self._extract_product_id(query)

        if intent == "similar":
            if product_id is None:
                return "Please include a product ID to find similar items."
            return self.rag.similar_products(product_id, top_k=5)

        if intent == "forecast":
            if product_id is None:
                return "Please include a product ID to forecast demand."
            return self.forecaster.predict(product_id=product_id, weeks_ahead=12)

        if intent == "recommend":
            # Simple: recommend for a dummy user_id=1
            return self.personalizer.recommend_products(user_id=1, top_k=5)

        # Default: semantic product search
        return self.rag.search(query, top_k=5)