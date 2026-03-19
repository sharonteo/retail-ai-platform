## Retail AI Platform
Live Demo: https://retail-ai-platform.streamlit.app/ 
A modular GenAI system for chatbot‑based product search, recommendations, forecasting, and retail assistance.

```
Example Queries
- “Show me some boots for ranch work”
- “Forecast demand for Product 123”
- “Find me something comfortable for long days on my feet”
- “Show me lightweight options for summer”
- “What are some durable work boots”
- “Find me jeans for riding”
- “What’s trending right now”
```

## Overview
The Retail AI Platform integrates retrieval‑augmented generation (RAG), semantic search, and machine‑learning forecasting into a unified chatbot‑driven retail assistant. Users can ask natural‑language questions, explore products, and request demand forecasts through a streamlined Streamlit interface.
The project demonstrates clean architecture, reproducible ML workflows, and practical LLM orchestration for real retail use cases.

## Features
Conversational Shopping Chatbot
- Natural‑language interface for product exploration
- Multi‑turn conversation support
- Retrieves relevant product context using semantic search
- Generates grounded responses based on retrieved data
Semantic Product Search (RAG)
- Embeds product descriptions using SentenceTransformers
- Performs similarity search to surface relevant items
- Returns product matches based on semantic meaning
Demand Forecasting
- XGBoost regression for SKU‑level or category‑level forecasting
- Trend visualization and future sales prediction
Streamlit Web App
- Clean, responsive UI
- Real‑time chatbot interaction
- Forecasting charts and product tables
- Deployed on Streamlit Cloud

```
Architecture
retail-ai-platform/
│
├── app/
│   └── app.py                # Streamlit interface + chatbot UI
│
├── src/
│   ├── assistant.py          # Chatbot orchestration + intent handling
│   ├── rag_pipeline.py       # Embeddings + semantic search
│   ├── forecasting.py        # XGBoost forecasting engine
│   └── utils.py              # Shared helpers
│
├── data/                     # Product and sales datasets
├── requirements.txt
├── runtime.txt
└── README.md

```

Tech Stack
- Streamlit
- SentenceTransformers, Transformers
- XGBoost, scikit‑learn
- Pandas, NumPy

```
Install dependencies:
pip install -r requirements.txt
```

Launch the app:
streamlit run app/app.py


Data
products.csv
- product_id, name, category, brand, price, description
sales.csv
- date, product_id, units_sold
customers.csv
- customer_id, segment, preferences


