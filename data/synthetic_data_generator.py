"""
Generate synthetic product, customer, and sales data.
"""

import pandas as pd
import numpy as np


def generate_products(n=200):
    categories = ["Boots", "Jeans", "Hats", "Shirts", "Accessories"]
    brands = ["Ariat", "Justin", "Wrangler", "Carhartt", "Tony Lama"]

    # Generate descriptions with richer text for embeddings
    random_brands = np.random.choice(brands, n)
    random_categories = np.random.choice(categories, n)

    df = pd.DataFrame({
        "product_id": range(1, n + 1),
        "name": [f"Product {i}" for i in range(1, n + 1)],
        "category": random_categories,
        "brand": random_brands,
        "price": np.random.uniform(40, 300, n).round(2),
        "description": [
            f"{brand} {category.lower()} designed for comfort, durability, and everyday wear."
            for brand, category in zip(random_brands, random_categories)
        ]
    })
    return df


def generate_customers(n=500):
    df = pd.DataFrame({
        "customer_id": range(1, n + 1),
        "age": np.random.randint(18, 70, n),
        "location": np.random.choice(["CA", "TX", "AZ", "NV", "CO"], n),
    })
    return df


def generate_sales(products, customers, weeks=12):
    rows = []
    for week in range(1, weeks + 1):
        for _, p in products.iterrows():
            demand = np.random.poisson(3)
            for _ in range(demand):
                rows.append({
                    "week": week,
                    "product_id": p.product_id,
                    "customer_id": np.random.choice(customers.customer_id),
                    "quantity": 1
                })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Generating synthetic datasets...")

    products = generate_products()
    customers = generate_customers()
    sales = generate_sales(products, customers)

    products.to_csv("products.csv", index=False)
    customers.to_csv("customers.csv", index=False)
    sales.to_csv("sales.csv", index=False)

    print("Done! Created products.csv, customers.csv, and sales.csv")