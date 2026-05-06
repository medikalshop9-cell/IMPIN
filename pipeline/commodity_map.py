"""
pipeline/commodity_map.py
Maps scraped product names to CPI basket subcategories using keyword matching.
Falls back to the top-level cpi_category already assigned by url.csv.
"""

import re
from pathlib import Path
from typing import Optional

import yaml
import pandas as pd


_BASKET_PATH = Path(__file__).parent.parent / "config" / "cpi_basket.yaml"


def _load_basket() -> dict:
    with open(_BASKET_PATH, encoding="utf-8") as fh:
        return yaml.safe_load(fh)["basket"]


def _build_keyword_index(basket: dict) -> list[tuple[str, str, list[str]]]:
    """Return list of (top_category, subcategory_label, [keywords]) tuples."""
    index = []
    for top_cat, top_data in basket.items():
        for sub_key, sub_data in top_data.get("subcategories", {}).items():
            keywords = [kw.lower() for kw in sub_data.get("keywords", [])]
            if keywords:
                index.append((top_cat, sub_data["label"], keywords))
    return index


_BASKET = _load_basket()
_KEYWORD_INDEX = _build_keyword_index(_BASKET)


def map_product(product_name: str, cpi_category: str) -> tuple[str, str]:
    """
    Given a product name and its top-level cpi_category, return
    (resolved_category, subcategory_label).

    Strategy:
    1. Try keyword match on product name within the assigned cpi_category first.
    2. Try keyword match across all categories.
    3. Fall back to (cpi_category, cpi_category).
    """
    name_lower = product_name.lower()

    # Step 1: match within assigned category
    for top_cat, sub_label, keywords in _KEYWORD_INDEX:
        if top_cat == cpi_category:
            if any(re.search(rf"\b{re.escape(kw)}\b", name_lower) for kw in keywords):
                return (top_cat, sub_label)

    # Step 2: match across all categories
    for top_cat, sub_label, keywords in _KEYWORD_INDEX:
        if any(re.search(rf"\b{re.escape(kw)}\b", name_lower) for kw in keywords):
            return (top_cat, sub_label)

    # Step 3: fallback
    return (cpi_category, cpi_category)


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add `resolved_category` and `subcategory` columns to a scraped price DataFrame.
    Operates in-place on a copy; does not modify the original.
    """
    df = df.copy()
    results = df.apply(
        lambda row: map_product(
            str(row.get("product_name", "")),
            str(row.get("cpi_category", "General")),
        ),
        axis=1,
    )
    df["resolved_category"] = results.apply(lambda t: t[0])
    df["subcategory"] = results.apply(lambda t: t[1])
    return df


if __name__ == "__main__":
    # Quick smoke test
    test_cases = [
        ("Rice (5kg bag)", "Food & Beverages"),
        ("Chicken breast boneless 1kg", "Food & Beverages"),
        ("Samsung Galaxy A15", "General"),
        ("Omo washing powder 500g", "Household"),
        ("Original Kente", "Clothing & Personal Care"),
        ("Tomatoes 1kg", "Food & Beverages"),
        ("Palm oil 1 litre", "Food & Beverages"),
    ]
    print(f"{'Product':<45} {'Category':<25} {'Subcategory'}")
    print("-" * 100)
    for name, cat in test_cases:
        resolved, sub = map_product(name, cat)
        print(f"{name:<45} {resolved:<25} {sub}")
