"""metmap.py — backward-compatible shim for the split ``pro`` engine package.

The engine now lives package-by-package under ``pro/`` (see pro/__init__.py).
This module re-exports the same public names so existing ``import metmap``
call sites keep working unchanged.
"""
from pro import *  # noqa: F401,F403
from pro import (  # noqa: F401
    config, data, physics, products, compute, render, engine,
    DEFAULT_N_DAYS, DEFAULT_PRODUCT,
    PRODUCTS, GROUP_ORDER, list_products, group_products,
    generate, generate_diff,
)

__all__ = [
    "config", "data", "physics", "products", "compute", "render", "engine",
    "DEFAULT_N_DAYS", "DEFAULT_PRODUCT",
    "PRODUCTS", "GROUP_ORDER", "list_products", "group_products",
    "generate", "generate_diff",
]
