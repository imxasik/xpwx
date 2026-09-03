"""pro — the XPWEATHER reanalysis map engine, split into focused modules.

Modules
-------
config    module constants & process-wide caches (datasets, fields, lat/lon, coast)
data      OPeNDAP access, cached field means, coastline, date resolution
physics   pure spherical/physical operators (no data access)
products  PRODUCTS registry + sidebar grouping
compute   per-product computation (obs minus 1991-2020 climatology)
render    matplotlib renderers (maps, Hovmöller, Rossby-wave composite)
engine    top-level public API (generate / generate_diff)

Everything below is re-exported so callers can simply ``import pro``.
"""
from . import config, data, physics, products, compute, render, engine  # noqa: F401
from . import addons  # noqa: F401

# --- public API / constants -------------------------------------------------
from .config import DEFAULT_N_DAYS, DEFAULT_PRODUCT  # noqa: F401
from .products import (  # noqa: F401
    PRODUCTS, GROUP_ORDER, list_products, group_products,
)
from .engine import generate, generate_diff  # noqa: F401

__all__ = [
    "config", "data", "physics", "products", "compute", "render", "engine",
    "addons",
    "DEFAULT_N_DAYS", "DEFAULT_PRODUCT",
    "PRODUCTS", "GROUP_ORDER", "list_products", "group_products",
    "generate", "generate_diff",
]
