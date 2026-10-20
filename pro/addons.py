"""pro.addons — drop-in product plugins.

The point of this module is to let you add a NEW product WITHOUT editing any
core engine file (products.py, compute.py, render.py, ...), and without
re-uploading the whole project on Deploy.

How it works
-------------
1. Create a folder named ``custom`` next to this package (in the project root,
   i.e. same place as app.py). It already exists.
2. Drop a file in it, e.g. ``custom/my_product.py``. It is loaded automatically
   on engine start. Any ``*.py`` is loaded; files starting with ``_`` are
   skipped (so you can keep a reference/template file that isn't active).
3. A file may define TWO things (both optional):

       PRODUCTS = { ... }     # new products, dropping in like in pro/products.py
       KINDS    = { ... }     # a brand-new "kind" (new diagnostic), see below

4. Restart the server. The new product appears in the sidebar /products list
   automatically.

Two ways to add a product
---------------------------
TIER 1 — reuse an existing kind. Just give it a config. No code. The most
common case (an anomaly of any single variable, a χ/vtp, ψ/psi, IVT, Eady, ...).
Look at pro/products.py to see every available "kind" and its required keys.

TIER 2 — a genuinely new diagnostic. Define a custom "kind": a compute function
(and optionally a render function). ``compute(pkg, dates)`` must return
``(lat, lon, data)`` where ``data`` is a dict with at least ``"main"`` (and
optionally ``"u"/"v"`` wind or ``"vec_u"/"vec_v"`` flux, ``"psi"``, ``"ks"``).

Example custom file is stored at:  custom/_EXAMPLE.py  (underscore = not loaded,
it is a reference). Copy it, rename without the underscore, edit, deploy.
"""
import os
import importlib.util
import logging

from . import config

log = logging.getLogger("pro.addons")

_CACHE = None  # (merged_products, custom_kinds)


def addon_dir():
    return os.path.join(config.BASE_DIR, "custom")


def _load_file(path):
    name = "pro_addon_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_addons():
    """Scan ./custom/*.py, return (merged_products, custom_kinds). Cached."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    products, kinds = {}, {}
    d = addon_dir()
    if os.path.isdir(d):
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".py") or fn.startswith("_"):
                continue
            path = os.path.join(d, fn)
            try:
                mod = _load_file(path)
            except Exception as exc:  # noqa: BLE001
                log.warning("addon %s failed to load: %s", fn, exc)
                continue
            products.update(getattr(mod, "PRODUCTS", {}) or {})
            kinds.update(getattr(mod, "KINDS", {}) or {})
            log.info("loaded addon %s (%d products, %d kinds)",
                     fn, len(getattr(mod, "PRODUCTS", {}) or {}),
                     len(getattr(mod, "KINDS", {}) or {}))

    _CACHE = (products, kinds)
    return _CACHE


def custom_products():
    """Merged dict of product configs contributed by custom/ files."""
    return load_addons()[0]


def custom_kinds():
    """Dict kind -> {compute: fn, render?: fn, tag?: str, title?: str}."""
    return load_addons()[1]
