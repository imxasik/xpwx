"""Deploy-time warm-up: build the Relative AAM climatology cache.

Run once from the project root, normally as part of the host's build command:

    python custom/_warm_cache.py

Why this exists
---------------
``rel_aam`` needs the latitude-resolved daily climatology of 1991–2020
(366 days x 17 levels x lat x lon).  Building it inside a request costs a few
hundred MB of RSS for a few seconds — fine on a laptop, too much for a 512 MB
free-tier instance, where it used to end in an OOM kill (Render 502/503).

So it is built once, here, and stored in ``ltm_rel_aam_cache.npy`` in the project
root.  Container hosts keep that file for the life of the deployed image and
rebuild it on the next deploy, so no request ever pays for it.  The same warm-up
also downloads the current year's ``uwnd`` file, which removes the first
request's download as a bonus.

Safe to run anywhere: it is idempotent, exits non-zero on failure (callers use
``|| true`` so a deploy is never blocked by it), and it never touches the
committed ``ltm_aam_cache.npy`` of the GWO product.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main():
    import numpy as np

    from custom import gwo as _gwo
    from custom import rel_aam as RA

    print("[warm_cache] reading the live levels …", flush=True)
    _times, lev, _lat, _lon, _uwnd = _gwo.fetch_recent_days(1)
    levels = sorted({int(round(x)) for x in np.asarray(lev, float)}, reverse=True)
    print(f"[warm_cache] {len(levels)} live levels: {levels}", flush=True)

    clim, lat_axis, lev_common = RA.compute_ltm_climatology(lev)
    size = os.path.getsize(RA.LTM_CACHE) / 1e3 if os.path.exists(RA.LTM_CACHE) else 0.0
    print(f"[warm_cache] climatology {clim.shape} × {len(lev_common)} levels on "
          f"{len(lat_axis)} latitudes", flush=True)
    print(f"[warm_cache] cache: {RA.LTM_CACHE} ({size:.0f} kB)", flush=True)
    if clim.shape[0] < 300 or not np.isfinite(clim).all():
        print("[warm_cache] climatology looks wrong — discarding it so the next "
              "request rebuilds from scratch", flush=True)
        try:
            os.remove(RA.LTM_CACHE)
        except OSError:
            pass
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
