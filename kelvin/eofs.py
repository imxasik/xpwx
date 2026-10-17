"""kelvin.eofs — the two digitized Kelvin VP200 EOF longitude patterns.

These are the exact Fourier coefficients from the user's standalone
``kelvin_phase`` script (constant, then cos(k*lon), sin(k*lon) for k = 1..4).
They were digitized from the supplied reference raster; the line-fit residual
is ~7e-4 in the reference's plotted units, so the projection is approximate.

Keep this file as the single place to replace the EOF pair: the physics and
the drawing read these arrays, nothing else hard-codes them.
"""
from __future__ import annotations

import numpy as np

# Fourier coefficients of the reference EOF longitude profiles.
EOF1_FOURIER = np.array(
    [0.000354367340, -0.00505585616, 0.00671341331,
     0.0565045996, 0.0698333258, 0.00727614851,
     0.00956273656, -0.00009571442, 0.00184338088],
    dtype=np.float64,
)

EOF2_FOURIER = np.array(
    [0.000829043607, -0.00624096, -0.00605724,
     -0.07015052, 0.05561797, -0.01024334,
     0.00702774, -0.00217647, 0.00011147],
    dtype=np.float64,
)


def eval_eof(lon_deg, coeffs=None):
    """Evaluate an EOF longitude profile at ``lon_deg`` (degrees).

    ``coeffs`` is the 9-element Fourier vector above; ``None`` selects EOF1.
    The result is periodic in 360 degrees, so any longitude convention works.
    """
    c = EOF1_FOURIER if coeffs is None else np.asarray(coeffs, dtype=np.float64)
    if c.size != 9:
        raise ValueError("EOF coefficient vectors must have 9 elements")
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    out = np.full(lon.shape, c[0], dtype=np.float64)
    for k in range(1, 5):
        out += c[2 * k - 1] * np.cos(k * lon) + c[2 * k] * np.sin(k * lon)
    return out
