"""Storm definitions. Each entry needs a date and a nominal 850-hPa centre.

The nominal centre is only the starting point of the search: the centre used by
the diagnostics is the 850-hPa vorticity maximum found in the data near it.
"""

STORMS = {
    "amphan": {
        "name": "AMPHAN",
        "kind": "Super Cyclonic Storm",
        "date": "2020-05-18",
        "nominal_center": (13.7, 86.5),   # approximate, read from the reference figure
    },
    "fani": {
        "name": "FANI",
        "kind": "Extremely Severe Cyclonic Storm",
        "date": "2019-05-02",
        "nominal_center": (16.5, 84.5),   # approximate, read from the reference figure
    },
}
