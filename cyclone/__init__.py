"""cyclone — tropical-cyclone diagnostics from NCEP/NCAR daily fields.

    core.py      numerical diagnostics (no I/O)
    fields.py    reads one day of u, v, T, z, omega via pro.data (cached)
    storms.py    storm definitions (name, date, nominal centre)
    analysis.py  runs the diagnostics for one storm
    plot.py      the two figures
"""
