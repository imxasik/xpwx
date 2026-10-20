"""cyclone — tropical-cyclone diagnostics from NCEP/NCAR daily fields.

    core.py          numerical diagnostics (no I/O); omega from mass continuity
    fields.py        reads one day of u, v, T, z (and omega, diagnosed) via pro.data (cached)
    analysis.py      runs the diagnostics for a user-given lat/lon and date
    borders.py       country-border lines for the maps (reads map/ne_50m_admin_0_borders.shp)
    build_borders.py rebuilds the border file from map/ne_50m_admin_0_countries.shp
    plot.py          the two figures (Dynamic Fig, ThermoDynamic Fig)
"""
