"""kelvin — ECMWF IFS Kelvin VP200 phase-space forecast.

Public entry points used by the sidebar addon:

    from kelvin import pipeline, plot
    forecast = pipeline.build_forecast(mode, date)
    png_bytes, n = plot.render_png_locked(forecast)
"""
from . import eofs, fetch, physics, pipeline, plot  # noqa: F401

__all__ = ["eofs", "fetch", "physics", "pipeline", "plot"]
