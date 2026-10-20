"""kelvin.pipeline — run the forecast through fetch -> filter -> EOF phase."""
from __future__ import annotations

import datetime as dt

import numpy as np

from . import fetch, physics


def build_forecast(mode="auto", date=None):
    """Return one phase-space forecast ready for the plot.

    ``mode`` is ``"auto"`` (latest complete run) or ``"manual"`` with a
    ``datetime.date`` selecting the newest run on or before that date.
    """
    run, steps, step_hours = fetch.select_run(mode=mode, date=date)
    coeff, nlon = fetch.fetch_coefficients(run, steps, step_hours)
    filtered = physics.filter_eastward_kelvin(coeff, step_hours)
    x, y, phase_deg, amp, scales = physics.phase_coordinates(filtered, nlon)
    return {
        "run": run,
        "run_label": fetch.run_label(run),
        "run_datetime": fetch.run_datetime(run),
        "steps": np.asarray(steps, dtype=int),
        "step_hours": int(step_hours),
        "coeff": coeff,
        "filtered": filtered,
        "nlon": int(nlon),
        "x": x,
        "y": y,
        "phase_deg": phase_deg,
        "amplitude": amp,
        "scales": scales,
        "valid_start": fetch.run_datetime(run) + dt.timedelta(hours=int(steps[0])),
        "valid_end": fetch.run_datetime(run) + dt.timedelta(hours=int(steps[-1])),
        "n_steps": len(steps),
    }
