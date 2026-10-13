"""rmm/rmm8.py — RMM1 / RMM2 daily series from the JMA MJO CSV (chi200 + u850 + u200).

Source (JMA, Tokyo Climate Center):
    https://ds.data.jma.go.jp/tcc/tcc/products/clisys/mjo/figs/chi200-sst1_1991-2021/rmm8.csv

The CSV is downloaded once per CACHE_TTL_HOURS (a stale copy is used if the
refresh fails). The header is located by its RMM1 column, and the date columns
are read from whichever layout the file uses (Year/Month/Day, a date column, or
YYYYMMDD). Rows without a valid RMM1 and RMM2 are skipped.

Override the source with the environment variable RMM8_URL (used for testing).
"""
import csv
import datetime
import io
import os
import re
import time

import numpy as np
import requests

from pro import config

URL = os.environ.get(
    "RMM8_URL",
    "https://ds.data.jma.go.jp/tcc/tcc/products/clisys/mjo/figs/chi200-sst1_1991-2021/rmm8.csv",
)
CACHE_FILE = os.path.join(config.CACHE_DIR, "jma", "rmm8.csv")
_RMM1 = re.compile(r"^\s*rmm\s*_?\s*1\b", re.IGNORECASE)
_RMM2 = re.compile(r"^\s*rmm\s*_?\s*2\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Parsing (pure)
# ---------------------------------------------------------------------------

def _find(names, pattern):
    for i, n in enumerate(names):
        if pattern.match(n):
            return i
    return None


def _to_date(cells, cols):
    """Date from the date columns named by ``cols`` (dict: kind -> index)."""
    if "date" in cols:
        s = cells[cols["date"]].strip()
        if re.fullmatch(r"\d{8}", s):
            return datetime.date(int(s[:4]), int(s[4:6]), int(s[6:8]))
        m = re.match(r"(\d{4})\D(\d{1,2})\D(\d{1,2})", s)
        if m:
            return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        raise ValueError(f"unrecognised date {s!r}")
    y = int(float(cells[cols["year"]]))
    mo = int(float(cells[cols["month"]]))
    d = int(float(cells[cols["day"]]))
    return datetime.date(y, mo, d)


def parse_csv(text):
    """Return (dates, rmm1, rmm2) as sorted numpy arrays from the CSV text."""
    rows = list(csv.reader(io.StringIO(text)))
    head_i = None
    for i, row in enumerate(rows):
        names = [c.strip().lstrip('#').strip() for c in row]
        if _find(names, _RMM1) is not None and _find(names, _RMM2) is not None:
            head_i = i
            break
    if head_i is None:
        sample = " | ".join(",".join(r) for r in rows[:3])
        raise ValueError(f"RMM1/RMM2 header not found in the JMA CSV. First rows: {sample}")

    # header names may carry a comment marker, e.g. '#year'
    names = [c.strip().lstrip("#").strip() for c in rows[head_i]]
    lower = [n.lower() for n in names]
    c1 = _find(names, _RMM1)
    c2 = _find(names, _RMM2)
    cols = {}
    if "date" in lower:
        cols["date"] = lower.index("date")
    elif "yyyymmdd" in lower:
        cols["date"] = lower.index("yyyymmdd")
    else:
        for key, cands in (("year", ("year", "yr", "yyyy")),
                           ("month", ("month", "mon", "mo", "mm")),
                           ("day", ("day", "dy", "dd"))):
            hit = [i for i, n in enumerate(lower) if n in cands]
            if not hit:
                raise ValueError(f"date column '{key}' not found; header: {names}")
            cols[key] = hit[0]

    dates, r1, r2 = [], [], []
    for row in rows[head_i + 1:]:
        if len(row) <= max(c1, c2, *cols.values()):
            continue
        try:
            a, b = float(row[c1]), float(row[c2])
            d = _to_date(row, cols)
        except ValueError:
            continue
        # missing: non-finite, or a sentinel such as -999 (|value| >= 999)
        if not (np.isfinite(a) and np.isfinite(b)) or abs(a) >= 999 or abs(b) >= 999:
            continue
        dates.append(d)
        r1.append(a)
        r2.append(b)

    if not dates:
        raise ValueError("no valid RMM rows in the JMA CSV")
    order = np.argsort(np.array(dates, dtype=object))
    dates = np.array(dates, dtype=object)[order]
    r1 = np.array(r1)[order]
    r2 = np.array(r2)[order]
    # keep the last value if a date is repeated
    keep = np.r_[dates[1:] != dates[:-1], True]
    return dates[keep], r1[keep], r2[keep]


# ---------------------------------------------------------------------------
# Download (cached) and public helpers
# ---------------------------------------------------------------------------

def _read_cache():
    with open(CACHE_FILE, "rb") as f:
        return f.read()


def _fetch_bytes():
    fresh = (os.path.exists(CACHE_FILE)
             and time.time() - os.path.getmtime(CACHE_FILE) < config.CACHE_TTL_HOURS * 3600)
    if fresh:
        return _read_cache()
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    try:
        r = requests.get(URL, timeout=(20, 120))
        r.raise_for_status()
        tmp = CACHE_FILE + ".part"
        with open(tmp, "wb") as f:
            f.write(r.content)
        os.replace(tmp, CACHE_FILE)
        return r.content
    except Exception:
        if os.path.exists(CACHE_FILE):
            print("[rmm8] refresh failed; using cached JMA CSV", flush=True)
            return _read_cache()
        raise


def _decode(raw):
    for enc in ("utf-8-sig", "cp932"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("JMA CSV is neither UTF-8 nor CP932")


def load():
    """(dates, rmm1, rmm2) for the whole JMA record."""
    return parse_csv(_decode(_fetch_bytes()))


def window(end_date=None, n_days=40):
    """The last ``n_days`` days ending on ``end_date`` (or the latest day in the file).

    If ``end_date`` is later than the file, the window ends on the last day in the file.
    Returns (dates, rmm1, rmm2) with dates as datetime.date objects.
    """
    dates, r1, r2 = load()
    if end_date is not None:
        m = dates <= end_date
        dates, r1, r2 = dates[m], r1[m], r2[m]
    if len(dates) < n_days:
        raise ValueError(f"only {len(dates)} days available up to {end_date}")
    dates, r1, r2 = dates[-n_days:], r1[-n_days:], r2[-n_days:]
    gaps = [d for d in np.diff(dates) if d != datetime.timedelta(days=1)]
    if gaps:
        raise ValueError("the JMA record has missing days inside the window")
    return dates, r1, r2
