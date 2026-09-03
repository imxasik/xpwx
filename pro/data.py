"""pro.data — OPeNDAP access, field means, coastline, date resolution."""
import io, os, re, zipfile, datetime
import numpy as np
import shapefile, requests
from pydap.client import open_url
from . import config

PSL = config.PSL
BASE_DIR = config.BASE_DIR
SHP_DIR = config.SHP_DIR
SHP_PATH = config.SHP_PATH
SHP_URL = config.SHP_URL
_DS_CACHE = config._DS_CACHE
_FIELD_CACHE = config._FIELD_CACHE
_LATLON_CACHE = config._LATLON_CACHE
_COAST = config._COAST

_PVAR = {"uwnd.sfc": "uwnd", "vwnd.sfc": "vwnd",
         "air.sfc": "air", "rhum.sfc": "rhum"}

def _pvar(key):
    return _PVAR.get(key, key)

def _open(varname, year=None):
    if year is None:
        url = f"{PSL}/{varname}.day.ltm.1991-2020.nc"
    else:
        url = f"{PSL}/{varname}.{year}.nc"
    if url not in _DS_CACHE:
        _DS_CACHE[url] = open_url(url)
    return _DS_CACHE[url]

def _latlon(var):
    if var not in _LATLON_CACHE:
        ds = _open(var, year=2024)
        _LATLON_CACHE[var] = (np.array(ds["lat"][:]), np.array(ds["lon"][:]))
    return _LATLON_CACHE[var]

def _level_idx(ds, hPa):
    lev = np.array(ds["level"][:])
    return int(np.argmin(np.abs(lev - hPa)))

def _epoch(ds):
    units = ds["time"].attributes.get("units", "hours since 1800-01-01")
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", units)
    return (datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            if m else datetime.date(1800, 1, 1))

def _time_idx(ds, target):
    raw = np.array(ds["time"][:])
    units = ds["time"].attributes.get("units", "hours since 1800-01-01")
    scale = 1.0 / 24.0 if "hours" in units else 1.0
    epoch = _epoch(ds)
    for i, t in enumerate(raw):
        d = epoch + datetime.timedelta(days=float(t) * scale)
        if d.year == target.year and d.month == target.month and d.day == target.day:
            return i
    raise ValueError(f"Date {target} not found in dataset")

def _read_slice(ds, varname, t, lv):
    """Read one time slice. If the dataset has a 'level' dim use it, else surface."""
    var = _pvar(varname)
    if "level" in ds:
        raw = np.array(ds[var][t, lv, :, :].data).squeeze().astype(np.float64)
    else:
        raw = np.array(ds[var][t, :, :].data).squeeze().astype(np.float64)
    attr = ds[var].attributes
    sf = float(attr.get("scale_factor", 1.0))
    ao = float(attr.get("add_offset", 0.0))
    mv = float(attr.get("missing_value", 32767.0))
    fill_mask = np.abs(raw - mv) < 0.5
    data = raw * sf + ao
    data[fill_mask] = np.nan
    return data

def ensure_coastline():
    if os.path.exists(SHP_PATH):
        return
    os.makedirs(SHP_DIR, exist_ok=True)
    r = requests.get(SHP_URL, timeout=60)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    for name in z.namelist():
        if os.path.splitext(name)[1] in (".shp", ".shx", ".dbf", ".prj"):
            with open(os.path.join(SHP_DIR, os.path.basename(name)), "wb") as f:
                f.write(z.read(name))

def load_coastlines():
    if config._COAST is None:
        ensure_coastline()
        sf = shapefile.Reader(SHP_PATH)
        segs = []
        for shape in sf.shapes():
            pts = shape.points
            parts = list(shape.parts) + [len(pts)]
            for i in range(len(shape.parts)):
                segs.append(np.array(pts[parts[i]:parts[i + 1]]))
        config._COAST = segs
    return config._COAST

def _mean_field(var, level, dates, kind):
    key = (var, level, tuple(d.isoformat() for d in dates), kind)
    if key in _FIELD_CACHE:
        return _FIELD_CACHE[key]

    if kind == "obs":
        by_year = {}
        for d in dates:
            by_year.setdefault(d.year, []).append(d)
        slices = []
        for year, ydates in sorted(by_year.items()):
            ds = _open(var, year)
            lv = _level_idx(ds, level) if level is not None else 0
            for d in ydates:
                ti = _time_idx(ds, d)
                slices.append(_read_slice(ds, var, ti, lv))
    else:
        ds = _open(var)
        lv = _level_idx(ds, level) if level is not None else 0
        n = len(np.array(ds["time"][:]))
        slices = []
        for d in dates:
            doy = d.timetuple().tm_yday
            ti = min(doy - 1, n - 1)
            slices.append(_read_slice(ds, var, ti, lv))

    mean = np.nanmean(slices, axis=0)
    _FIELD_CACHE[key] = mean
    return mean

def _mean_multi(var, levels, dates, kind):
    """Period-mean field over several pressure levels -> (nlevels, nlat, nlon)."""
    key = (var, "multi", tuple(levels), tuple(d.isoformat() for d in dates), kind)
    if key in _FIELD_CACHE:
        return _FIELD_CACHE[key]

    if kind == "obs":
        by_year = {}
        for d in dates:
            by_year.setdefault(d.year, []).append(d)
        stack = []
        for year, ydates in sorted(by_year.items()):
            ds = _open(var, year)
            for lv in range(len(levels)):
                sl = []
                for d in ydates:
                    ti = _time_idx(ds, d)
                    sl.append(_read_slice(ds, var, ti, lv))
                stack.append(np.nanmean(sl, axis=0))
    else:
        ds = _open(var)
        n = len(np.array(ds["time"][:]))
        stack = []
        for lv in range(len(levels)):
            sl = []
            for d in dates:
                doy = d.timetuple().tm_yday
                ti = min(doy - 1, n - 1)
                sl.append(_read_slice(ds, var, ti, lv))
            stack.append(np.nanmean(sl, axis=0))

    arr = np.stack(stack, axis=0)          # (nlevels, nlat, nlon)
    _FIELD_CACHE[key] = arr
    return arr

def _daily_stack(var, level, dates, kind):
    """Daily per-day field (ntime, nlat, nlon) for obs or climatology."""
    key = (var, level, tuple(d.isoformat() for d in dates), "daily", kind)
    if key in _FIELD_CACHE:
        return _FIELD_CACHE[key]
    if kind == "obs":
        by_year = {}
        for d in dates:
            by_year.setdefault(d.year, []).append(d)
        rows = []
        for year, ydates in sorted(by_year.items()):
            ds = _open(var, year)
            lv = _level_idx(ds, level) if level is not None else 0
            for d in ydates:
                ti = _time_idx(ds, d)
                rows.append(_read_slice(ds, var, ti, lv))
    else:
        ds = _open(var)
        lv = _level_idx(ds, level) if level is not None else 0
        n = len(np.array(ds["time"][:]))
        rows = []
        for d in dates:
            ti = min(d.timetuple().tm_yday - 1, n - 1)
            rows.append(_read_slice(ds, var, ti, lv))
    arr = np.stack(rows, axis=0)
    _FIELD_CACHE[key] = arr
    return arr

def _resolve_dates(mode, manual_date, n_days):
    n_days = max(1, int(n_days))
    if mode == "manual" and manual_date:
        date_end = datetime.date.fromisoformat(manual_date)
    else:
        current_year = datetime.date.today().year
        try:
            ds_temp = open_url(f"{PSL}/uwnd.{current_year}.nc")
        except Exception:
            current_year -= 1
            ds_temp = open_url(f"{PSL}/uwnd.{current_year}.nc")
        raw_times = np.array(ds_temp["time"][:])
        units = ds_temp["time"].attributes.get("units", "hours since 1800-01-01")
        scale = 1.0 / 24.0 if "hours" in units else 1.0
        m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", units)
        epoch = (datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                 if m else datetime.date(1800, 1, 1))
        date_end = epoch + datetime.timedelta(days=float(raw_times[-1]) * scale)

    date_start = date_end - datetime.timedelta(days=n_days - 1)
    return [date_start + datetime.timedelta(days=i) for i in range(n_days)]
