"""pro.data — NOAA PSL file access, field means, coastline, date resolution.

Data are plain netCDF files on https://downloads.psl.noaa.gov.  Each file is
downloaded once into ``config.CACHE_DIR`` and then read locally with
``netCDF4`` (no raw values are rescaled here — ``_read_slice`` applies the
packing attributes, exactly as before).

``open_remote(url)`` is the single entry point.  It returns an ``NCFile`` that
offers the small slice of the old pydap API the rest of the code uses:

    ds["uwnd"][t, lv, :, :].data      -> numpy array
    ds["uwnd"].array[t0:t1, ...]      -> lazily-read numpy-like slice
    ds["uwnd"].attributes             -> dict of netCDF attributes
    np.array(ds["lat"][:])            -> full coordinate array
    "level" in ds                     -> variable exists
"""
import io, os, re, zipfile, datetime, threading, urllib.parse
import numpy as np
import shapefile, requests
import netCDF4
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

# ── local NetCDF access ──────────────────────────────────────────────────────
# netCDF4/HDF5 is not thread-safe; every read goes through one lock.
_NC_LOCK = threading.RLock()
# Serialises downloads so two threads never fetch the same file twice.
_DL_LOCK = threading.Lock()


class _Arr:
    """Result of an indexed read.  Wraps a numpy array, pydap-style."""

    __slots__ = ("_a",)

    def __init__(self, a):
        self._a = np.ascontiguousarray(a)

    @property
    def data(self):
        return self._a

    @property
    def shape(self):
        return self._a.shape

    @property
    def ndim(self):
        return self._a.ndim

    def __len__(self):
        return len(self._a)

    def __getitem__(self, idx):
        return self._a[idx]

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self._a, dtype=dtype)


class _Indexer:
    """Lazy ``var.array[...]`` accessor (reads only the requested block)."""

    __slots__ = ("_var",)

    def __init__(self, var):
        self._var = var

    def __getitem__(self, idx):
        return self._var[idx]


class NCVar:
    def __init__(self, nc_var):
        self._v = nc_var

    def __getitem__(self, idx):
        with _NC_LOCK:
            return _Arr(np.asarray(self._v[idx]))

    def __len__(self):
        return self._v.shape[0]

    @property
    def data(self):
        """Whole variable as a numpy array."""
        with _NC_LOCK:
            return np.asarray(self._v[:])

    @property
    def array(self):
        return _Indexer(self)

    @property
    def attributes(self):
        return {k: self._v.getncattr(k) for k in self._v.ncattrs()}

    @property
    def shape(self):
        return self._v.shape

    @property
    def dimensions(self):
        return self._v.dimensions


class NCFile:
    """A locally cached netCDF file (opened read-only, raw packed values)."""

    def __init__(self, path):
        self.path = path
        self._nc = netCDF4.Dataset(path, "r")
        # Keep raw values: scale/offset/fill are applied by the callers, as before.
        self._nc.set_auto_mask(False)
        self._nc.set_auto_scale(False)

    def __getitem__(self, name):
        return NCVar(self._nc.variables[name])

    def __contains__(self, name):
        return name in self._nc.variables

    def get(self, name, default=None):
        return self[name] if name in self._nc.variables else default

    def keys(self):
        return list(self._nc.variables.keys())


def release_memory():
    """Hand freed temporaries back to the OS.

    numpy frees the big read/integration arrays as soon as they go out of scope,
    but the C allocator keeps the pages, so RSS stays at the request's peak for
    the life of the process.  On a 512 MB host (Render free) that is the
    difference between a service that survives a session of map requests and one
    that is OOM-killed.  Best-effort no-op where glibc's malloc_trim is missing.
    """
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:                                           # noqa: BLE001
        pass


def _local_path(url):
    """Map a URL to a stable path under CACHE_DIR, e.g. .../Datasets/ncep/uwnd.2026.nc."""
    rel = urllib.parse.urlparse(url).path.lstrip("/")
    return os.path.join(config.CACHE_DIR, rel)


def _needs_download(local, url):
    if not os.path.exists(local) or os.path.getsize(local) == 0:
        return True
    # Files for the current year grow daily on NOAA's side; refresh them.
    year = str(datetime.date.today().year)
    if f".{year}." in os.path.basename(url):
        age = datetime.datetime.now().timestamp() - os.path.getmtime(local)
        return age > config.CACHE_TTL_HOURS * 3600
    return False


def _ensure_local(url):
    """Return a local path holding ``url``'s contents, downloading if needed."""
    local = _local_path(url)
    with _DL_LOCK:
        if not _needs_download(local, url):
            return local
        os.makedirs(os.path.dirname(local), exist_ok=True)
        tmp = local + ".part"
        try:
            with requests.get(url, stream=True, timeout=(20, 900)) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            f.write(chunk)
            os.replace(tmp, local)
            print(f"[data] downloaded {url} ({os.path.getsize(local) / 1e6:.1f} MB)",
                  flush=True)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            if os.path.exists(local):
                # A stale copy is better than failing the whole request.
                print(f"[data] refresh failed for {url}; using cached copy", flush=True)
                return local
            raise
    return local


def open_remote(url):
    """Download (once, cached) and open a PSL netCDF file.  Returns an NCFile."""
    local = _ensure_local(url)
    mtime = os.path.getmtime(local)
    with _NC_LOCK:
        entry = _DS_CACHE.get(url)
        if entry is not None and entry[0] == mtime:
            return entry[1]
        try:
            ds = NCFile(local)
        except OSError as e:
            # Not a netCDF file (e.g. an HTML error page saved with HTTP 200).
            try:
                os.remove(local)
            except OSError:
                pass
            raise RuntimeError(f"{url} is not a readable netCDF file: {e}") from e
        # The previous handle (if any) is intentionally left open: a caller may
        # still be reading from it.  Refreshes are rare (once per CACHE_TTL_HOURS).
        _DS_CACHE[url] = (mtime, ds)
        return ds


_PVAR = {"uwnd.sfc": "uwnd", "vwnd.sfc": "vwnd",
         "air.sfc": "air", "rhum.sfc": "rhum"}

def _pvar(key):
    return _PVAR.get(key, key)

def _open(varname, year=None):
    if year is None:
        url = f"{PSL}/{varname}.day.ltm.1991-2020.nc"
    else:
        url = f"{PSL}/{varname}.{year}.nc"
    return open_remote(url)

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
            ds_temp = open_remote(f"{PSL}/uwnd.{current_year}.nc")
        except Exception:
            current_year -= 1
            ds_temp = open_remote(f"{PSL}/uwnd.{current_year}.nc")
        raw_times = np.array(ds_temp["time"][:])
        units = ds_temp["time"].attributes.get("units", "hours since 1800-01-01")
        scale = 1.0 / 24.0 if "hours" in units else 1.0
        m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", units)
        epoch = (datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                 if m else datetime.date(1800, 1, 1))
        date_end = epoch + datetime.timedelta(days=float(raw_times[-1]) * scale)

    date_start = date_end - datetime.timedelta(days=n_days - 1)
    return [date_start + datetime.timedelta(days=i) for i in range(n_days)]
