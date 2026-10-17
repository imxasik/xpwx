"""
engine.py — ECMWF Forecast Map Maker (web সংস্করণ)
====================================================
ব্যবহারকারীর দেওয়া standalone স্ক্রিপ্ট থেকে হুবহু নেওয়া ইঞ্জিন:
ECMWF IFS open-data (data.ecmwf.int) থেকে GRIB2 রেঞ্জ-রিকোয়েস্টে
শুধু দরকারি ফিল্ড নামায়, pure-Python GRIB2/CCSDS (libaec) ডিকোডারে
ডিকোড করে, আর একই ম্যাপ-ডিজাইনে (হেডার, কালারবার, শহর, ওয়াটারমার্ক)
রেন্ডার করে।

ওয়েবের জন্য বদল (ডিজাইন/লজিক একই রাখা হয়েছে):
  * plot_single() এখন ফাইলে সেভ না করে PNG bytes রিটার্ন করে।
  * GIF / overview প্যানেল আপাতত নেই (ওয়েবে লাগবে না)।
  * একবারে একটা ম্যাপ (একটা ফোরকাস্ট আওয়ার) রেন্ডার হয় —
    রান (auto = সর্বশেষ / manual = তারিখ+ঘণ্টা), রিজিয়ন আর
    ফোরকাস্ট আওয়ার ওয়েব UI থেকে আসে।
  * REGION_PRESETS-এ 'bangladesh' প্রিসেট হুবহু আছে; ওয়েবের জন্য
    আরও কয়েকটা রিজিয়ন যোগ করা হয়েছে (bbox শুধু)।
"""

# ======================== CONFIG — এখানে নিজের মতো বদলান ========================
import os

def _envlist(name, default):
    v = os.environ.get(name)
    if not v:
        return default
    return [x.strip() for x in v.split(',') if x.strip()]

REGION      = os.environ.get('ECMWF_REGION',  'bangladesh')  # ম্যাপের এলাকা (নিচে REGION_PRESETS দেখুন)
STEP_H      = int(os.environ.get('ECMWF_STEP_H', '3'))      # স্টেপ ব্যবধান (ঘণ্টা): 3 / 6 / 12 / 24
THEME       = os.environ.get('ECMWF_THEME', 'light')       # 'light' অথবা 'dark'
DPI           = 170                                          # ছবির রেজোলিউশন
CITIES_ON     = True                     # প্রেসেট এলাকায় শহরের নাম দেখাবে (যেখানে আছে)
STATES_ON     = True                     # বিভাগ/স্টেট বর্ডার দেখাবে: True/False অথবা 1/0 লিখুন
STATES_ON     = str(os.environ.get('ECMWF_STATES', STATES_ON)).strip().lower() not in ('0', 'false', 'no', 'off', '')
_HERE         = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR     = os.environ.get('ECMWF_CACHE_DIR', os.path.join(_HERE, 'ecmwf_cache'))

# ---- ECMWF সার্ভারের প্রতি শালীনতা / লিমিট এড়ানো (দয়া করে ছোট করবেন না) ----
REQUEST_DELAY = 0.35      # প্রতিটি HTTP রিকোয়েস্টের মধ্যে বিরতি (সেকেন্ড) — সব থ্রেড মিলেও এই ব্যবধান বজায় থাকে
MAX_RETRIES   = 4         # ব্যর্থ হলে কতবার পুনঃচেষ্টা (exponential backoff সহ)
POOL_CACHE    = True      # নামানো ডেটা ক্যাশে রাখবে — পুনরানে আবার ডাউনলোড হয় না
# ===============================================================================

import io
import json
import struct
import sys
import time
import math
import datetime as dt

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')          # সার্ভারে ফাইলে/মেমরিতে সেভ করার জন্য নিরাপদ ব্যাকএন্ড
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.collections import LineCollection
    from matplotlib import patheffects
except ImportError as e:
    print('ত্রুটি:', e, '\nPip থেকে numpy, matplotlib ইন্সটল করুন।')
    raise

try:
    import urllib.request
    import urllib.error
except ImportError:
    pass

BASE_URL = 'https://data.ecmwf.int/forecasts'
UA = {'User-Agent': 'ecmwf-map-maker/1.0 (xpwx web; contact: local-user)'}

# =============================== ম্যাপ রেজিস্ট্রি ===============================
# প্রতিটি এন্ট্রি: কোন GRIB প্যারামিটার লাগবে, কীভাবে রূপান্তর/প্লট হবে
MAP_HELP = """MAP কী-গুলো: t2m (2m তাপমাত্রা °C) | msl (আপ. চাপ+কনট্যুর) | msl_wind (চাপ+10m বাতাস)
tp (সঞ্চিত বৃষ্টি mm) | tp24 (গত 24ঘণ্টার বৃষ্টি) | wind10 (10m বাতাসের গতি+বার্ব)
gh500 (500hPa উচ্চতা gpm) | t850 (850hPa তাপমাত্রা °C) | tcc (মেঘাচ্ছাদন %) | cape (J/kg)"""

# ওয়েব সাইডবারে দেখানোর ক্রম (ডিফল্ট 'combo' — মূল স্ক্রিপ্টের MAPS ডিফল্ট)
MAP_ORDER = ['combo', 't2m', 'msl', 'msl_wind', 'tp', 'tp24',
             'wind10', 'gh500', 't850', 'tcc', 'cape',
             'steer', 'satdef', 'secondary', 'outflow']

# কোর-কেন্দ্রিক (cyclone) ম্যাপ — ব্যবহারকারীর দেওয়া সেন্টার lat/lon লাগে
CENTRE_MAPS = {'secondary', 'outflow'}

MAP_DESCRIPTIONS = {
    'combo':    'Rain (last 3 h) shaded + MSLP contours + 10 m wind arrows.',
    't2m':      '2 m temperature forecast (°C).',
    'msl':      'Mean sea level pressure with contours (hPa).',
    'msl_wind': 'MSL pressure + 10 m wind arrows.',
    'tp':       'Total accumulated precipitation since run start (mm).',
    'tp24':     'Precipitation accumulated over the last 24 h (mm).',
    'wind10':   '10 m wind speed with direction arrows (m/s).',
    'gh500':    '500 hPa geopotential height (gpm).',
    't850':     '850 hPa temperature (°C).',
    'tcc':      'Total cloud cover (%).',
    'cape':     'Most-unstable CAPE (J/kg).',
    'steer':    '850–200 hPa deep-layer mean wind speed (shaded) + streamlines.',
    'satdef':   '700–500 hPa layer-mean saturation deficit (g/kg) — dry-air intrusion.',
    'secondary': 'Azimuthal-mean secondary circulation: radius–height overturning around a centre you enter.',
    'outflow':  '200 hPa divergence + outflow streamlines and radial-outflow profile around a centre you enter.',
}

# ============================== রিজিয়ন প্রিসেট ==============================
REGION_PRESETS = {
    'bangladesh':  dict(bbox=(19.0, 28.0, 86.0, 94.2), tick=1,
                        cities={'Dhaka': (23.8103, 90.4125),
                                'Chattogram': (22.3569, 91.7832),
                                'Khulna': (22.8456, 89.5403),
                                'Rajshahi': (24.3745, 88.6042),
                                'Barishal': (22.7010, 90.3535),
                                'Sylhet': (24.8949, 91.8687),
                                'Rangpur': (25.7483, 89.2510),
                                'Mymensingh': (24.7471, 90.4203)}),
    # — ওয়েবের জন্য অতিরিক্ত রিজিয়ন (শুধু bbox; শহর/টিক বাংলাদেশ প্রিসেটেই আছে) —
    'south_asia':  dict(bbox=(5.0, 40.0, 65.0, 100.0), tick=5),
    'bay_of_bengal': dict(bbox=(5.0, 25.0, 78.0, 100.0), tick=2),
    'indian_ocean': dict(bbox=(-10.0, 30.0, 40.0, 110.0), tick=5),
    'asia':        dict(bbox=(0.0, 55.0, 25.0, 145.0), tick=10),
    'global':      dict(bbox=(-85.0, 85.0, -180.0, 180.0), tick=30),
}

REGION_LABELS = {
    'bangladesh': 'Bangladesh',
    'south_asia': 'South Asia',
    'bay_of_bengal': 'Bay of Bengal',
    'indian_ocean': 'Indian Ocean (North)',
    'asia': 'Asia',
    'global': 'Global',
}

# ================================ থিম / স্টাইল ================================
THEMES = {
    'light': dict(fig='#f4f6f8', ax='#dcebf6', coast='#10161e',
                  border='#39434d', grid='#c3ccd4', text='#15212c', sub='#5a6b7a',
                  accent='#0b6bcb', badge_fg='#ffffff'),
    'dark':  dict(fig='#0e1626', ax='#122238', coast='#e8edf2', border='#5a6b7a',
                  grid='#3d5165', text='#eef3f8', sub='#9fb2c2', accent='#2f9df0',
                  badge_fg='#0e1626'),
}

def _cmap(colors, name='custom'):
    return mcolors.LinearSegmentedColormap.from_list(name, colors)


def _get_cmap(colors, kind):
    if isinstance(colors, str):
        return plt.get_cmap(colors)
    if isinstance(colors, mcolors.Colormap):
        return colors
    return _cmap(colors, kind)

# — কালারম্যাপ প্যালেট (প্রফেশনাল ওয়েদার-গ্রেড) —
TEMP_COLORS = ['#313695', '#4169c1', '#3b90d9', '#56b6e3', '#8fd6ec', '#c4ecf0',
               '#e6f7e8', '#f9fcb8', '#fef088', '#fdd654', '#fdb33f', '#f58a26',
               '#e85b12', '#d43a09', '#b02202', '#7f1502', '#5a0d04']
PRECIP_COLORS = ['#f3f8fb', '#d3e9f5', '#a9dbf0', '#74c9e5', '#3fb6dd', '#2fa9c8',
                 '#31b98f', '#59b95e', '#9fc94a', '#f2d437', '#f5a52c', '#ef6a22',
                 '#dd2c1d', '#c1121f', '#d81b8f', '#7b1fa2']
WIND_COLORS = ['#ffffd9', '#edf8b1', '#c7e9b4', '#7fcdbb', '#41b6c4', '#1d91c0',
               '#225ea8', '#253494', '#081d58']
GH_CMAP = 'Spectral_r'
MSL_CMAP = 'RdYlBu_r'
TCC_CMAP = _cmap(['#fdfdfd', '#e5eef4', '#c3d6e2', '#97b9c9', '#6b93a9', '#406d84', '#1d3d52'])
CAPE_CMAP = _cmap(['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c',
                   '#cb181d', '#a50f15', '#67000d'])

LEVELS = {
    't2m':    (np.arange(-5, 45, 2).tolist(), TEMP_COLORS, 'both'),
    't850':   (np.arange(-8, 35, 2).tolist(), TEMP_COLORS, 'both'),
    'tp':     ([0.2, 1, 2, 5, 10, 20, 35, 55, 80, 110, 150, 200], PRECIP_COLORS, 'max'),
    'tp24':   ([0.2, 1, 2, 5, 10, 20, 35, 55, 80, 110, 150, 200], PRECIP_COLORS, 'max'),
    'combo':  ([0.1, 0.5, 1, 2, 4, 8, 12, 18, 28, 45, 70, 110], PRECIP_COLORS, 'max'),
    'wind10': ([0, 2, 4, 6, 8, 10, 12, 14, 17, 20, 24, 28, 33], WIND_COLORS, 'max'),
    'msl':    (np.arange(948, 1056, 4).tolist(), MSL_CMAP, 'neither'),
    'msl_wind': (np.arange(948, 1056, 4).tolist(), MSL_CMAP, 'neither'),
    'gh500':  (np.arange(4400, 6100, 60).tolist(), GH_CMAP, 'both'),
    'tcc':    (np.arange(0, 105, 10).tolist(), TCC_CMAP, 'neither'),
    'cape':   ([0, 100, 250, 500, 1000, 1500, 2000, 2500, 3000, 4000], CAPE_CMAP, 'max'),
    'steer':  ([0, 2, 4, 6, 8, 10, 12, 14, 17, 20, 24, 28, 33], WIND_COLORS, 'max'),
    'satdef': ([0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 10, 12], 'YlOrBr', 'max'),
}

MAP_TITLES = {
    't2m': ('2m Temperature', '°C'),
    't850': ('850 hPa Temperature', '°C'),
    'msl': ('Mean Sea Level Pressure', 'hPa'),
    'msl_wind': ('MSL Pressure + 10 m Wind', 'hPa'),
    'tp': ('Accumulated Precipitation', 'mm'),
    'tp24': ('24-hour Precipitation', 'mm'),
    'combo': ('Rain + MSLP + Wind', 'mm/3h'),
    'wind10': ('10 m Wind Speed', 'm/s'),
    'gh500': ('500 hPa Geopotential Height', 'gpm'),
    'tcc': ('Total Cloud Cover', '%'),
    'cape': ('CAPE (most-unstable)', 'J/kg'),
    'steer': ('Deep-Layer Steering Flow (850–200 hPa)', 'm/s'),
    'satdef': ('Saturation Deficit (700–500 hPa)', 'g/kg'),
    'secondary': ('Azimuthal-Mean Secondary Circulation', 'm/s'),
    'outflow': ('Outflow Strength (200 hPa)', '10⁻⁵ s⁻¹'),
}

# — সাইক্লোন ডায়াগনস্টিকের লেভেল সেট (ECMWF IFS pressure-level ডেটা) —
STEER_LEVELS = [850, 700, 500, 300, 200]        # steering layer (চাপ-ওজনিত গড়)
SATDEF_LEVELS = [700, 600, 500]                  # saturation deficit layer
SECTION_LEVELS = [200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]  # ascending p পরে সাজানো হয়

# ============================ ভূ-লেয়ার (GeoJSON) =============================
GEOJSON_BASES = [
    'https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/',
    'https://cdn.jsdelivr.net/gh/nvkelso/natural-earth-vector@master/geojson/',
]


class Http:
    """ECMWF-অনুগ্রহী HTTP ক্লায়েন্ট: ডিলে + রিট্রাই + ব্যাকঅফ"""

    def __init__(self):
        self.last = 0.0
        self.n_requests = 0
        self.n_bytes = 0
        import threading
        self._lock = threading.Lock()

    def _polite(self):
        with self._lock:  # থ্রেড-নিরাপদ গ্লোবাল ব্যবধান
            wait = REQUEST_DELAY - (time.time() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()

    def get(self, url, byte_range=None, retries=None):
        headers = dict(UA)
        if byte_range:
            headers['Range'] = 'bytes=%d-%d' % byte_range
        delay = 3.0
        max_retries = MAX_RETRIES if retries is None else max(1, int(retries))
        for attempt in range(max_retries):
            self._polite()
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=90) as r:
                    data = r.read()
                self.n_requests += 1
                self.n_bytes += len(data)
                return data
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    raise FileNotFoundError(url)
                print(f'  ⚠ HTTP {e.code} — {delay:.0f}s পর আবার চেষ্টা ({attempt+1}/{max_retries})')
            except Exception as e:
                print(f'  ⚠ নেটওয়ার্ক সমস্যা: {e} — {delay:.0f}s পর আবার ({attempt+1}/{max_retries})')
            if attempt + 1 < max_retries:
                time.sleep(delay)
                delay = min(delay * 3, 90)
        raise ConnectionError('সার্ভার থেকে ডাউনলোড ব্যর্থ: ' + url)

    def status_ok(self, url):
        try:
            self.get(url)
            return True
        except FileNotFoundError:
            return False

    def exists(self, url):
        """হালকা প্রোব (1-বাইট রেঞ্জ + একবার চেষ্টা) — রান খোঁজার জন্য।
        যেকোনো ব্যর্থতায় False, যাতে পরের প্রার্থী রান চেষ্টা করা যায়।"""
        try:
            self.get(url, byte_range=(0, 0), retries=1)
            return True
        except Exception:
            return False


HTTP = Http()


def load_geojson(kind, res):
    """kind: 'coastline' | 'admin_0_boundary_lines_land'; res: '110m' | '50m'"""
    fname = f'ne_{res}_{kind}.geojson'
    os.makedirs(CACHE_DIR, exist_ok=True)
    local = os.path.join(CACHE_DIR, fname)
    if not (POOL_CACHE and os.path.exists(local)):
        last_err = None
        for base in GEOJSON_BASES:
            try:
                print(f'  বেসম্যাপ লেয়ার: {fname} ডাউনলোড…')
                data = HTTP.get(base + fname)
                open(local, 'wb').write(data)
                break
            except Exception as e:
                last_err = e
        else:
            raise ConnectionError(f'বেসম্যাপ লেয়ার পাওয়া যায়নি: {last_err}')
    with open(local, 'r', encoding='utf-8') as f:
        gj = json.load(f)
    lines = []
    for feat in gj.get('features', []):
        geom = feat.get('geometry') or {}
        t = geom.get('type')
        if t == 'LineString':
            lines.append(geom['coordinates'])
        elif t == 'MultiLineString':
            lines.extend(geom['coordinates'])
    return lines


# ====================== GRIB2 পার্সার + CCSDS ডিকোডার ======================
# (pure Python — Eccodes/Cgrib ছাড়াই GRIB2 ফাইল পড়ে)

ROS = 5  # libaec এর zero-block ধ্রুবক


def _make_se_table():
    t = []
    k = 0
    for i in range(13):
        ms = k
        for j in range(i + 1):
            t.append((i, ms))
            k += 1
    return t


_SE_TAB = _make_se_table()


class _Bits:
    __slots__ = ('data', 'n', 'pos', 'acc', 'bits')

    def __init__(self, data):
        self.data = data
        self.n = len(data)
        self.pos = 0
        self.acc = 0
        self.bits = 0

    def _fill(self):
        need = (63 - self.bits) >> 3
        if need:
            avail = self.n - self.pos
            if avail <= 0:
                return
            m = need if need < avail else avail
            b = self.data[self.pos:self.pos + m]
            self.acc = (self.acc << (m << 3)) | int.from_bytes(b, 'big')
            self.bits += m << 3
            self.pos += m

    def get(self, nb):
        if self.bits < nb:
            self._fill()
        self.bits -= nb
        if self.bits < 0:
            self.bits = 0
            return 0
        return (self.acc >> self.bits) & ((1 << nb) - 1)

    def get_fs(self):
        fs = 0
        bits = self.bits
        acc = self.acc & ((1 << bits) - 1)
        while True:
            hb = acc.bit_length()
            if hb == 0:
                fs += bits
                bits = 0
                need = (63 - bits) >> 3
                avail = self.n - self.pos
                if avail <= 0:
                    self.acc = 0
                    self.bits = 0
                    return fs
                m = need if need < avail else avail
                acc = int.from_bytes(self.data[self.pos:self.pos + m], 'big')
                bits = m << 3
                self.pos += m
                continue
            fs += bits - hb
            self.bits = hb - 1
            self.acc = acc & ((1 << (hb - 1)) - 1)
            return fs


def _aec_decode(buf, n_out, bits, block, rsi):
    """CCSDS 121.0 / libaec স্ট্রিম ডিকোড (PREPROCESS mode, unsigned)।
       ECMWF-এর ccsdsFlags=14 ধারণা করে। libaec-এর decode.c-এর বিশ্বস্ত পোর্ট।"""
    if bits > 16:
        id_len = 5
    elif bits > 8:
        id_len = 4
    else:
        id_len = 3
    modi = 1 << id_len
    xmax = (1 << bits) - 1
    med = (xmax >> 1) + 1
    rsi_size = rsi * block
    br = _Bits(buf)
    get = br.get
    get_fs = br.get_fs
    out = [0] * n_out
    oi = 0
    se = _SE_TAB

    while oi < n_out:
        seg_n = min(rsi_size, n_out - oi)
        seg = [0] * seg_n
        i = 0
        ref = True
        while i < seg_n:
            opt = get(id_len)
            nthis = (block - 1) if ref else block
            if opt == 0:
                kind = get(1)
                if ref:
                    seg[i] = get(bits)
                    i += 1
                if kind == 1:
                    j = 1 if ref else 0
                    while j < block:
                        mval = get_fs()
                        s_, base_ = se[mval]
                        d1 = mval - base_
                        if (j & 1) == 0:
                            seg[i] = s_ - d1
                            i += 1
                            j += 1
                        seg[i] = d1
                        i += 1
                        j += 1
                else:
                    zb = get_fs() + 1
                    if zb == ROS:
                        b = i // block
                        rem = rsi - b
                        zb = 64 - (b % 64)
                        if zb > rem:
                            zb = rem
                    elif zb > ROS:
                        zb -= 1
                    zc = zb * block - (1 if ref else 0)
                    if i + zc > seg_n:
                        zc = seg_n - i
                    i += zc
            elif opt == modi - 1:
                for _ in range(block):
                    if i < seg_n:
                        seg[i] = get(bits)
                        i += 1
                    else:
                        get(bits)
            else:
                k = opt - 1
                if ref:
                    seg[i] = get(bits)
                    i += 1
                hi = [get_fs() for _ in range(nthis)]
                if k:
                    for t in range(nthis):
                        seg[i] = (hi[t] << k) + get(k)
                        i += 1
                else:
                    for t in range(nthis):
                        seg[i] = hi[t]
                        i += 1
            ref = False
        if seg_n:
            data = seg[0]
            out[oi] = data
            mm = med
            xm = xmax
            for idx in range(1, seg_n):
                d = seg[idx]
                half_d = (d >> 1) + (d & 1)
                mask = xm if (data & mm) else 0
                if half_d <= (mask ^ data):
                    data += (d >> 1) if (d & 1) == 0 else -((d >> 1) + 1)
                else:
                    data = mask ^ d
                out[oi + idx] = data
        oi += seg_n
    return out


def _simple_unpack(sec5, sec7, npts):
    """GRIB2 template 5.0 (simple packing) ডিকোড — ফলব্যাক হিসেবে রাখা হয়েছে"""
    R = struct.unpack('>f', sec5[11:15])[0]
    e = int.from_bytes(sec5[15:17], 'big')
    E = -(e & 0x7FFF) if e & 0x8000 else e
    d = int.from_bytes(sec5[17:19], 'big')
    D = -(d & 0x7FFF) if d & 0x8000 else d
    bits = sec5[19]
    if bits == 0:
        return np.full(npts, R, dtype=np.float64)
    vals = np.zeros(npts, dtype=np.int64)
    cur = 0
    nbits = 0
    vi = 0
    for byte in sec7:
        cur = (cur << 8) | byte
        nbits += 8
        while nbits >= bits and vi < npts:
            nbits -= bits
            vals[vi] = (cur >> nbits) & ((1 << bits) - 1)
            vi += 1
        cur &= (1 << nbits) - 1
    return (R + vals.astype(np.float64) * 2.0 ** E) / 10.0 ** D


class GribField:
    __slots__ = ('values', 'lats', 'lons', 'name', 'step', 'units', 'level')

    def __init__(self, values, lats, lons):
        self.values = values
        self.lats = lats
        self.lons = lons


def decode_grib2_message(buf):
    """একটি GRIB2 মেসেজ (বাইটস) → GribField (2D numpy অ্যারে, lats, lons)"""
    if buf[:4] != b'GRIB':
        raise ValueError('GRIB মেসেজ নয়')
    total = int.from_bytes(buf[8:16], 'big')
    off = 16
    s3 = s5 = s7 = None
    bitmap = 255
    while off < total - 4:
        ln = int.from_bytes(buf[off:off + 4], 'big')
        num = buf[off + 4]
        if num == 3:
            s3 = buf[off:off + ln]
        elif num == 5:
            s5 = buf[off:off + ln]
        elif num == 6:
            bitmap = buf[off + 5]
        elif num == 7:
            s7 = buf[off + 5:off + ln]
        off += ln
    if s3 is None or s5 is None or s7 is None:
        raise ValueError('অপূর্ণ GRIB মেসেজ')
    gtmpl = int.from_bytes(s3[12:14], 'big')
    if gtmpl != 0:
        raise ValueError(f'গ্রিড টেমপ্লেট {gtmpl} — শুধু regular lat/lon (3.0) সাপোর্টেড')
    npts = int.from_bytes(s3[6:10], 'big')
    Ni = int.from_bytes(s3[30:34], 'big')
    Nj = int.from_bytes(s3[34:38], 'big')
    lat1 = int.from_bytes(s3[46:50], 'big', signed=True) / 1e6
    lon1 = int.from_bytes(s3[50:54], 'big') / 1e6
    lat2 = int.from_bytes(s3[55:59], 'big', signed=True) / 1e6
    di = int.from_bytes(s3[63:67], 'big') / 1e6
    dj = int.from_bytes(s3[67:71], 'big') / 1e6

    if bitmap != 255:
        print('  ⚠ সতর্কতা: মিসিং-ভ্যালু বিটম্যাপ উপস্থিত — হ্যান্ডল করা হয়নি')

    tmpl5 = int.from_bytes(s5[9:11], 'big')
    if tmpl5 == 42:
        R = struct.unpack('>f', s5[11:15])[0]
        e = int.from_bytes(s5[15:17], 'big')
        E = -(e & 0x7FFF) if e & 0x8000 else e
        d = int.from_bytes(s5[17:19], 'big')
        D = -(d & 0x7FFF) if d & 0x8000 else d
        bits = s5[19]
        block = s5[22]
        rsi = int.from_bytes(s5[23:25], 'big')
        npts5 = int.from_bytes(s5[5:9], 'big')
        ints = _aec_decode(s7, npts5, bits, block, rsi)
        values = (R + np.asarray(ints, dtype=np.float64) * 2.0 ** E) / 10.0 ** D
    elif tmpl5 == 0:
        values = _simple_unpack(s5, s7, npts)
    else:
        raise ValueError(f'প্যাকিং টেমপ্লেট 5.{tmpl5} সাপোর্টেড নয় (শুধু 5.0 ও 5.42)')

    if values.size != npts:
        raise ValueError(f'ডেটা পয়েন্ট মিলছে না: {values.size} != {npts}')

    # গ্রিড অভিমুখ: ECMWF গ্লোবাল — lat 90→-90 (উত্তর থেকে), lon1 থেকে পূর্বে
    lats = lat1 - dj * np.arange(Nj) if lat1 > lat2 else lat1 + dj * np.arange(Nj)
    raw = (lon1 + di * np.arange(Ni)) % 360.0
    vals2d = values.reshape(Nj, Ni)
    # lon গুলোকে -180..180 এ নিয়ে ascending sort (dateline খুঁজে roll)
    lons180 = (raw + 180.0) % 360.0 - 180.0
    diffs = np.diff(lons180)
    seam = int(np.argmin(diffs))  # সবচেয়ে নেতিবাচক লাফ = সীমারেখা
    if diffs[seam] < 0:
        k = seam + 1
        lons180 = np.concatenate([lons180[k:], lons180[:k]])
        vals2d = np.concatenate([vals2d[:, k:], vals2d[:, :k]], axis=1)
    return GribField(vals2d, lats, lons180)


# ============================= ECMWF ডেটা ক্লায়েন্ট =============================

def find_latest_run():
    """সর্বশেষ উপলব্ধ IFS রান খুঁজে বের করে (আজ → গতকাল, 18/12/06/00Z)"""
    now = dt.datetime.now(dt.timezone.utc)
    cands = []
    for dback in range(0, 3):
        day = now - dt.timedelta(days=dback)
        dstr = day.strftime('%Y%m%d')
        for hh in ('18', '12', '06', '00'):
            rd = dt.datetime(day.year, day.month, day.day, int(hh), tzinfo=dt.timezone.utc)
            age_h = (now - rd).total_seconds() / 3600.0
            if age_h < 5.5:      # রান শেষ + ওপেন ডেটা প্রকাশে ~৫-৬ ঘণ্টা লাগে
                continue
            cands.append((dstr, hh))
        if cands:
            break
    cache_file = os.path.join(CACHE_DIR, 'last_run.json')
    for dstr, hh in cands[:8]:
        url = f'{BASE_URL}/{dstr}/{hh}z/ifs/0p25/oper/{dstr}{hh}0000-0h-oper-fc.index'
        print(f'  রান খোঁজা হচ্ছে: {dstr} {hh}Z …', flush=True)
        # প্রোব ব্যর্থ হলে (404/নেটওয়ার্ক) এই রান বাদ দিয়ে পরেরটা দেখো —
        # একটা প্রার্থীর সমস্যায় পুরো খোঁজ থামবে না।
        if HTTP.exists(url):
            run = {'date': dstr, 'hour': hh, 'found': time.time()}
            try:
                os.makedirs(CACHE_DIR, exist_ok=True)
                json.dump(run, open(cache_file, 'w'))
            except Exception:
                pass
            return dstr, hh
    raise RuntimeError('কোনো উপলব্ধ রান পাওয়া গেল না — পরে চেষ্টা করুন বা ম্যানুয়াল দিন।')


def run_datetime(run_date, run_hour):
    return dt.datetime.strptime(run_date + f'{int(run_hour):02d}', '%Y%m%d%H').replace(tzinfo=dt.timezone.utc)


def fmt_time(t):
    return t.strftime('%d %b %Y %H:%M UTC')


def fmt_short(t):
    return t.strftime('%d %b %HZ')   # '16 Oct 06Z' — সাবটাইটেলের জন্য কমপ্যাক্ট


def index_url(run, step):
    d, h = run
    return f'{BASE_URL}/{d}/{h}z/ifs/0p25/oper/{d}{h}0000-{step}h-oper-fc.index'


def grib_url(run, step):
    d, h = run
    return f'{BASE_URL}/{d}/{h}z/ifs/0p25/oper/{d}{h}0000-{step}h-oper-fc.grib2'


def run_cache_dir(run):
    """রান-ভিত্তিক ক্যাশ-ফোল্ডার (তারিখ_সময়z) — পরিষ্কার ব্যবস্থাপনার জন্য"""
    d = os.path.join(CACHE_DIR, f'{run[0]}_{run[1]}z')
    os.makedirs(d, exist_ok=True)
    return d


def load_index(run, step):
    """স্টেপের index ফাইল (ছোট JSON-lines) নামায় + ক্যাশে করে"""
    cf = os.path.join(run_cache_dir(run), f'idx_{step}.jsonl')
    if POOL_CACHE and os.path.exists(cf):
        lines = open(cf, 'r', encoding='utf-8').read().splitlines()
    else:
        try:
            data = HTTP.get(index_url(run, step), retries=2)
        except Exception:
            # 404 হোক বা সাময়িক নেটওয়ার্ক সমস্যা — স্টেপ এখন उपलब्ध নয় ধরো;
            # কলার (auto ফলব্যাক / পরিষ্কার এরর মেসেজ) সিদ্ধান্ত নেবে।
            return None
        lines = data.decode().splitlines()
        try:
            open(cf, 'w', encoding='utf-8').write('\n'.join(lines))
        except Exception:
            pass
    entries = []
    for ln in lines:
        try:
            entries.append(json.loads(ln))
        except Exception:
            pass
    return entries


def step_available(run, step, map_key='combo'):
    """এই রানে কাঙ্ক্ষিত ফোরকাস্ট স্টেপের ডেটা প্রকাশিত হয়েছে কি না।
    (tp24/combo-র পেছনের উইন্ডো-স্টেপও লাগে, সেটাও যাচাই করে।)"""
    step = int(step)
    if load_index(run, step) is None:
        return False
    if map_key == 'tp24':
        return step >= 24 and load_index(run, step - 24) is not None
    if map_key == 'combo':
        prev = max(0, step - max(1, STEP_H))
        if 0 < prev < step and load_index(run, prev) is None:
            return False
    return True


def msg_cache_path(run, step, param, lev):
    levstr = 'sfc' if lev is None else str(lev)
    return os.path.join(run_cache_dir(run), f'msg_{step}_{param}_{levstr}.grb')


def fetch_fields(run, requests, step):
    """একই স্টেপের একাধিক প্যারামিটার একসাথে নামায় (সংলগ্ন রেঞ্জ মার্জ করে)।
       requests: [(grib_param, level_or_None), ...]
       ফলাফল: {(param, level): GribField}"""
    out = {}
    todo = []
    for key in requests:
        cf = msg_cache_path(run, step, key[0], key[1])
        if POOL_CACHE and os.path.exists(cf):
            continue
        todo.append(key)
    if todo:
        idx = load_index(run, step)
        if idx is None:
            print(f'  ⚠ স্টেপ +{step}h এখনো উপলব্ধ নয় — বাদ দেওয়া হলো')
            return out

        def find_ent(key):
            p, lev = key
            for e in idx:
                if e.get('param') == p and (lev is None or str(e.get('levelist')) == str(lev)):
                    return e
            return None

        spans = []  # (start, end_inclusive, [keys,...])
        for key in todo:
            ent = find_ent(key)
            if ent is None:
                print(f'  ⚠ {key[0]} (level={key[1]}) এই স্টেপে নেই')
                continue
            o, l = int(ent['_offset']), int(ent['_length'])
            spans.append([o, o + l - 1, [key]])
        if spans:
            spans.sort(key=lambda x: x[0])
            merged = [spans[0]]
            for s in spans[1:]:
                cur = merged[-1]
                if s[0] - cur[1] <= 512 and s[1] - cur[0] < 14_000_000:
                    cur[1] = max(cur[1], s[1])
                    cur[2].extend(s[2])
                else:
                    merged.append(s)
            ent_cache = {}
            for chunk in merged:
                a, b, keys = chunk[0], chunk[1], chunk[2]
                print(f'  ↓ +{step}h: {len(keys)}টি ফিল্ড ({(b-a+1)/1e6:.2f} MB)', flush=True)
                try:
                    data = HTTP.get(grib_url(run, step), byte_range=(a, b))
                except Exception as e:
                    print(f'  ⚠ ডাউনলোড ব্যর্থ: {e}')
                    continue
                for key in keys:
                    if key not in ent_cache:
                        ent_cache[key] = find_ent(key)
                    ent = ent_cache[key]
                    if ent is None:
                        continue
                    o, l = int(ent['_offset']), int(ent['_length'])
                    cf = msg_cache_path(run, step, key[0], key[1])
                    try:
                        open(cf, 'wb').write(data[o - a:o - a + l])
                    except Exception:
                        pass
    for key in requests:
        cf = msg_cache_path(run, step, key[0], key[1])
        if os.path.exists(cf):
            try:
                out[key] = decode_grib2_message(open(cf, 'rb').read())
            except Exception as e:
                print(f'  ⚠ ডিকোড সমস্যা {key}: {e}')
                try:
                    os.remove(cf)
                except OSError:
                    pass
    return out


# ============================ প্যারামিটার প্রস্তুতি ============================

def get_field(run, step, requests):
    """ধাপে ধাপে প্রয়োজনীয় ফিল্ড সংগ্রহ; কিছু মিস হলে None"""
    got = fetch_fields(run, requests, step)
    for r in requests:
        if r not in got:
            return None
    return got


def convert(f, kind):
    v = f.values
    if kind in ('t2m', 't850'):
        return v - 273.15
    if kind == 'msl':
        return v / 100.0
    if kind in ('tp', 'tp24'):
        return v * 1000.0
    return v


def _fetch_levels(run, step, params, levels):
    """এক স্টেপে অনেক লেভেলের ফিল্ড একসাথে নামায়; কিছু মিস হলে None।"""
    reqs = [(p, lev) for lev in levels for p in params]
    return get_field(run, step, reqs)


def _layer_mean_wind(fields, levels):
    """চাপ-ওজনিত (trapezoid) স্তর-গড় u,v — ফিল্ড ডিক্ট থেকে।"""
    p = np.asarray(sorted(levels, reverse=True), dtype=float)   # 850 → 200
    w = np.abs(np.gradient(p))
    w[0] *= 0.5
    w[-1] *= 0.5
    w = w / w.sum()
    u = sum(wi * fields[('u', int(pi))].values for wi, pi in zip(w, p))
    v = sum(wi * fields[('v', int(pi))].values for wi, pi in zip(w, p))
    f0 = fields[('u', int(p[0]))]
    return f0.lats, f0.lons, u, v


def _qsat_gkg(T_k, p_hpa):
    """সম্পৃক্ত স্পেসিফিক হিউমিডিটি (g/kg) — Bolton-এর সূত্রে।"""
    Tc = T_k - 273.15
    es = 6.112 * np.exp(17.67 * Tc / (Tc + 243.5))          # hPa
    qs = 0.622 * es / (p_hpa - 0.378 * es)                  # kg/kg
    return qs * 1000.0


def prepare_map_data(run, kind, step):
    """প্রতিটি ম্যাপ কী-এর জন্য ডেটা নামায় + নির্ণীত অ্যারে রিটার্ন করে"""
    if kind == 't2m':
        g = get_field(run, step, [('2t', None)])
        if not g:
            return None
        f = g[('2t', None)]
        return f.lats, f.lons, convert(f, 't2m'), None
    if kind == 't850':
        g = get_field(run, step, [('t', 850)])
        if not g:
            return None
        f = g[('t', 850)]
        return f.lats, f.lons, convert(f, 't850'), None
    if kind == 'msl':
        g = get_field(run, step, [('msl', None)])
        if not g:
            return None
        f = g[('msl', None)]
        return f.lats, f.lons, convert(f, 'msl'), None
    if kind == 'msl_wind':
        g = get_field(run, step, [('msl', None), ('10u', None), ('10v', None)])
        if not g:
            return None
        f = g[('msl', None)]
        u, v = g[('10u', None)].values, g[('10v', None)].values
        return f.lats, f.lons, convert(f, 'msl'), (u, v)
    if kind == 'wind10':
        g = get_field(run, step, [('10u', None), ('10v', None)])
        if not g:
            return None
        u, v = g[('10u', None)].values, g[('10v', None)].values
        f = g[('10u', None)]
        return f.lats, f.lons, np.hypot(u, v), (u, v)
    if kind == 'tp':
        g = get_field(run, step, [('tp', None)])
        if not g:
            return None
        f = g[('tp', None)]
        return f.lats, f.lons, convert(f, 'tp'), None
    if kind == 'tp24':
        if step < 24:
            return None
        g1 = get_field(run, step, [('tp', None)])
        g0 = get_field(run, step - 24, [('tp', None)])
        if not g1 or not g0:
            return None
        f = g1[('tp', None)]
        arr = np.maximum(0.0, convert(f, 'tp') - convert(g0[('tp', None)], 'tp'))
        return f.lats, f.lons, arr, None
    if kind == 'combo':
        # বৃষ্টি (গত STEP_H ঘণ্টা) শেইড + MSLP কনট্যুর + 10m বাতাসের তীর — অ্যানিমেশনের জন্য
        prev = max(0, step - max(1, STEP_H))
        g = get_field(run, step, [('msl', None), ('10u', None), ('10v', None), ('tp', None)])
        if not g:
            return None
        f = g[('tp', None)]
        if prev >= step or prev == 0:
            rain = convert(f, 'tp')
        else:
            g0 = get_field(run, prev, [('tp', None)])
            if not g0:
                return None
            rain = np.maximum(0.0, convert(f, 'tp') - convert(g0[('tp', None)], 'tp'))
        u, v = g[('10u', None)].values, g[('10v', None)].values
        return f.lats, f.lons, rain, (u, v, convert(g[('msl', None)], 'msl'))
    if kind == 'gh500':
        g = get_field(run, step, [('gh', 500)])
        if not g:
            return None
        f = g[('gh', 500)]
        return f.lats, f.lons, f.values, None
    if kind == 'tcc':
        g = get_field(run, step, [('tcc', None)])
        if not g:
            return None
        f = g[('tcc', None)]
        return f.lats, f.lons, f.values, None
    if kind == 'cape':
        g = get_field(run, step, [('mucape', None)])
        if not g:
            return None
        f = g[('mucape', None)]
        return f.lats, f.lons, f.values, None
    if kind == 'steer':
        # 850–200 hPa চাপ-ওজনিত গড় বাতাস: গতি শেইড + স্ট্রিমলাইন (extra)
        g = _fetch_levels(run, step, ['u', 'v'], STEER_LEVELS)
        if not g:
            return None
        lats, lons, u, v = _layer_mean_wind(g, STEER_LEVELS)
        return lats, lons, np.hypot(u, v), (u, v)
    if kind == 'satdef':
        # 700–500 hPa স্তর-গড় saturation deficit (g/kg): qs·(1 − RH/100)
        g = _fetch_levels(run, step, ['t', 'r'], SATDEF_LEVELS)
        if not g:
            return None
        defs = []
        for lev in SATDEF_LEVELS:
            T = g[('t', lev)].values
            RH = np.clip(g[('r', lev)].values, 0.0, 100.0)
            defs.append(_qsat_gkg(T, float(lev)) * (1.0 - RH / 100.0))
        f = g[('t', SATDEF_LEVELS[0])]
        return f.lats, f.lons, np.mean(defs, axis=0), None
    print(f'  ⚠ অজানা ম্যাপ কী: {kind}')
    return None


# ================================ প্লট ইঞ্জিন ================================

def norm_lat(v):
    return f'{abs(v):g}°' + ('N' if v >= 0 else 'S')


def norm_lon(v):
    v = ((v + 180) % 360) - 180
    return f'{abs(v):g}°' + ('E' if v >= 0 else 'W')


def crop_to_bbox(lats, lons, arr, bbox, extra=()):
    la0, la1, lo0, lo1 = bbox
    pad = 0.2
    im = (lats >= la0 - pad) & (lats <= la1 + pad)
    jm = (lons >= lo0 - pad) & (lons <= lo1 + pad)
    if im.sum() < 3 or jm.sum() < 3:
        raise ValueError('রিজিয়নের মধ্যে গ্রিড পয়েন্ট নেই')
    outs = [arr[np.ix_(im, jm)]] + [x[np.ix_(im, jm)] for x in extra]
    return lats[im], lons[jm], outs[0], tuple(outs[1:])


def nice_tick_step(span, target=7):
    for c in (0.5, 1, 2, 2.5, 5, 10, 15, 30, 45, 60):
        if span / c <= target:
            return c
    return 60


def make_figure(bbox, th, title, subtitle, unit_tag):
    """ম্যাপ-কেন্দ্রিক লেয়াউট: ফিগারের মাপ ম্যাপের অস্পেক্ট থেকে বের হয় — কোনো দিকে অতিরিক্ত ফাঁকা থাকে না"""
    la0, la1, lo0, lo1 = bbox
    latspan, lonspan = la1 - la0, lo1 - lo0
    aspect = latspan / lonspan                      # equirectangular অনুপাত
    map_w = min(11.0, 8.8 / aspect)                 # ম্যাপের ইঞ্চি-মাপ (প্রস্থ ≤ 11, উচ্চতা ≤ 8.8)
    map_h = map_w * aspect
    W = map_w + 1.0                                 # বামে 0.55 (টিক-লেবেল), ডানে 0.45
    HEADER_IN, CBAR_IN, FOOT_IN = 1.05, 0.90, 0.45
    H = HEADER_IN + map_h + CBAR_IN + FOOT_IN
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=th['fig'])
    L = 0.55 / W                                    # বাম মার্জিন (ফিগার-ভগ্নাংশ)
    R = 1.0 - 0.45 / W                              # ডান প্রান্ত

    # ---- হেডার ----
    fig.add_artist(plt.Rectangle((0.0, 1 - 0.05 / H), 1.0, 0.05 / H, transform=fig.transFigure,
                                 facecolor=th['accent'], edgecolor='none', clip_on=False))
    fig.text(L, 1 - 0.46 / H, title, fontsize=15, weight='bold', color=th['text'], va='center')
    fig.text(L, 1 - 0.86 / H, subtitle, fontsize=8.8, color=th['sub'], va='center')
    fig.text(R, 1 - 0.46 / H, 'ECMWF · IFS', fontsize=9.5, weight='bold',
             color=th['badge_fg'], ha='right', va='center', clip_on=False,
             bbox=dict(boxstyle='round,pad=0.38', facecolor=th['accent'], edgecolor='none'))
    if unit_tag:
        fig.text(R, 1 - 0.86 / H, unit_tag, fontsize=8.5, color=th['sub'], ha='right', va='center')

    # ---- ম্যাপ (পুরো ফ্রেম পূর্ণ) ----
    ax = fig.add_axes([L, (FOOT_IN + CBAR_IN) / H, map_w / W, map_h / H])
    ax.set_facecolor(th['ax'])

    # ---- কালারবার (ম্যাপের নিচে, ম্যাপ-অনুপাতে) ----
    cbw = 0.72 * map_w / W
    cax = fig.add_axes([L + (map_w / W - cbw) / 2, (FOOT_IN + 0.36) / H, cbw, 0.15 / H])

    return fig, ax, cax


# ---------- স্টেট/বিভাগ বর্ডার: বান্ডল করা bd_states.geojson (অফলাইনে চলে) ----------
_STATE_LINES = None


def state_lines():
    global _STATE_LINES
    if _STATE_LINES is None:
        _STATE_LINES = []
        here = os.path.dirname(os.path.abspath(__file__))
        for cand in (os.path.join(here, 'bd_states.geojson'), 'bd_states.geojson',
                     os.path.join(CACHE_DIR, 'bd_states.geojson')):
            if os.path.exists(cand):
                try:
                    with open(cand, 'r', encoding='utf-8') as f:
                        gj = json.load(f)
                    for feat in gj.get('features', []):
                        geom = feat.get('geometry') or {}
                        tp = geom.get('type')
                        if tp == 'LineString':
                            _STATE_LINES.append(geom['coordinates'])
                        elif tp == 'MultiLineString':
                            _STATE_LINES.extend(geom['coordinates'])
                except Exception:
                    pass
                break
    return _STATE_LINES


_BASE_LINES = {}


def base_lines(kind, res):
    key = (kind, res)
    if key not in _BASE_LINES:
        _BASE_LINES[key] = load_geojson(kind, res)
    return _BASE_LINES[key]


def draw_baseline(ax, bbox, th, scale):
    res = '110m' if (bbox[3] - bbox[2]) > 70 else '50m'
    coast, border = [], []
    try:
        coast = base_lines('coastline', res)
    except Exception as e:
        print(f'  ⚠ কোস্টলাইন লোড হয়নি: {e}')
    try:
        border = base_lines('admin_0_boundary_lines_land', res)
    except Exception:
        pass
    la0, la1, lo0, lo1 = bbox

    def to_segments(lines):
        segs = []
        for line in lines:
            xy = [(x, y) for x, y in line if la0 - 1 <= y <= la1 + 1 and lo0 - 1 <= x <= lo1 + 1]
            if len(xy) >= 2:
                segs.append(xy)
        return segs

    csegs = to_segments(coast)
    if csegs:
        ax.add_collection(LineCollection(csegs, colors=th['coast'], linewidths=0.9 * scale,
                                         zorder=6, capstyle='round', joinstyle='round'))
    bsegs = to_segments(border)
    if bsegs:
        ax.add_collection(LineCollection(bsegs, colors=th['border'], linewidths=0.75 * scale,
                                         zorder=5, alpha=0.95))
    if STATES_ON:
        sseg = to_segments(state_lines())
        if sseg:
            ax.add_collection(LineCollection(sseg, colors=th['border'], linewidths=0.5 * scale,
                                             zorder=4, alpha=0.5, linestyle=(0, (4, 2.5))))


def style_axes(ax, bbox, th, tick_step=None):
    la0, la1, lo0, lo1 = bbox
    ax.set_xlim(lo0, lo1)
    ax.set_ylim(la0, la1)
    ax.set_aspect('equal')
    for s in ax.spines.values():
        s.set_color(th['sub'])
        s.set_linewidth(0.7)
    ts = tick_step or nice_tick_step(max(la1 - la0, lo1 - lo0))
    xt = np.arange(math.floor(lo0 / ts) * ts, lo1 + ts * 0.5, ts)
    yt = np.arange(math.floor(la0 / ts) * ts, la1 + ts * 0.5, ts)
    ax.set_xticks([x for x in xt if -360 <= x <= 360])
    ax.set_yticks(yt)
    ax.set_xticklabels([norm_lon(x) for x in ax.get_xticks()], fontsize=8, color=th['sub'])
    ax.set_yticklabels([norm_lat(y) for y in yt], fontsize=8, color=th['sub'])
    ax.tick_params(length=0, pad=2)
    ax.grid(True, color=th['grid'], linewidth=0.6, alpha=0.65, zorder=3,
            linestyle=(0, (3, 4)))


def draw_cities(ax, bbox, th, cities):
    la0, la1, lo0, lo1 = bbox
    span = max(la1 - la0, lo1 - lo0)
    fs = 7.5 if span < 12 else 6.5
    for name, (lat, lon) in cities.items():
        if not (la0 < lat < la1 and lo0 < lon < lo1):
            continue
        clean = name.split('/')[-1]
        ax.plot(lon, lat, 'o', ms=4.2, color='#111', mec='#fff', mew=0.9, zorder=9)
        t = ax.text(lon + span * 0.006, lat + span * 0.006, clean, fontsize=fs,
                    color='#111', weight='bold', zorder=9)
        t.set_path_effects([patheffects.withStroke(linewidth=2.2, foreground='white', alpha=0.85)])


def plot_single(kind, step, data3, run, region_key, bbox, th):
    """একটা ম্যাপ রেন্ডার করে PNG bytes রিটার্ন করে (ওয়েব সংস্করণ)।"""
    lats, lons, arr, extra = data3
    levels, colors, extend = LEVELS[kind]
    cmap = _get_cmap(colors, kind)
    title0, unit = MAP_TITLES[kind]
    region_name = 'Global' if region_key == 'global' else REGION_LABELS.get(
        region_key, region_key.replace('_', ' ').title())
    rdt = run_datetime(*run)
    vdt = rdt + dt.timedelta(hours=step)
    if kind == 'tp24':
        valid_txt = f'Valid {fmt_short(vdt - dt.timedelta(hours=24))} → {fmt_short(vdt)}'
    elif kind == 'combo':
        valid_txt = f'Rain last {min(STEP_H, step)}h · Valid {fmt_short(vdt)} (+{step} h)'
    else:
        valid_txt = f'Valid {fmt_short(vdt)} (+{step} h)'
    subtitle = f'IFS HRES 0.25°   ·   Run {fmt_short(rdt)}   ·   {valid_txt}'
    fig, ax, cax = make_figure(bbox, th, f'{title0} — {region_name}', subtitle, f'[{unit}]')

    lats_c, lons_c, arr_c, extra_c = crop_to_bbox(lats, lons, arr, bbox,
                                                  extra=extra if extra else ())
    dlat = abs(lats_c[0] - lats_c[1])
    dlon = abs(lons_c[0] - lons_c[1])
    extent = [lons_c[0] - dlon / 2, lons_c[-1] + dlon / 2, lats_c[-1] - dlat / 2, lats_c[0] + dlat / 2]

    norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N if hasattr(cmap, 'N') else 256, clip=False)
    interp = 'bilinear' if (bbox[3] - bbox[2]) < 28 else 'nearest'
    im = ax.imshow(arr_c, extent=extent, origin='upper', cmap=cmap, norm=norm,
                   interpolation=interp, zorder=2)

    style_axes(ax, bbox, th, REGION_PRESETS.get(region_key, {}).get('tick') if region_key else None)

    # কনট্যুর / বার্ব ওভারলে
    scale = 1.35 if (bbox[3] - bbox[2]) < 30 else 1.0
    if kind in ('msl', 'msl_wind'):
        clev = np.arange(940, 1068, 4)
        cs = ax.contour(lons_c, lats_c, arr_c, levels=clev, colors='#14202b',
                        linewidths=0.65, alpha=0.85, zorder=7)
        ax.clabel(cs, fmt='%d', fontsize=7, inline=True, inline_spacing=1)
    if kind == 'combo':
        cs = ax.contour(lons_c, lats_c, extra_c[2], levels=np.arange(940, 1068, 4),
                        colors='#14202b', linewidths=0.6, alpha=0.8, zorder=7)
        ax.clabel(cs, fmt='%d', fontsize=7, inline=True, inline_spacing=1)
    if kind == 'gh500':
        cs = ax.contour(lons_c, lats_c, arr_c, levels=np.arange(4200, 6400, 60),
                        colors='#14202b', linewidths=0.6, alpha=0.8, zorder=7)
        ax.clabel(cs, fmt='%d', fontsize=7, inline=True)
    if extra_c and kind in ('msl_wind', 'wind10', 'combo'):
        u_c, v_c = extra_c[0], extra_c[1]
        span = bbox[3] - bbox[2]
        st = max(2, int(round(span / (0.25 * 20))))
        xs, ys = np.meshgrid(lons_c[::st], lats_c[::st])
        us, vs = u_c[::st, ::st], v_c[::st, ::st]  # m/s
        spd = np.maximum(np.hypot(us, vs), 0.6)  # শূন্য-গতি গার্ড
        # ইউনিফর্ম সরু ছোট তীর (দিক) + গতি কালার-ফিলে — পাতলা ও সূক্ষ্ম, কিছু ঢাকে না
        ax.quiver(xs, ys, us / spd, vs / spd,
                  angles='xy', scale_units='xy', scale=4.6,
                  width=0.0021 * scale, headwidth=3.0, headlength=4.0, headaxislength=3.4,
                  minlength=0.5, pivot='middle', linewidth=0.4, color='#26313b',
                  alpha=0.78, zorder=8)
    if extra_c and kind == 'steer':
        # স্ট্রিমলাইন (ল্যাট ascending করে দিতে হয়)
        u_c, v_c = extra_c[0], extra_c[1]
        o = np.argsort(lats_c)
        density = 1.4 if (bbox[3] - bbox[2]) < 40 else 1.8
        ax.streamplot(lons_c, lats_c[o], u_c[o], v_c[o], color='#14202b',
                      density=density, linewidth=0.65, arrowsize=0.75, zorder=8)

    draw_baseline(ax, bbox, th, scale)
    if CITIES_ON and region_key in REGION_PRESETS and 'cities' in REGION_PRESETS[region_key]:
        draw_cities(ax, bbox, th, REGION_PRESETS[region_key]['cities'])

    # ---- ম্যাপের ভিতরে ওয়াটারমার্ক (ডান-নিচে) ----
    ax.text(0.985, 0.008, '@XPWEATHER', transform=ax.transAxes,
            fontsize=12 * min(scale, 1.35), weight='bold', color='#0d1217', alpha=0.6,
            ha='right', va='bottom', zorder=30).set_path_effects(
        [patheffects.withStroke(linewidth=2.6, foreground='white', alpha=0.75)])

    tick_stride = 1 if len(levels) <= 13 else 2
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', extend=extend,
                      ticks=levels[::tick_stride])
    cb.set_label(unit, fontsize=9, color=th['text'], labelpad=2)
    cb.ax.tick_params(labelsize=7.5, color=th['sub'], length=2.5)
    for t in cb.ax.get_xticklabels():
        t.set_color(th['sub'])
    cb.ax.xaxis.set_label_position('bottom')
    cb.outline.set_visible(False)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


# ===================== কোর-কেন্দ্রিক সাইক্লোন ডায়াগনস্টিক =====================
# ECMWF IFS pressure-level ডেটা থেকে: সেন্টার খোঁজা (850 hPa vorticity),
# azimuthal-mean secondary circulation (radius–height) ও 200 hPa outflow।
# omega আলাদা ফিল্ড না ধরে winds থেকে mass continuity দিয়ে নির্ণয় করা হয়
# (reanalysis cyclone/ প্যাকেজের একই পদ্ধতি)।

EARTH_R = 6.371e6
R_D = 287.04
GRAV = 9.80665


def _bilinear(lats, lons, arr, plat, plon):
    """(nlat, nlon) গ্রিড থেকে পয়েন্টে bilinear স্যাম্পল (lat যেকোনো ক্রমে)।"""
    lats = np.asarray(lats, float)
    lons = np.asarray(lons, float)
    if lats[0] > lats[-1]:
        lats, arr = lats[::-1], arr[::-1]
    plat = np.asarray(plat, float)
    plon = ((np.asarray(plon, float) + 180.0) % 360.0) - 180.0
    fi = np.interp(plat, lats, np.arange(lats.size), left=np.nan, right=np.nan)
    fj = np.interp(plon, lons, np.arange(lons.size), left=np.nan, right=np.nan)
    i0 = np.floor(fi).astype(int)
    j0 = np.floor(fj).astype(int)
    ok = np.isfinite(fi) & np.isfinite(fj)
    i0 = np.clip(i0, 0, lats.size - 2)
    j0 = np.clip(j0, 0, lons.size - 2)
    ti = np.where(ok, fi - i0, 0.0)
    tj = np.where(ok, fj - j0, 0.0)
    out = (arr[i0, j0] * (1 - ti) * (1 - tj) + arr[i0, j0 + 1] * (1 - ti) * tj +
           arr[i0 + 1, j0] * ti * (1 - tj) + arr[i0 + 1, j0 + 1] * ti * tj)
    return np.where(ok, out, np.nan)


def _divergence2d(u, v, lats, lons):
    phi = np.deg2rad(np.asarray(lats, float))
    lam = np.deg2rad(np.asarray(lons, float))
    cos = np.cos(phi)[:, None]
    du_dlam = np.gradient(u, lam, axis=1)
    dvc_dphi = np.gradient(v * cos, phi, axis=0)
    return (du_dlam + dvc_dphi) / (EARTH_R * cos)


def _vorticity2d(u, v, lats, lons):
    phi = np.deg2rad(np.asarray(lats, float))
    lam = np.deg2rad(np.asarray(lons, float))
    cos = np.cos(phi)[:, None]
    dv_dlam = np.gradient(v, lam, axis=1)
    duc_dphi = np.gradient(u * cos, phi, axis=0)
    return (dv_dlam - duc_dphi) / (EARTH_R * cos)


def _find_centre(u850, v850, lats, lons, lat0, lon0, box_deg=2.5, step_deg=0.25):
    """850 hPa cyclonic vorticity maximum, nominal সেন্টারের ±box_deg-এর মধ্যে।"""
    zeta = _vorticity2d(u850, v850, lats, lons)
    sign = 1.0 if lat0 >= 0 else -1.0
    score = sign * zeta
    la = np.arange(lat0 - box_deg, lat0 + box_deg + step_deg / 2, step_deg)
    lo = np.arange(lon0 - box_deg, lon0 + box_deg + step_deg / 2, step_deg)
    LA, LO = np.meshgrid(la, lo, indexing='ij')
    vals = _bilinear(lats, lons, score, LA, LO)
    if not np.isfinite(vals).any():
        return float(lat0), float(lon0), False
    i, j = np.unravel_index(np.nanargmax(vals), vals.shape)
    edge = i in (0, vals.shape[0] - 1) or j in (0, vals.shape[1] - 1)
    return float(LA[i, j]), float(LO[i, j]), bool(edge)


def _omega_from_divergence(u3d, v3d, lev_top_first, lats, lons):
    """ω (Pa/s): ∂ω/∂p = −∇·V, উপরে (সর্বনিম্ন p) ω=0 ধরে নিচে ইন্টিগ্রেট।"""
    divs = np.stack([_divergence2d(u3d[k], v3d[k], lats, lons)
                     for k in range(u3d.shape[0])])
    p = np.asarray(lev_top_first, float) * 100.0
    omega = np.zeros_like(divs)
    for k in range(1, len(p)):
        omega[k] = omega[k - 1] - 0.5 * (divs[k] + divs[k - 1]) * (p[k] - p[k - 1])
    return omega


def _polar_latlon(lat_c, lon_c, rmax_km=1000.0, dr_km=25.0, dth_deg=10.0):
    r = np.arange(0.0, rmax_km + dr_km / 2, dr_km)
    th = np.deg2rad(np.arange(0.0, 360.0, dth_deg))
    R, TH = np.meshgrid(r, th, indexing='ij')
    x, y = R * np.sin(TH), R * np.cos(TH)                    # east, north (km)
    lat = lat_c + y * 1000.0 / EARTH_R * (180.0 / np.pi)
    lon = lon_c + x * 1000.0 / (EARTH_R * np.cos(np.deg2rad(lat_c))) * (180.0 / np.pi)
    return R, TH, lat, lon


def _fetch_section_fields(run, step):
    """secondary circulation-এর জন্য u/v/t সব SECTION_LEVELS-এ।"""
    g = _fetch_levels(run, step, ['u', 'v', 't'], SECTION_LEVELS)
    if not g:
        return None
    lev = sorted(SECTION_LEVELS)                            # ascending p: 200 → 1000
    f0 = g[('u', lev[0])]
    u3d = np.stack([g[('u', p)].values for p in lev])
    v3d = np.stack([g[('v', p)].values for p in lev])
    T3d = np.stack([g[('t', p)].values for p in lev])
    return lev, f0.lats, f0.lons, u3d, v3d, T3d


def compute_secondary(run, step, centre):
    """আজিমুথাল-গড় radial/tangential wind ও w — radius × pressure গ্রিডে।"""
    got = _fetch_section_fields(run, step)
    if got is None:
        return None
    lev, lats, lons, u3d, v3d, T3d = got
    k850 = lev.index(850)
    lat_c, lon_c, edge = _find_centre(u3d[k850], v3d[k850], lats, lons, *centre)
    omega = _omega_from_divergence(u3d, v3d, lev, lats, lons)
    R, TH, pla, plo = _polar_latlon(lat_c, lon_c)
    sinT, cosT = np.sin(TH), np.cos(TH)
    lev_arr = np.asarray(lev, float)
    ur = np.empty((len(lev),) + R.shape)
    vt = np.empty_like(ur)
    w = np.empty_like(ur)
    for k in range(len(lev)):
        us = _bilinear(lats, lons, u3d[k], pla, plo)
        vs = _bilinear(lats, lons, v3d[k], pla, plo)
        oms = _bilinear(lats, lons, omega[k], pla, plo)
        Ts = _bilinear(lats, lons, T3d[k], pla, plo)
        ur[k] = us * sinT + vs * cosT                       # radial (বাইরের দিকে +)
        vt[k] = -us * cosT + vs * sinT                      # tangential (cyclonic +)
        w[k] = -oms * R_D * Ts / (lev_arr[k] * 100.0 * GRAV)
    with np.errstate(all='ignore'):
        out = {
            'centre': (lat_c, lon_c), 'at_edge': edge, 'lev': lev_arr,
            'r_km': R[:, 0],
            'ur': np.nanmean(ur, axis=2), 'vt': np.nanmean(vt, axis=2),
            'w': np.nanmean(w, axis=2),
        }
    return out


def compute_outflow(run, step, centre):
    """200 hPa divergence ম্যাপ + azimuthal-mean radial outflow প্রোফাইল।"""
    g = _fetch_levels(run, step, ['u', 'v'], [850, 200])
    if not g:
        return None
    f200 = g[('u', 200)]
    lats, lons = f200.lats, f200.lons
    lat_c, lon_c, edge = _find_centre(g[('u', 850)].values, g[('v', 850)].values,
                                      lats, lons, *centre)
    u200, v200 = f200.values, g[('v', 200)].values
    div200 = _divergence2d(u200, v200, lats, lons) * 1e5    # 1e-5 s-1
    gh = get_field(run, step, [('gh', 200)])
    gh200 = gh[('gh', 200)].values if gh else None
    R, TH, pla, plo = _polar_latlon(lat_c, lon_c)
    us = _bilinear(lats, lons, u200, pla, plo)
    vs = _bilinear(lats, lons, v200, pla, plo)
    ur = us * np.sin(TH) + vs * np.cos(TH)
    with np.errstate(all='ignore'):
        profile = np.nanmean(ur, axis=1)                    # (nr,)
    kmax = int(np.nanargmax(profile)) if np.isfinite(profile).any() else 0
    return {'centre': (lat_c, lon_c), 'at_edge': edge, 'lats': lats, 'lons': lons,
            'div200': div200, 'u200': u200, 'v200': v200, 'gh200': gh200,
            'r_km': R[:, 0], 'ur_profile': profile,
            'ur_max': float(profile[kmax]), 'ur_max_r': float(R[kmax, 0])}


def _section_figure(th, title, subtitle, unit_tag):
    """ম্যাপ-হেডারের একই স্টাইলে radius–height সেকশনের ফিগার।"""
    W, H = 10.6, 8.0
    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=th['fig'])
    L, R = 0.62 / W, 1.0 - 0.45 / W
    fig.add_artist(plt.Rectangle((0.0, 1 - 0.05 / H), 1.0, 0.05 / H, transform=fig.transFigure,
                                 facecolor=th['accent'], edgecolor='none', clip_on=False))
    fig.text(L, 1 - 0.46 / H, title, fontsize=14, weight='bold', color=th['text'], va='center')
    fig.text(L, 1 - 0.86 / H, subtitle, fontsize=8.8, color=th['sub'], va='center')
    fig.text(R, 1 - 0.46 / H, 'ECMWF · IFS', fontsize=9.5, weight='bold',
             color=th['badge_fg'], ha='right', va='center', clip_on=False,
             bbox=dict(boxstyle='round,pad=0.38', facecolor=th['accent'], edgecolor='none'))
    if unit_tag:
        fig.text(R, 1 - 0.86 / H, unit_tag, fontsize=8.5, color=th['sub'], ha='right', va='center')
    ax = fig.add_axes([L, 1.55 / H, (R - L), 1 - 2.85 / H])
    cax = fig.add_axes([L + 0.14 * (R - L), 0.72 / H, 0.72 * (R - L), 0.15 / H])
    return fig, ax, cax


def _valid_txt(kind, rdt, vdt, step):
    return f'Valid {fmt_short(vdt)} (+{step} h)'


def render_secondary(run, step, centre, th):
    data = compute_secondary(run, step, centre)
    if data is None:
        raise RuntimeError('ECMWF pressure-level u/v/t fields could not be fetched for the secondary circulation.')
    lat_c, lon_c = data['centre']
    rdt = run_datetime(*run)
    vdt = rdt + dt.timedelta(hours=step)
    subtitle = (f'IFS HRES 0.25°   ·   Run {fmt_short(rdt)}   ·   {_valid_txt("secondary", rdt, vdt, step)}'
                f'   ·   centre {lat_c:.2f}°N {lon_c:.2f}°E (850 hPa vorticity max)')
    fig, ax, cax = _section_figure(
        th, 'Azimuthal-Mean Secondary Circulation', subtitle, '[w: m/s]')
    r, lev, w, vt, ur = data['r_km'], data['lev'], data['w'], data['vt'], data['ur']
    z = np.log(1000.0 / lev)
    order = np.argsort(z)
    z_s, w_s, vt_s, ur_s = z[order], w[order], vt[order], ur[order]
    lim = max(float(np.nanmax(np.abs(w_s))), 0.01)
    levels = np.linspace(-lim, lim, 21)
    im = ax.contourf(r, z_s, w_s, levels=levels, cmap='RdBu_r', extend='both')
    cs = ax.contour(r, z_s, vt_s, levels=np.arange(-10, 45, 5), colors='#111111',
                    linewidths=0.7)
    ax.clabel(cs, fmt='%d', fontsize=7)
    # overturning arrows: radial (x) ও w (y) আলাদা স্কেলে (আকৃতি বোঝার জন্য)
    sx = max(1, r.size // 14)
    sz = max(1, z_s.size // 9)
    X, Z = np.meshgrid(r[::sx], z_s[::sz])
    kx = 0.055 * (r[-1] - r[0]) / max(float(np.nanmax(np.abs(ur_s))), 0.1)
    kz = 0.055 * (z_s[-1] - z_s[0]) / lim
    ax.quiver(X, Z, ur_s[::sz, ::sx] * kx, w_s[::sz, ::sx] * kz,
              angles='xy', scale_units='xy', scale=1.0, width=0.0032,
              headwidth=4.0, headlength=4.4, color='#111111', zorder=7)
    ticks = [t for t in (1000, 850, 700, 500, 300, 200) if lev.min() <= t <= lev.max()]
    ax.set_yticks([np.log(1000.0 / t) for t in ticks])
    ax.set_yticklabels([str(t) for t in ticks], fontsize=8, color=th['sub'])
    ax.set_xlabel('Radius from centre (km)', fontsize=9, color=th['text'])
    ax.set_ylabel('Pressure (hPa)', fontsize=9, color=th['text'])
    ax.tick_params(labelsize=8, colors=th['sub'], length=0)
    for s in ax.spines.values():
        s.set_color(th['sub'])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_label('vertical velocity w (m/s)  — red = ascent', fontsize=9, color=th['text'])
    cb.ax.tick_params(labelsize=7.5, color=th['sub'], length=2.5)
    cb.outline.set_visible(False)
    fig.text(0.985, 0.012, '@XPWEATHER', transform=fig.transFigure, fontsize=11,
             weight='bold', color='#0d1217', alpha=0.6, ha='right', va='bottom')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue(), data['centre']


def render_outflow(run, step, centre, th):
    data = compute_outflow(run, step, centre)
    if data is None:
        raise RuntimeError('ECMWF 850/200 hPa wind fields could not be fetched for the outflow map.')
    lat_c, lon_c = data['centre']
    rdt = run_datetime(*run)
    vdt = rdt + dt.timedelta(hours=step)
    bbox = (max(-80.0, lat_c - 10.0), min(80.0, lat_c + 10.0),
            max(-180.0, lon_c - 12.0), min(180.0, lon_c + 12.0))
    subtitle = (f'IFS HRES 0.25°   ·   Run {fmt_short(rdt)}   ·   {_valid_txt("outflow", rdt, vdt, step)}'
                f'   ·   centre {lat_c:.2f}°N {lon_c:.2f}°E')
    fig, ax, cax = make_figure(bbox, th, 'Outflow Strength (200 hPa)', subtitle, '[10⁻⁵ s⁻¹]')
    lats_c, lons_c, div_c, extra_c = crop_to_bbox(
        data['lats'], data['lons'], data['div200'], bbox,
        extra=(data['u200'], data['v200']) + ((data['gh200'],) if data['gh200'] is not None else ()))
    dlat = abs(lats_c[0] - lats_c[1])
    dlon = abs(lons_c[0] - lons_c[1])
    extent = [lons_c[0] - dlon / 2, lons_c[-1] + dlon / 2, lats_c[-1] - dlat / 2, lats_c[0] + dlat / 2]
    div_levels = np.arange(-10, 10.1, 1.0).tolist()
    cmap = plt.get_cmap('RdBu_r')
    norm = mcolors.BoundaryNorm(div_levels, ncolors=cmap.N, clip=False)
    im = ax.imshow(div_c, extent=extent, origin='upper', cmap=cmap, norm=norm,
                   interpolation='bilinear', zorder=2)
    style_axes(ax, bbox, th, None)
    if data['gh200'] is not None:
        cs = ax.contour(lons_c, lats_c, extra_c[2], levels=np.arange(11000, 12600, 60),
                        colors='#14202b', linewidths=0.55, alpha=0.8, zorder=6)
        ax.clabel(cs, fmt='%d', fontsize=6.5, inline=True)
    o = np.argsort(lats_c)
    ax.streamplot(lons_c, lats_c[o], extra_c[0][o], extra_c[1][o], color='#14202b',
                  density=1.3, linewidth=0.65, arrowsize=0.75, zorder=7)
    draw_baseline(ax, bbox, th, 1.2)
    ax.plot(lon_c, lat_c, '+', ms=13, mew=2.2, color='#111111', zorder=9)
    ax.text(lon_c + 0.3, lat_c + 0.3, 'centre', fontsize=8, weight='bold', color='#111111',
            zorder=9).set_path_effects([patheffects.withStroke(linewidth=2.2, foreground='white', alpha=0.85)])
    # প্রোফাইল ইনসেট: azimuthal-mean radial outflow (200 hPa)
    inset = ax.inset_axes([0.585, 0.045, 0.395, 0.33])
    inset.set_facecolor((1, 1, 1, 0.92))
    inset.plot(data['r_km'], data['ur_profile'], color='#b2182b', lw=1.6)
    inset.axhline(0, color='#666666', lw=0.7)
    inset.fill_between(data['r_km'], data['ur_profile'], 0,
                       where=(data['ur_profile'] > 0), color='#b2182b', alpha=0.25)
    inset.set_xlabel('radius (km)', fontsize=6.5)
    inset.set_ylabel('radial outflow (m/s)', fontsize=6.5)
    inset.tick_params(labelsize=6.5, length=2)
    inset.set_title(f"max {data['ur_max']:.1f} m/s at {data['ur_max_r']:.0f} km (200 hPa)",
                    fontsize=7)
    for s in inset.spines.values():
        s.set_color('#888888')
    ax.text(0.985, 0.008, '@XPWEATHER', transform=ax.transAxes, fontsize=13,
            weight='bold', color='#0d1217', alpha=0.6, ha='right', va='bottom',
            zorder=30).set_path_effects([patheffects.withStroke(linewidth=2.6, foreground='white', alpha=0.75)])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', extend='both',
                      ticks=div_levels[::2])
    cb.set_label('200 hPa divergence (10⁻⁵ s⁻¹) — red = outflow', fontsize=9, color=th['text'])
    cb.ax.tick_params(labelsize=7.5, color=th['sub'], length=2.5)
    cb.outline.set_visible(False)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue(), data['centre']


# ============================ ওয়েব এন্ট্রি পয়েন্ট ============================

def cleanup_old_runs(current_run, keep=2):
    """CACHE_DIR-এর পুরনো রান-ফোল্ডার ডিলিট — সর্বশেষ ২টি রানের ডেটাই থাকে"""
    import shutil
    import glob
    try:
        dirs = [d for d in os.listdir(CACHE_DIR)
                if os.path.isdir(os.path.join(CACHE_DIR, d)) and
                len(d) == 12 and d[:8].isdigit() and d.endswith('z')]
    except Exception:
        return
    dirs.sort(reverse=True)
    keep_names = {f'{current_run[0]}_{current_run[1]}z'} | set(dirs[:keep])
    for d in dirs:
        if d not in keep_names:
            try:
                shutil.rmtree(os.path.join(CACHE_DIR, d), ignore_errors=True)
            except Exception:
                pass
    # পুরনো (ফ্ল্যাট) ফর্ম্যাটের ভাসমান ফাইলও সরাও
    for f in glob.glob(os.path.join(CACHE_DIR, '*.grb')) + glob.glob(os.path.join(CACHE_DIR, '*.jsonl')):
        try:
            os.remove(f)
        except Exception:
            pass


def render_map(map_key, run, step, region_key='bangladesh', centre=None):
    """একটা ECMWF ম্যাপ রেন্ডার করে (PNG bytes, meta dict) রিটার্ন করে।

    run: (YYYYMMDD, HH) — resolve করা রান; step: ফোরকাস্ট আওয়ার।
    centre: (lat, lon) — CENTRE_MAPS-এর জন্য আবশ্যক (নমিনাল সেন্টার)।
    ডেটা/স্টেপ না থাকলে ValueError/RuntimeError ওঠে (ওয়েব লেয়ার ধরে)।"""
    if map_key not in MAP_TITLES:
        raise ValueError(f"unknown ECMWF map '{map_key}'")
    if region_key not in REGION_PRESETS:
        raise ValueError(f"unknown ECMWF region '{region_key}'")
    step = int(step)
    if map_key == 'tp24' and step < 24:
        raise ValueError('24-hour Precipitation (tp24) needs a forecast hour of at least 24.')
    if load_index(run, step) is None:
        raise ValueError(
            f'ECMWF forecast step +{step}h is not available yet for run '
            f'{run[0]} {run[1]}Z — try Auto Date (latest run) or a smaller forecast hour.')
    if map_key == 'tp24' and load_index(run, step - 24) is None:
        raise ValueError(f'ECMWF step +{step - 24}h (needed for tp24) is not available for this run.')

    cleanup_old_runs(run)
    th = THEMES.get(THEME, THEMES['light'])
    centre_found = None
    if map_key in CENTRE_MAPS:
        if centre is None:
            raise ValueError('this map needs a centre latitude and longitude')
        if map_key == 'secondary':
            png, centre_found = render_secondary(run, step, centre, th)
        else:
            png, centre_found = render_outflow(run, step, centre, th)
    else:
        data3 = prepare_map_data(run, map_key, step)
        if data3 is None:
            raise RuntimeError(
                f'ECMWF data for {map_key} at +{step}h could not be fetched for run '
                f'{run[0]} {run[1]}Z (parameter missing in this run/step, or download failed).')
        bbox = REGION_PRESETS[region_key]['bbox']
        png = plot_single(map_key, step, data3, run, region_key, bbox, th)

    rdt = run_datetime(*run)
    vdt = rdt + dt.timedelta(hours=step)
    title0, unit = MAP_TITLES[map_key]
    meta = {
        'map': map_key,
        'name': title0,
        'unit': unit,
        'run': f'{rdt:%Y-%m-%d %H}Z',
        'run_compact': f'{run[0]}{run[1]}z',
        'valid': fmt_time(vdt),
        'step': step,
        'region': REGION_LABELS.get(region_key, region_key),
        'region_key': region_key,
        'bytes_downloaded': round(HTTP.n_bytes / 1e6, 2),
        'requests': HTTP.n_requests,
    }
    if centre_found is not None:
        meta['centre'] = f'{centre_found[0]:.2f}°N {centre_found[1]:.2f}°E'
    return png, meta
