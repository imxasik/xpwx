import os, io, re, json, time, zipfile, datetime, warnings

warnings.filterwarnings("ignore")

import numpy as np
from scipy.ndimage import gaussian_filter1d
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shapefile
import requests
import urllib.request, urllib.error
from pydap.client import open_url


# =========================================================
#  কনফিগারেশন
# =========================================================
AUTO        = True                              # True = সর্বশেষ AIFS run খুঁজে নেবে
MANUAL_BASE = datetime.datetime(2026, 9, 12, 0)  # AUTO=False হলে এই run টা ব্যবহার হবে (UTC)

N_DAYS      = 1          # কত দিনের forecast গড় করা হবে
                         # LEAD_HOURS=None থাকলে: day+1 … day+N_DAYS (পুরনো নিয়ম)
                         # LEAD_HOURS=L থাকলে: L-এ শেষ হওয়া N_DAYS দিনের গড়
                         #   যেমন N_DAYS=3, LEAD_HOURS=240 → +192/+216/+240 h
                         # N_DAYS=1 → ওই একটি lead এর sample (title-এ শুধু date)
                         # N_DAYS=0/None → LEAD_HOURS এর snapshot (6/12/18… চলবে)
LEAD_HOURS  = 240       # None = N_DAYS অনুযায়ী 24,48,72… h
                         # অথবা নিজের পছন্দের ঘণ্টা:  240  /  [240]  /  [90]  /  [144]
                         # বা explicit গড়:  [96, 102, 108, 114]
                         # (৬ এর গুণিতক হতে হবে, 0 … 360)
BASE_HOURS  = (0, 6, 12, 18)   # কোন base run গুলো খোঁজা হবে (unpublished চুপচাপ skip)
LEVEL_HPA   = 200        # pressure level

WORK_STRIDE = 2          # 1 = 0.25° (full, ধীর), 2 = 0.5°, 4 = 1.0°  — plot/compute grid
SMOOTH_ANOM_DEG  = 3.75  # wind anomaly gaussian sigma (ডিগ্রিতে)  [পুরনো স্ক্রিপ্টে 1.5 × 2.5°]
SMOOTH_CHI_DEG   = 5.00  # velocity potential sigma (ডিগ্রিতে)    [পুরনো স্ক্রিপ্টে 2.0 × 2.5°]
ARROW_SPACING_DEG = 7.5  # তীর চিহ্নের ফাঁকা (ডিগ্রি)
VLIM        = 10.0       # χ colour scale সীমা (×10⁶ m²/s)

SHP_DIR     = "map"
SHP_PATH    = os.path.join(SHP_DIR, "ne_110m_coastline.shp")
SHP_URL     = "https://naciscdn.org/naturalearth/110m/physical/ne_110m_coastline.zip"

CACHE_DIR   = "aifs_cache"   # ডাউনলোড করা GRIB message ক্যাশ (খালি string "" = ক্যাশ বন্ধ)
CACHE_KEEP_DAYS = 3          # এর চেয়ে পুরনো ক্যাশ GRIB মুছে যাবে (দিন)
OUT_FILE    = "vp.png"

# ---- ECMWF Open Data ----
EC_ROOT   = "https://data.ecmwf.int/forecasts"
EC_MODEL  = "aifs-single"    # ECMWF এর operational AI model
EC_RES    = "0p25"           # 0.25° regular lat-lon
EC_STREAM = "oper"

# ---- NOAA PSL (climatology) ----
PSL       = "https://psl.noaa.gov/thredds/dodsC/Datasets/ncep"
LTM_FILE  = "day.ltm.1991-2020.nc"

TIMEOUT   = 180
RETRIES   = 4


# =========================================================
#  সাধারণ HTTP helper (retry সহ)
# =========================================================
def http_get(url, headers=None, timeout=TIMEOUT, tries=RETRIES):
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (404, 410, 416):        # ফাইল/ফোল্ডার নেই → retry অর্থহীন
                raise
            last = e
            wait = 2.0 * (k + 1)
            print(f"    ! retry {k+1}/{tries} in {wait:.0f}s  ({type(e).__name__}: {e})")
            time.sleep(wait)
        except Exception as e:                   # noqa
            last = e
            wait = 2.0 * (k + 1)
            print(f"    ! retry {k+1}/{tries} in {wait:.0f}s  ({type(e).__name__}: {e})")
            time.sleep(wait)
    raise RuntimeError(f"HTTP failed: {url}\n{last}")


# =========================================================
#  AEC / CCSDS ডিকোডার  (GRIB2 template 5.42)
#  libaec 1.1.3 এর সাথে হুবহু মিলিয়ে যাচাই করা
# =========================================================
AEC_DATA_SIGNED    = 1
AEC_DATA_3BYTE     = 2
AEC_DATA_MSB       = 4
AEC_DATA_PREPROCESS = 8
AEC_RESTRICTED     = 16
AEC_PAD_RSI        = 32

_ROS = 5
_SE_TABLE_SIZE = 90


def _se_tables():
    t = np.zeros(2 * (_SE_TABLE_SIZE + 1), dtype=np.int64)
    k = 0
    for i in range(13):
        ms = k
        for _ in range(i + 1):
            t[2 * k] = i
            t[2 * k + 1] = ms
            k += 1
    return t[0::2].copy(), t[1::2].copy()


_SE_I, _SE_MS = _se_tables()


class _BitReader:
    """MSB-first bit reader"""
    __slots__ = ("data", "bits", "pos", "nbits")

    def __init__(self, data):
        self.data = data + b"\x00" * 16
        self.bits = np.unpackbits(np.frombuffer(self.data, dtype=np.uint8))
        self.pos = 0
        self.nbits = len(self.bits)

    def get_bits(self, n):
        if n == 0:
            return 0
        p = self.pos
        bo = p & 7
        bs = p >> 3
        nb = (bo + n + 7) >> 3
        v = int.from_bytes(self.data[bs:bs + nb], "big")
        self.pos = p + n
        return (v >> (nb * 8 - bo - n)) & ((1 << n) - 1)

    def get_array(self, m, k):
        """m টা k-bit সংখ্যা একসাথে পড়া"""
        p = self.pos
        total = m * k
        bo = p & 7
        bs = p >> 3
        nb = (bo + total + 7) >> 3
        b = np.unpackbits(np.frombuffer(self.data[bs:bs + nb], dtype=np.uint8))
        b = b[bo:bo + total].reshape(m, k).astype(np.int64)
        self.pos = p + total
        return b @ (np.int64(1) << np.arange(k - 1, -1, -1, dtype=np.int64))

    def get_fs(self, m):
        """m টা fundamental sequence (CCSDS)"""
        bits = self.bits
        p = self.pos
        w = 256 if m <= 32 else 4 * m
        while True:
            hi = min(p + w, self.nbits)
            o = np.flatnonzero(bits[p:hi])
            if o.size >= m or hi == self.nbits:
                break
            w *= 4
        if o.size < m:
            raise ValueError("AEC: unexpected end of input")
        o = o[:m]
        fs = np.empty(m, dtype=np.int64)
        fs[0] = o[0]
        if m > 1:
            fs[1:] = o[1:] - o[:-1] - 1
        self.pos = int(p + o[m - 1] + 1)
        return fs


def _pp_rsi(raw, xmin, xmax, signed, sign_bit):
    """CCSDS reference-sample predictor (inverse)"""
    L = len(raw)
    out = np.empty(L, dtype=np.int64)
    x0 = int(raw[0])
    if signed:
        x0 = (x0 ^ sign_bit) - sign_bit
    out[0] = x0
    if L == 1:
        return out, x0

    d = raw[1:].astype(np.int64, copy=False)
    half = (d >> 1) + (d & 1)
    nd = d.size

    if xmin == 0:
        med = (xmax >> 1) + 1
        delta = np.where((d & 1) == 0, half, -half)
        i = 0
        x = x0
        W = 1024
        while i < nd:
            hi = min(i + W, nd)
            cs = np.cumsum(delta[i:hi])
            prev = np.empty(hi - i, dtype=np.int64)
            prev[0] = x
            if hi - i > 1:
                prev[1:] = x + cs[:-1]
            lim = np.where(prev >= med, xmax - prev, prev)
            bad = np.flatnonzero(half[i:hi] > lim)
            if bad.size == 0:
                out[i + 1:hi + 1] = x + cs
                x = int(out[hi])
                i = hi
            else:
                k = i + int(bad[0])
                out[i + 1:k + 1] = x + cs[:k - i]
                x = int(out[k])
                dk = int(d[k])
                x = xmax - dk if x >= med else dk
                out[k + 1] = x
                i = k + 1
    else:
        x = x0
        for i in range(nd):
            di = int(d[i])
            hd = (di >> 1) + (di & 1)
            step = (di >> 1) if (di & 1) == 0 else -((di >> 1) + 1)
            if x < 0:
                x = x + step if hd <= xmax + x + 1 else di - xmax - 1
            else:
                x = x + step if hd <= xmax - x else xmax - di
            out[i + 1] = x
    return out, x


def aec_decode(data, n_samples, bits_per_sample, block_size, rsi, flags):
    """AEC/CCSDS কম্প্রেসড বাফার ডিকোড"""
    if not (1 <= bits_per_sample <= 32):
        raise ValueError("bad bits_per_sample")

    pp     = bool(flags & AEC_DATA_PREPROCESS)
    signed = bool(flags & AEC_DATA_SIGNED)

    if bits_per_sample > 16:
        bytes_per_sample = 3 if (bits_per_sample <= 24 and (flags & AEC_DATA_3BYTE)) else 4
        id_len = 5
    elif bits_per_sample > 8:
        bytes_per_sample = 2
        id_len = 4
    else:
        id_len = (1 if bits_per_sample <= 2 else 2) if (flags & AEC_RESTRICTED) else 3
        bytes_per_sample = 1

    modi     = 1 << id_len
    rsi_size = rsi * block_size

    if signed:
        xmax    = (1 << (bits_per_sample - 1)) - 1
        xmin    = -(xmax + 1)
        signbit = 1 << (bits_per_sample - 1)
    else:
        xmin    = 0
        xmax    = (1 << bits_per_sample) - 1
        signbit = 0

    br  = _BitReader(data)
    raw = np.zeros(rsi_size, dtype=np.int64)
    out = np.empty(n_samples, dtype=np.int64)

    got = nraw = 0
    ref = 1 if pp else 0
    ebs = block_size - ref
    pad_rsi = bool(flags & AEC_PAD_RSI)

    def flush():
        nonlocal got, nraw, ref, ebs
        if nraw == 0:
            return
        L = min(nraw, n_samples - got)
        if pp:
            s, _ = _pp_rsi(raw[:nraw], xmin, xmax, signed, signbit)
            out[got:got + L] = s[:L]
        else:
            out[got:got + L] = raw[:L]
        got += L
        nraw = 0
        ref = 1 if pp else 0
        ebs = block_size - ref
        if pad_rsi:
            br.pos += (-br.pos) % 8

    while got < n_samples:
        if nraw >= rsi_size or nraw >= n_samples - got:
            flush()
            if got >= n_samples:
                break

        oid = br.get_bits(id_len)

        if oid == modi - 1:                                   # uncompressed
            vals = np.fromiter((br.get_bits(bits_per_sample) for _ in range(block_size)),
                               dtype=np.int64, count=block_size)
            raw[nraw:nraw + block_size] = vals
            nraw += block_size

        elif oid == 0:                                        # low entropy
            sub = br.get_bits(1)
            if ref:
                raw[nraw] = br.get_bits(bits_per_sample)
                nraw += 1
            if sub == 1:                                      # second extension
                i = ref
                while i < block_size:
                    m = int(br.get_fs(1)[0])
                    if m > _SE_TABLE_SIZE:
                        raise ValueError("AEC: bad second-extension index")
                    d1 = m - int(_SE_MS[m])
                    if (i & 1) == 0:
                        raw[nraw] = int(_SE_I[m]) - d1
                        nraw += 1
                        i += 1
                    raw[nraw] = d1
                    nraw += 1
                    i += 1
            else:                                             # run of zero blocks
                zb = int(br.get_fs(1)[0]) + 1
                if zb == _ROS:
                    b = nraw // block_size
                    zb = min(rsi - b, 64 - (b % 64))
                elif zb > _ROS:
                    zb -= 1
                zs = zb * block_size - ref
                if nraw + zs > rsi_size:
                    raise ValueError("AEC: zero run overflows RSI")
                raw[nraw:nraw + zs] = 0
                nraw += zs
            ref = 0
            ebs = block_size

        else:                                                 # k-split
            k = oid - 1
            nb = ebs
            if ref:
                raw[nraw] = br.get_bits(bits_per_sample)
                nraw += 1
            fs = br.get_fs(nb)
            vals = (fs << k) if k else fs
            if k:
                vals = vals + br.get_array(nb, k)
            raw[nraw:nraw + nb] = vals
            nraw += nb
            ref = 0
            ebs = block_size

    flush()
    return out.astype({1: np.uint8, 2: np.uint16, 4: np.uint32}[bytes_per_sample])


# =========================================================
#  ছোট GRIB2 রিডার (শুধু যা দরকার)
# =========================================================
def _sint4(bs):
    v = int.from_bytes(bs, "big")
    return -(v & 0x7FFFFFFF) if (v & 0x80000000) else v


def _sint2(bs):
    v = int.from_bytes(bs, "big")
    return -(v & 0x7FFF) if (v & 0x8000) else v


def _sections(msg):
    import struct
    if msg[:4] != b"GRIB":
        raise ValueError("GRIB magic missing")
    secs, p = {}, 16
    while p < len(msg) - 4:
        L = struct.unpack(">I", msg[p:p + 4])[0]
        n = msg[p + 4]
        if L < 5 or p + L > len(msg):
            break
        secs[n] = msg[p + 5:p + L]
        p += L
        if n == 8:
            break
    return secs


def _grid(b):
    """template 3.0 (regular lat/lon) — ecCodes Ni/Nj 4-byte ভেরিয়েন্ট সাপোর্ট"""
    npoints = int.from_bytes(b[1:5], "big")
    if int.from_bytes(b[7:9], "big") != 0:
        raise NotImplementedError("only lat/lon grid template 3.0 supported")
    wide = len(b) >= 67
    if wide:
        ni, nj = int.from_bytes(b[25:29], "big"), int.from_bytes(b[29:33], "big")
        basic, subdiv, o = b[33:37], b[37:41], 41
    else:
        ni, nj = int.from_bytes(b[25:27], "big"), int.from_bytes(b[27:29], "big")
        basic, subdiv, o = b[29:33], b[33:37], 37
    if ni * nj != npoints:
        raise ValueError(f"grid size mismatch {ni}x{nj} != {npoints}")
    basic, subdiv = int.from_bytes(basic, "big"), int.from_bytes(subdiv, "big")
    div = 1e6 if (basic == 0 or subdiv in (0, 0xFFFFFFFF)) else subdiv / float(basic)
    return dict(npoints=npoints, ni=ni, nj=nj,
                lat1=_sint4(b[o:o + 4]) / div,      lon1=_sint4(b[o + 4:o + 8]) / div,
                resflags=b[o + 8],
                lat2=_sint4(b[o + 9:o + 13]) / div, lon2=_sint4(b[o + 13:o + 17]) / div,
                di=_sint4(b[o + 17:o + 21]) / div,  dj=_sint4(b[o + 21:o + 25]) / div,
                scan=b[o + 25])


def _grid_latlon(g):
    ni, nj, scan = g["ni"], g["nj"], g["scan"]
    lon = g["lon1"] - np.arange(ni) * g["di"] if (scan & 0x80) else g["lon1"] + np.arange(ni) * g["di"]
    lat = g["lat1"] + np.arange(nj) * g["dj"] if (scan & 0x40) else g["lat1"] - np.arange(nj) * g["dj"]
    return lat, np.mod(lon, 360.0)


def _unpack(secs):
    b5 = secs[5]
    ndp = int.from_bytes(b5[0:4], "big")
    drt = int.from_bytes(b5[4:6], "big")
    R = float(np.frombuffer(bytes(b5[6:10]), dtype=">f4")[0])
    E = _sint2(b5[10:12])
    D = _sint2(b5[12:14])

    bitmap = None
    if 6 in secs and secs[6][0] != 255:
        if secs[6][0] != 0:
            raise NotImplementedError("pre-defined bitmap not supported")
        bitmap = np.unpackbits(np.frombuffer(secs[6][1:], dtype=np.uint8))[:ndp].astype(bool)

    bpv = b5[14]
    if bpv == 0:
        return np.full(ndp, R * 10.0 ** (-D)), bitmap
    if drt == 42:                                   # CCSDS / AEC  (ECMWF open data)
        raw = aec_decode(secs[7], ndp, bpv, b5[17], int.from_bytes(b5[18:20], "big"),
                         b5[16]).astype(np.int64)
    elif drt == 0:                                  # simple packing
        bits = np.unpackbits(np.frombuffer(secs[7], dtype=np.uint8))[:ndp * bpv].reshape(ndp, bpv)
        raw = bits.astype(np.int64) @ (np.int64(1) << np.arange(bpv - 1, -1, -1, dtype=np.int64))
    else:
        raise NotImplementedError(f"GRIB2 data representation template 5.{drt} not supported")
    return (raw.astype(np.float64) * (2.0 ** E) + R) * (10.0 ** (-D)), bitmap


def decode_grib(msg):
    """returns (field[nj,ni], lat, lon)"""
    secs = _sections(msg)
    g = _grid(secs[3])
    vals, bitmap = _unpack(secs)
    if bitmap is not None:
        full = np.full(g["npoints"], np.nan)
        full[bitmap] = vals
        vals = full
    a = vals.reshape(g["nj"], g["ni"])
    scan = g["scan"]
    if scan & 0x80:
        a = a[:, ::-1]
    if scan & 0x40:
        a = a[::-1, :]
    if scan & 0x20:
        raise NotImplementedError("column-major scanning not supported")
    lat, lon = _grid_latlon(g)
    return np.ascontiguousarray(a, dtype=np.float64), lat, lon, g


def roll_to_zero_lon(a, lon):
    """ECMWF 0.25° grid টা 180°E থেকে শুরু → 0°E এ সরিয়ে আনা"""
    i0 = int(np.argmin(np.abs(lon)))
    if i0:
        a = np.roll(a, -i0, axis=1)
        lon = np.mod(lon[i0] + np.arange(a.shape[1]) * (lon[1] - lon[0]), 360.0)
    return a, lon


# =========================================================
#  Coastline
# =========================================================
def ensure_coastline():
    if os.path.exists(SHP_PATH):
        print("  Coastline shapefile already present.")
        return
    os.makedirs(SHP_DIR, exist_ok=True)
    r = requests.get(SHP_URL, timeout=90); r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    for name in z.namelist():
        if os.path.splitext(name)[1] in (".shp", ".shx", ".dbf", ".prj"):
            with open(os.path.join(SHP_DIR, os.path.basename(name)), "wb") as f:
                f.write(z.read(name))


def load_coastlines():
    sf = shapefile.Reader(SHP_PATH)
    segs = []
    for shp in sf.shapes():
        pts = shp.points
        parts = list(shp.parts) + [len(pts)]
        for i in range(len(shp.parts)):
            segs.append(np.array(pts[parts[i]:parts[i + 1]]))
    return segs


# =========================================================
#  ECMWF AIFS open-data access
# =========================================================
def ec_dir_url(base):
    return f"{EC_ROOT}/{base:%Y%m%d}/{base:%H}z/{EC_MODEL}/{EC_RES}/{EC_STREAM}/"


def ec_file_url(base, step):
    return f"{ec_dir_url(base)}{base:%Y%m%d%H}0000-{step}h-{EC_STREAM}-fc.grib2"


def find_latest_run(steps):
    """সব দরকারি step যার আছে এমন সর্বশেষ AIFS base run খোঁজা"""
    now = datetime.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    cands = []
    for back in range(0, 8):
        d = (now - datetime.timedelta(days=back)).date()
        for h in sorted(BASE_HOURS, reverse=True):
            bt = datetime.datetime(d.year, d.month, d.day, h)
            if bt <= now:
                cands.append(bt)

    for bt in cands:
        durl = ec_dir_url(bt)
        try:
            lst = json.loads(http_get(durl, headers={"Accept": "application/json"}))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"    {bt:%Y%m%d %Hz} – run এখনো publish হয়নি")
                continue
            raise
        except Exception as e:                                    # noqa
            print(f"    {bt:%Y%m%d %Hz} – listing failed ({type(e).__name__})")
            continue
        names = {x["name"]: x.get("size", 0) for x in lst if not x.get("directory")}
        miss = [s for s in steps
                if f"{bt:%Y%m%d%H}0000-{s}h-{EC_STREAM}-fc.grib2" not in names
                or f"{bt:%Y%m%d%H}0000-{s}h-{EC_STREAM}-fc.index" not in names]
        if miss:
            print(f"    {bt:%Y%m%d/%Hz} – not complete yet (missing +{miss[:3]}…)")
            continue
        print(f"    ✔ using AIFS run {bt:%Y-%m-%d %Hz}  (all steps {steps} ready)")
        return bt
    raise RuntimeError("কোনো সম্পূর্ণ AIFS run পাওয়া যায়নি (৮ দিন পর্যন্ত খোঁজা হয়েছে)")


def fetch_index(url):
    txt = http_get(url[:-6] + ".index").decode("utf-8", "replace")
    return [json.loads(l) for l in txt.splitlines() if l.strip()]


def _find_rec(recs, param, level):
    for r in recs:
        if (r.get("param") == param and r.get("levtype") == "pl"
                and str(r.get("levelist", "")) == str(level)):
            return r
    raise KeyError(f"{param} @{level} hPa index-এ পাওয়া যায়নি")


def fetch_aifs_uv(base, steps):
    """AIFS forecast এর 200 hPa u,v — steps গুলোর গড়।  lon 0..360 এ ফেরত দেয়।"""
    urls = [ec_file_url(base, s) for s in steps]

    # 1) index গুলো নামানো (ছোট ফাইল)
    idxs = []
    for u in urls:
        idxs.append(fetch_index(u))

    # 2) দরকারি message টুকু byte-range দিয়ে নামানো (parallel I/O)
    jobs = []
    for step, u, recs in zip(steps, urls, idxs):
        for p in ("u", "v"):
            r = _find_rec(recs, p, LEVEL_HPA)
            jobs.append((step, p, u, r["_offset"], r["_length"]))

    def dl(job):
        step, p, url, off, length = job
        key = ""
        if CACHE_DIR:
            os.makedirs(CACHE_DIR, exist_ok=True)
            key = os.path.join(CACHE_DIR, f"{os.path.basename(url)}_{p}{LEVEL_HPA}.bin")
            if os.path.exists(key) and os.path.getsize(key) == length:
                with open(key, "rb") as f:
                    return f.read()
        data = http_get(url, headers={"Range": f"bytes={off}-{off + length - 1}"})
        if len(data) != length:                 # ফাইল তখনো লেখা হচ্ছিল → আবার চেষ্টা
            time.sleep(3)
            data = http_get(url, headers={"Range": f"bytes={off}-{off + length - 1}"})
        if len(data) != length:
            raise RuntimeError(f"incomplete GRIB message ({len(data)}/{length} bytes)")
        if key:
            try:
                with open(key, "wb") as f:
                    f.write(data)
            except Exception:
                pass
        return data

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=3) as pool:
        blobs = list(pool.map(dl, jobs))
    print(f"    downloaded {len(blobs)} fields "
          f"({sum(len(b) for b in blobs)/1e6:.1f} MB) in {time.time()-t0:.1f}s")

    # 3) ডিকোড
    us, vs, lat, lon = [], [], None, None
    for (step, p, url, off, length), blob in zip(jobs, blobs):
        t1 = time.time()
        fld, la, lo, g = decode_grib(blob)
        fld, lo = roll_to_zero_lon(fld, lo)
        if lat is None:
            lat, lon = la, lo
        (us if p == "u" else vs).append(fld)
        print(f"      +{step:>3}h {p}{LEVEL_HPA}   {np.nanmin(fld):7.1f} … "
              f"{np.nanmax(fld):7.1f} m/s   ({time.time()-t1:.1f}s)")
    return lat, lon, np.mean(us, axis=0), np.mean(vs, axis=0)


# =========================================================
#  NCEP/NCAR 1991-2020 daily climatology (anomaly-র বেসলাইন)
# =========================================================
def fetch_ltm_uv(dates):
    """দেওয়া তারিখগুলোর 2.5° daily climatology (u,v) — গড় করা"""
    def one(var):
        ds = open_url(f"{PSL}/{var}.{LTM_FILE}")
        lev = np.array(ds["level"][:])
        li = int(np.argmin(np.abs(lev - LEVEL_HPA)))
        n_t = len(np.array(ds["time"][:]))
        at = ds[var].attributes
        sf = float(at.get("scale_factor", 1.0))
        ao = float(at.get("add_offset", 0.0))
        mv = float(at.get("missing_value", -9.96921e36))
        out = []
        for d in dates:
            doy = d.timetuple().tm_yday
            ti = min(doy - 1, n_t - 1)                  # LTM time axis = day-of-year
            a = None
            if CACHE_DIR:                               # climo static (1991-2020) → cache
                os.makedirs(CACHE_DIR, exist_ok=True)
                key = os.path.join(CACHE_DIR, f"ltm_{var}_doy{doy:03d}.npy")
                if os.path.exists(key):
                    a = np.load(key)
            if a is None:                               # server-side slice, NOT full file
                a = np.array(ds[var][ti, li, :, :].data).squeeze().astype(np.float64)
                bad = np.abs(a - mv) < abs(mv) * 1e-6 if abs(mv) > 1e30 else (a == mv)
                a = a * sf + ao
                a[bad] = np.nan
                if CACHE_DIR:
                    np.save(key, a)
                print(f"      {var} DOY {doy:3d} → downloaded (~42 KB)")
            out.append(a)
        la = np.array(ds["lat"][:]).astype(np.float64)
        lo = np.array(ds["lon"][:]).astype(np.float64)
        return la, lo, np.nanmean(out, axis=0)

    with ThreadPoolExecutor(max_workers=2) as pool:
        fu, fv = pool.submit(one, "uwnd"), pool.submit(one, "vwnd")
        lat_u, lon_u, u = fu.result()
        lat_v, lon_v, v = fv.result()
    return lat_u, lon_u, u, v


def interp_to_grid(clim, clat, clon, tlat, tlon):
    """2.5° climatology → fine grid (bilinear, longitude periodic)"""
    nlat, nlon = clim.shape
    dlat = clat[1] - clat[0]
    dlon = clon[1] - clon[0]

    fj = np.clip((tlat - clat[0]) / dlat, 0, nlat - 1 - 1e-9)
    j0 = np.floor(fj).astype(int)
    wj = (fj - j0)

    fi = np.mod((tlon - clon[0]) / dlon, nlon)
    i0f = np.floor(fi)
    i0 = i0f.astype(int) % nlon
    i1 = (i0 + 1) % nlon
    wi = fi - i0f

    A = clim[np.ix_(j0, i0)]
    B = clim[np.ix_(j0, i1)]
    C = clim[np.ix_(j0 + 1, i0)]
    D = clim[np.ix_(j0 + 1, i1)]
    Wj = (1 - wj)[:, None]
    Wi = (1 - wi)[None, :]
    return (A * Wi * Wj + B * (1 - Wi) * Wj +
            C * Wi * (1 - Wj) + D * (1 - Wi) * (1 - Wj))


# =========================================================
#  dynamics
# =========================================================
def smooth2d(a, sigma):
    if sigma <= 0:
        return a
    a = gaussian_filter1d(a, sigma, axis=1, mode="wrap")      # longitude periodic
    return gaussian_filter1d(a, sigma, axis=0, mode="nearest")


def divergence(u, v, lat, lon):
    """div = 1/(R cosφ)·∂u/∂λ + 1/(R cosφ)·∂(v cosφ)/∂φ   (λ periodic)"""
    R = 6.371e6
    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    coslat = np.cos(lat_r)

    dlon = lon_r[1] - lon_r[0]
    dudx = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dlon * R * coslat[:, None])
    dvdy = np.gradient(v * coslat[:, None], lat_r, axis=0) / (R * coslat[:, None])
    return dudx + dvdy


def poisson_fft(rhs, lat, lon):
    R = 6.371e6
    lat_r = np.deg2rad(lat)
    lon_r = np.deg2rad(lon)
    dy = R * abs(lat_r[1] - lat_r[0])
    coslat = np.cos(lat_r)
    dx_mean = R * (lon_r[1] - lon_r[0]) * np.mean(np.abs(coslat))

    nlat, nlon = rhs.shape
    rhs_clean = np.nan_to_num(rhs, nan=0.0, posinf=0.0, neginf=0.0)

    taper = np.ones(nlat)
    for i, la in enumerate(lat):
        if abs(la) > 75.0:
            taper[i] = np.cos(np.deg2rad((abs(la) - 75.0) * 90.0 / 15.0)) ** 2
    taper[np.abs(lat) > 88.0] = 0.0
    rhs_clean = rhs_clean * taper[:, None]

    kx = 2.0 * np.pi * np.fft.fftfreq(nlon, d=dx_mean)
    ky = 2.0 * np.pi * np.fft.fftfreq(nlat, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    K2 = KX ** 2 + KY ** 2
    K2[0, 0] = 1.0
    F = np.fft.fft2(rhs_clean) / (-K2)
    F[0, 0] = 0.0
    return np.real(np.fft.ifft2(F))


# =========================================================
#  প্লট
# =========================================================
def draw_map(lat, lon, chi_anom, u_anom, v_anom, coast_segs, meta):
    from matplotlib.colors import LinearSegmentedColormap

    LON2D, LAT2D = np.meshgrid(lon, lat)
    chi_plot = chi_anom * 1e-6

    fig = plt.figure(figsize=(12, 7), facecolor="white")
    MAP_L, MAP_B, MAP_W, MAP_H = 0.045, 0.145, 0.910, 0.785
    ax = fig.add_axes([MAP_L, MAP_B, MAP_W, MAP_H])
    ax.set_facecolor("#f4f0e8")

    lon_min, lon_max = lon.min(), lon.max()
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(-75, 75)

    # ── 1. χ200' filled shading ──────────────────────────────────
    levels_fill = np.linspace(-VLIM, VLIM, 20)
    _cdict = {
        "red":   [(0.0, 0.08, 0.08), (0.35, 0.40, 0.40), (0.50, 0.97, 0.97),
                  (0.65, 0.92, 0.92), (1.0, 0.55, 0.55)],
        "green": [(0.0, 0.38, 0.38), (0.35, 0.72, 0.72), (0.50, 0.97, 0.97),
                  (0.65, 0.78, 0.78), (1.0, 0.30, 0.30)],
        "blue":  [(0.0, 0.45, 0.45), (0.35, 0.78, 0.78), (0.50, 0.97, 0.97),
                  (0.65, 0.52, 0.52), (1.0, 0.10, 0.10)],
    }
    chi_cmap = LinearSegmentedColormap("chi_cmap", _cdict, N=512)
    cf = ax.contourf(LON2D, LAT2D, chi_plot, levels=levels_fill, cmap=chi_cmap,
                     extend="both", zorder=1, alpha=0.88)

    # ── 2. χ200' thin contour lines ─────────────────────────────
    lev = np.arange(-VLIM, VLIM + 0.01, VLIM / 4.0)
    lev = lev[lev != 0]
    ax.contour(LON2D, LAT2D, chi_plot, levels=lev[lev > 0],
               colors="#5c3d11", linewidths=0.55, alpha=0.55, zorder=2)
    ax.contour(LON2D, LAT2D, chi_plot, levels=lev[lev < 0],
               colors="#1b4f6b", linewidths=0.55, linestyles="--", alpha=0.55, zorder=2)

    # ── 3. wind anomaly vectors ─────────────────────────────────
    step = max(1, int(round(ARROW_SPACING_DEG / abs(lon[1] - lon[0]))))
    qs = slice(None, None, step)
    Xq, Yq = LON2D[qs, qs], LAT2D[qs, qs]
    Uq, Vq = u_anom[qs, qs], v_anom[qs, qs]
    mag = np.sqrt(Uq ** 2 + Vq ** 2)
    mask = ~np.isnan(mag) & (np.abs(Yq) <= 70)
    ax.quiver(Xq[mask], Yq[mask], Uq[mask], Vq[mask], color="#111111",
              scale=50.0, scale_units="inches", width=0.0018,
              headwidth=4.5, headlength=5.5, headaxislength=4.8,
              minshaft=1.2, pivot="middle", zorder=6, alpha=0.92)

    ax.quiver(lon_max - 28, -68, 5.0, 0, color="#111111",
              scale=50.0, scale_units="inches", width=0.0018,
              headwidth=4.5, headlength=5.5, headaxislength=4.8,
              pivot="tail", zorder=9)
    ax.text(lon_max - 28, -72, "5 m/s", fontsize=8, color="#111111", ha="center", zorder=9)

    # ── 4. coastlines ───────────────────────────────────────────
    for seg in coast_segs:
        lons = np.where(seg[:, 0] < 0, seg[:, 0] + 360.0, seg[:, 0])
        lats = seg[:, 1]
        breaks = np.where(np.abs(np.diff(lons)) > 180)[0] + 1
        for part in np.split(np.column_stack([lons, lats]), breaks):
            ax.plot(part[:, 0], part[:, 1], color="#2c2c2c", lw=0.80, zorder=7)

    # ── 5. grid lines ───────────────────────────────────────────
    for x in range(int(lon_min), int(lon_max) + 1, 30):
        ax.axvline(x, color="#b0a898", lw=0.35, ls=":", zorder=0, alpha=0.7)
    for y in range(-60, 61, 20):
        ax.axhline(y, color="#b0a898", lw=0.35, ls=":", zorder=0, alpha=0.7)
    ax.axhline(0, color="#666655", lw=0.75, zorder=0, alpha=0.8)

    def xlab(v):
        if v in (0, 360): return "0°"
        if v == 180:      return "180°"
        return f"{v}°E" if v <= 180 else f"{360 - v}°W"

    def ylab(v):
        return "EQ" if v == 0 else f"{abs(v)}°{'N' if v > 0 else 'S'}"

    xticks = list(range(0, 360, 30))
    yticks = list(range(-80, 81, 20))
    ax.set_xticks(xticks)
    ax.set_xticklabels([xlab(x) for x in xticks], fontsize=9.5, color="#333322",
                       fontfamily="DejaVu Sans")
    ax.set_yticks(yticks)
    ax.set_yticklabels([ylab(y) for y in yticks], fontsize=9.5, color="#333322",
                       fontfamily="DejaVu Sans")
    ax.tick_params(axis="both", length=3.5, color="#888878", width=0.7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#999988"); spine.set_linewidth(0.8)

    # ── 6. colorbar ─────────────────────────────────────────────
    cax = fig.add_axes([0.12, 0.057, 0.760, 0.028])
    cbar = plt.colorbar(cf, cax=cax, orientation="horizontal",
                        ticks=np.arange(-VLIM, VLIM + 0.01, VLIM / 4.0))
    cbar.ax.tick_params(labelsize=8.5, colors="#222211", length=3.5, width=0.7)
    cbar.ax.set_xticklabels([f"{v:.1f}" for v in np.arange(-VLIM, VLIM + 0.01, VLIM / 4.0)],
                            fontsize=8.5, color="#222211")
    cbar.outline.set_edgecolor("#999988"); cbar.outline.set_linewidth(0.7)
    cax.text(0.5, -1.55, r"Velocity-Potential Anomaly  (1e6 m²s)",
             transform=cax.transAxes, ha="center", va="top",
             fontsize=12, color="#222211", fontstyle="italic")

    # ── 7. titles & watermark ───────────────────────────────────
    fig.text(0.50, 0.985,
             f"Velocity Potential Anomaly & Wind Anomaly  ·  "
             f"{meta['period']}  (AIFS forecast)\n",
             ha="center", va="top", fontsize=16, fontweight="bold",
             color="#111100", fontfamily="DejaVu Sans")

    ax.text(0.985, 0.016, "@XPWEATHER", transform=ax.transAxes, fontsize=11,
            va="bottom", ha="right", color="#222211", fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#ccccbb",
                      alpha=0.92, lw=0.9), zorder=10)

    ax.text(0.015, 0.016, meta["init_txt"], transform=ax.transAxes, fontsize=11,
            va="bottom", ha="left", color="#222211", fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#ccccbb",
                      alpha=0.92, lw=0.9), zorder=10)

    plt.savefig(OUT_FILE, dpi=220, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"\n✅  Map saved → {OUT_FILE}")


# =========================================================
CACHE_KEEP_DAYS = 3    # cached GRIB slices older than this are auto-deleted


def prune_cache(cache_dir, keep_days=None):
    """delete cached GRIB byte-range files (.bin, ~0.6 MB each) whose base date
    is older than keep_days days — they are the bulky part of the cache;
    the climatology .npy slices are only ~40 kB each and are kept."""
    keep = CACHE_KEEP_DAYS if keep_days is None else int(keep_days)
    if not cache_dir or not os.path.isdir(cache_dir):
        return 0.0
    today = datetime.datetime.utcnow().date()
    freed = 0
    for fn in os.listdir(cache_dir):
        if fn.endswith(".bin") and fn[:8].isdigit():
            d = datetime.datetime.strptime(fn[:8], "%Y%m%d").date()
            if (today - d).days > keep:
                freed += os.path.getsize(os.path.join(cache_dir, fn))
                os.remove(os.path.join(cache_dir, fn))
    if freed:
        print(f"  cache: pruned {freed / 1e6:.1f} MB of GRIB older than {keep} d")
    return freed / 1e6


def steps_from_config(n_days, lead_hours):
    """N_DAYS>=2 + LEAD_HOURS=L -> L-এ শেষ হওয়া N-day mean (24 h spacing)
       N_DAYS==1 + L -> single sample;  N_DAYS 0/None + L -> snapshot at L
       LEAD_HOURS=None -> forward window +24…+24*N;  list -> explicit steps;
       [L] (1-element list) = scalar L"""
    n = 0 if n_days is None else int(n_days)
    if isinstance(lead_hours, (list, tuple)) and len(lead_hours) > 1:
        steps = [int(h) for h in lead_hours]
    elif lead_hours is not None:
        L = int(lead_hours[0]) if isinstance(lead_hours, (list, tuple)) \
            else int(lead_hours)
        if n >= 2:
            steps = [L - 24 * (n - 1 - i) for i in range(n)]   # window ENDS at L
            if steps[0] < 0:          # window doesn't fit before L -> truncate
                steps = [h for h in steps if h >= 0]
                print(f"  note: window truncated to {len(steps)} day(s) "
                      f"ending at +{L} h")
            print(f"    window: {len(steps)}-day mean ending at +{L} h -> {steps}")
        else:
            steps = [L]
            if n <= 0:
                print(f"    snapshot at +{L} h -> {steps}")
            else:
                print(f"    single sample at +{L} h -> {steps}")
    else:
        if n <= 0:
            print("  note: N_DAYS 0/None and LEAD_HOURS None → snapshot at +24 h")
            return [24]
        steps = [24 * (i + 1) for i in range(n)]
    bad = [h for h in steps if h < 0 or h > 360 or h % 6]
    if bad:
        raise ValueError(f"AIFS open data ৬ ঘণ্টা পরপর, 0–360 h — ভুল lead: {bad}")
    return steps


def period_text(valid):
    """title period: '21–23 Sep 2026' | '22 Sep 2026' | '22 Sep 2026, 18Z'"""
    days = sorted({v.date() for v in valid})
    hours = sorted({v.hour for v in valid})
    if len(days) == 1:
        d = days[0]
        if hours == [0]:
            return f"{d:%-d %b %Y}"
        if len(valid) == 1:
            return f"{valid[0]:%-d %b %Y, %HZ}"
        return f"{d:%-d %b %Y}, {hours[0]:02d}–{hours[-1]:02d}Z"
    d0, d1 = days[0], days[-1]
    if d0.year != d1.year:
        return f"{d0:%-d %b %Y}–{d1:%-d %b %Y}"
    if d0.month != d1.month:
        return f"{d0:%-d %b}–{d1:%-d %b %Y}"
    return f"{d0:%-d}–{d1:%-d %b %Y}"


# =========================================================
def main():
    prune_cache(CACHE_DIR, CACHE_KEEP_DAYS)
    steps = steps_from_config(N_DAYS, LEAD_HOURS)

    print("=" * 64)
    print("  ECMWF AIFS  200 hPa  Velocity-Potential & Wind Anomaly")
    print("=" * 64)

    print("\n[0/5] Coastline …")
    ensure_coastline()
    coast_segs = load_coastlines()
    print(f"  {len(coast_segs)} segments loaded.")

    print("\n[1/5] AIFS run খোঁজা …")
    if AUTO:
        base = find_latest_run(steps)
    else:
        base = MANUAL_BASE
        print(f"    manual run {base:%Y-%m-%d %Hz}")
    valid = [base + datetime.timedelta(hours=s) for s in steps]
    d0, d1 = valid[0], valid[-1]
    period = period_text(valid)
    print(f"    valid {d0:%Y-%m-%d %HZ} … {d1:%Y-%m-%d %HZ}   steps {steps}")

    print("\n[2/5] ECMWF AIFS forecast (u,v) ডাউনলোড …")
    lat_f, lon_f, u_fc, v_fc = fetch_aifs_uv(base, steps)

    print("\n[3/5] NCEP 1991-2020 daily climatology (u,v) …")
    lat_c, lon_c, u_cl, v_cl = fetch_ltm_uv([v.date() for v in valid])
    u_cl = interp_to_grid(u_cl, lat_c, lon_c, lat_f, lon_f)
    v_cl = interp_to_grid(v_cl, lat_c, lon_c, lat_f, lon_f)

    # working grid (ফোনে দ্রুত চলে — পুরো 0.25° লাগলে WORK_STRIDE = 1)
    s = WORK_STRIDE
    if s > 1:
        sl = slice(None, None, s)
        lat, lon = lat_f[sl], lon_f[sl]
        u_fc, v_fc = u_fc[sl, sl], v_fc[sl, sl]
        u_cl, v_cl = u_cl[sl, sl], v_cl[sl, sl]
    else:
        lat, lon = lat_f, lon_f

    print(f"    working grid {lat.size}×{lon.size}  "
          f"({abs(lat[1]-lat[0]):.2f}°)")

    print("\n[4/5] anomaly → divergence → Poisson (χ) …")
    ddeg = abs(lon[1] - lon[0])
    u_anom = smooth2d(u_fc - u_cl, SMOOTH_ANOM_DEG / ddeg)
    v_anom = smooth2d(v_fc - v_cl, SMOOTH_ANOM_DEG / ddeg)
    print(f"    wind anom  u {np.nanmin(u_anom):6.1f} … {np.nanmax(u_anom):6.1f}   "
          f"v {np.nanmin(v_anom):6.1f} … {np.nanmax(v_anom):6.1f} m/s")

    div = divergence(u_anom, v_anom, lat, lon)
    chi = smooth2d(poisson_fft(div, lat, lon), SMOOTH_CHI_DEG / ddeg)
    print(f"    chi range: {np.nanmin(chi)*1e-6:.2f} … {np.nanmax(chi)*1e-6:.2f}  ×10⁶ m²/s")

    print("\n[5/5] Drawing map …")
    draw_map(lat, lon, chi, u_anom, v_anom, coast_segs,
             dict(period=period,
                  init_txt=f"Run: {base:%HZ} • {base:%-d %b %Y}"))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n⏱  total {(time.time()-t0)/60:.1f} min")
