# -*- coding: utf-8 -*-
"""
================================================================================
 aifs_core.py — shared engine for the AIFS anomaly product family
 (vp / Z / U / V / vertical wind shear)

 ECMWF AIFS-Single open data (no API key) + NCEP 1991-2020 daily climatology.
 GRIB2 CCSDS/AEC ডিকোডিং pure-python (numpy) — Pydroid 3 এ চলে।
================================================================================
"""
import os, io, re, json, time, zipfile, datetime, warnings

warnings.filterwarnings("ignore")

import numpy as np
from scipy.ndimage import gaussian_filter1d
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import shapefile
import requests
import urllib.request, urllib.error

# ---- ECMWF Open Data ----
EC_ROOT   = "https://data.ecmwf.int/forecasts"
EC_MODEL  = "aifs-single"
EC_RES    = "0p25"
EC_STREAM = "oper"

# ---- NOAA PSL climatology — direct HTTP file download (no pydap/OPeNDAP) ----
# fileServer URL returns the raw NetCDF binary; far more reliable than thredds OPeNDAP.
PSL_FILE = "https://psl.noaa.gov/thredds/fileServer/Datasets/ncep/{var}.{ltm}"
LTM_FILE = "day.ltm.1991-2020.nc"

SHP_URL = "https://naciscdn.org/naturalearth/110m/physical/ne_110m_coastline.zip"

G_STD   = 9.80665          # m/s²  (geopotential → height)
TIMEOUT = 180
RETRIES = 6   # increased from 4 — ECMWF open-data CDN can return 502/503 transiently


# =========================================================
#  HTTP
# =========================================================
def http_get(url, headers=None, timeout=TIMEOUT, tries=RETRIES):
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (404, 410, 416):
                raise
            # 502/503/504 are transient proxy/upstream errors — always retry
            last = e
            wait = 3.0 * (k + 1) if e.code in (502, 503, 504) else 2.0 * (k + 1)
            print(f"    ! retry {k+1}/{tries} in {wait:.0f}s  ({type(e).__name__}: {e})")
            time.sleep(wait)
        except Exception as e:                       # noqa
            last = e
            wait = 2.0 * (k + 1)
            print(f"    ! retry {k+1}/{tries} in {wait:.0f}s  ({type(e).__name__}: {e})")
            time.sleep(wait)
    raise RuntimeError(f"HTTP failed: {url}\n{last}")


# =========================================================
#  AEC / CCSDS ডিকোডার (GRIB2 template 5.42) — libaec-verified
# =========================================================
AEC_DATA_SIGNED     = 1
AEC_DATA_3BYTE      = 2
AEC_DATA_MSB        = 4
AEC_DATA_PREPROCESS = 8
AEC_RESTRICTED      = 16
AEC_PAD_RSI         = 32

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

        if oid == modi - 1:
            vals = np.fromiter((br.get_bits(bits_per_sample) for _ in range(block_size)),
                               dtype=np.int64, count=block_size)
            raw[nraw:nraw + block_size] = vals
            nraw += block_size

        elif oid == 0:
            sub = br.get_bits(1)
            if ref:
                raw[nraw] = br.get_bits(bits_per_sample)
                nraw += 1
            if sub == 1:
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
            else:
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

        else:
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
#  GRIB2
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
    if drt == 42:
        raw = aec_decode(secs[7], ndp, bpv, b5[17], int.from_bytes(b5[18:20], "big"),
                         b5[16]).astype(np.int64)
    elif drt == 0:
        bits = np.unpackbits(np.frombuffer(secs[7], dtype=np.uint8))[:ndp * bpv].reshape(ndp, bpv)
        raw = bits.astype(np.int64) @ (np.int64(1) << np.arange(bpv - 1, -1, -1, dtype=np.int64))
    else:
        raise NotImplementedError(f"GRIB2 template 5.{drt} not supported")
    return (raw.astype(np.float64) * (2.0 ** E) + R) * (10.0 ** (-D)), bitmap


def decode_grib(msg):
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
    return np.ascontiguousarray(a, dtype=np.float64), lat, lon


def roll_to_zero_lon(a, lon):
    i0 = int(np.argmin(np.abs(lon)))
    if i0:
        a = np.roll(a, -i0, axis=1)
        lon = np.mod(lon[i0] + np.arange(a.shape[1]) * (lon[1] - lon[0]), 360.0)
    return a, lon


# =========================================================
#  Coastline
# =========================================================
def ensure_coastline(shp_dir="map", shp_path=None):
    shp_path = shp_path or os.path.join(shp_dir, "ne_50m_coastline.shp")
    if os.path.exists(shp_path):
        return shp_path
    os.makedirs(shp_dir, exist_ok=True)
    r = requests.get(SHP_URL, timeout=90); r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    for name in z.namelist():
        if os.path.splitext(name)[1] in (".shp", ".shx", ".dbf", ".prj"):
            with open(os.path.join(shp_dir, os.path.basename(name)), "wb") as f:
                f.write(z.read(name))
    return shp_path


def load_coastlines(shp_path):
    sf = shapefile.Reader(shp_path)
    segs = []
    for shp in sf.shapes():
        pts = shp.points
        parts = list(shp.parts) + [len(pts)]
        for i in range(len(shp.parts)):
            segs.append(np.array(pts[parts[i]:parts[i + 1]]))
    return segs


# =========================================================
#  ECMWF AIFS open data
# =========================================================
def ec_dir_url(base):
    return f"{EC_ROOT}/{base:%Y%m%d}/{base:%H}z/{EC_MODEL}/{EC_RES}/{EC_STREAM}/"


def ec_file_url(base, step):
    return f"{ec_dir_url(base)}{base:%Y%m%d%H}0000-{step}h-{EC_STREAM}-fc.grib2"


def find_latest_run(steps, base_hours=(0,)):
    now = datetime.datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    cands = []
    for back in range(0, 8):
        d = (now - datetime.timedelta(days=back)).date()
        for h in sorted(base_hours, reverse=True):
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
        print(f"    ✔ using AIFS run {bt:%Y-%m-%d %Hz}  (steps {steps} ready)")
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


def fetch_aifs_fields(base, steps, wants, cache_dir="aifs_cache"):
    """wants = [(param, level), ...]  →  (lat, lon, {(param,level): mean field})"""
    urls = [ec_file_url(base, s) for s in steps]
    idxs = [fetch_index(u) for u in urls]

    jobs = []
    for step, u, recs in zip(steps, urls, idxs):
        for (p, lv) in wants:
            r = _find_rec(recs, p, lv)
            jobs.append((step, p, lv, u, r["_offset"], r["_length"]))

    def dl(job):
        step, p, lv, url, off, length = job
        key = ""
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            key = os.path.join(cache_dir, f"{os.path.basename(url)}_{p}{lv}.bin")
            if os.path.exists(key) and os.path.getsize(key) == length:
                with open(key, "rb") as f:
                    return f.read()
        data = http_get(url, headers={"Range": f"bytes={off}-{off + length - 1}"})
        if len(data) != length:
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

    acc, cnt, lat, lon = {}, {}, None, None
    for (step, p, lv, url, off, length), blob in zip(jobs, blobs):
        t1 = time.time()
        fld, la, lo = decode_grib(blob)
        fld, lo = roll_to_zero_lon(fld, lo)
        if lat is None:
            lat, lon = la, lo
        key = (p, lv)
        acc[key] = fld if key not in acc else acc[key] + fld
        cnt[key] = cnt.get(key, 0) + 1
        print(f"      +{step:>3}h {p}{lv}   {np.nanmin(fld):9.1f} … {np.nanmax(fld):9.1f}"
              f"   ({time.time()-t1:.1f}s)")
    return lat, lon, {k: acc[k] / cnt[k] for k in acc}


# =========================================================
#  NCEP 1991-2020 daily climatology (cached per day-of-year)
#  Uses direct HTTP download of the NetCDF file — no pydap/OPeNDAP.
#  Caches the full variable×level array (365×73×144 ≈ 3 MB) as a single
#  .npy so PSL is only hit once per var/level ever, even across restarts
#  (when mounted on a Fly volume at /data/aifs_cache).
# =========================================================

def _nc_read_var(nc_bytes, varname):
    """
    Minimal pure-numpy Classic NetCDF-3 reader.
    Returns (data_int16_or_int32, attributes_dict, dim_sizes_list).
    Only reads the one variable we ask for — no extra deps.
    """
    import struct
    buf = io.BytesIO(nc_bytes)
    magic = buf.read(4)
    if magic[:3] != b"CDF":
        raise ValueError("Not a NetCDF-3 file")
    # version byte: 1=classic, 2=64bit
    buf.read(4)  # numrecs (may be 0xFFFFFFFF for streaming)

    def read_int():
        return struct.unpack(">i", buf.read(4))[0]

    def read_str():
        n = read_int()
        s = buf.read(n)
        pad = (4 - n % 4) % 4
        buf.read(pad)
        return s.decode("utf-8", "replace")

    def read_values():
        nc_type = read_int()
        n = read_int()
        sizes = {1: 1, 2: 1, 3: 2, 4: 4, 5: 4, 6: 8}
        fmt_map = {1: "b", 2: "B", 3: ">{}h", 4: ">{}i", 5: ">{}f", 6: ">{}d"}
        sz = sizes[nc_type]
        raw = buf.read(n * sz)
        pad = (4 - (n * sz) % 4) % 4
        buf.read(pad)
        fmt = fmt_map[nc_type]
        if "{}" in fmt:
            arr = np.frombuffer(raw, dtype=fmt.format(n).replace(">{}","").replace("h","int16").replace("i","int32").replace("f","float32").replace("d","float64"))
            # use struct for small arrays
            arr = np.array(struct.unpack(fmt.format(n), raw))
        else:
            arr = np.array(struct.unpack(f">{n}{fmt}", raw))
        if n == 1:
            return arr[0]
        return arr

    def read_att_list():
        tag = read_int()
        n = read_int()
        if tag == 0 and n == 0:
            return {}
        atts = {}
        for _ in range(n):
            name = read_str()
            atts[name] = read_values()
        return atts

    def read_dim_list():
        tag = read_int()
        n = read_int()
        if tag == 0 and n == 0:
            return [], {}
        dims, dim_sizes = [], {}
        for _ in range(n):
            name = read_str()
            size = read_int()
            dims.append(name)
            dim_sizes[name] = size
        return dims, dim_sizes

    dim_names, dim_sizes = read_dim_list()
    gatts = read_att_list()
    tag = read_int()
    n_vars = read_int()

    var_meta = {}
    for _ in range(n_vars):
        vname = read_str()
        nd = read_int()
        vdims = [dim_names[read_int()] for _ in range(nd)]
        vatts = read_att_list()
        nc_type = read_int()
        vsize = read_int()
        offset_bytes = buf.read(4)  # classic = 4 bytes
        offset = struct.unpack(">I", offset_bytes)[0]
        var_meta[vname] = dict(dims=vdims, atts=vatts, nc_type=nc_type,
                               vsize=vsize, offset=offset)

    if varname not in var_meta:
        raise KeyError(f"{varname!r} not found in file; available: {list(var_meta)}")

    vm = var_meta[varname]
    nc_type_map = {3: (np.int16, 2, ">i2"),
                   4: (np.int32, 4, ">i4"),
                   5: (np.float32, 4, ">f4"),
                   6: (np.float64, 8, ">f8")}
    dtype_np, item_sz, dtype_str = nc_type_map[vm["nc_type"]]

    shape = []
    for d in vm["dims"]:
        shape.append(dim_sizes.get(d, 0))

    nc_bytes_arr = np.frombuffer(nc_bytes, dtype=np.uint8)
    off = vm["offset"]
    total = int(np.prod(shape)) if shape else 1
    raw = nc_bytes_arr[off: off + total * item_sz].tobytes()
    data = np.frombuffer(raw, dtype=dtype_str).reshape(shape)
    return data, vm["atts"], shape, vm["dims"]


# Cache of fully-downloaded + decoded LTM arrays: key → (lat, lon, full_array[365,nlat,nlon])
_LTM_MEM: dict = {}
_LTM_LOCK = __import__("threading").Lock()

def _ltm_nc_url(var):
    return PSL_FILE.format(var=var, ltm=LTM_FILE)


def _fetch_ltm_level(var, lv, cache_dir):
    """
    Download (once ever) the full LTM NetCDF for `var`, extract level `lv`,
    and return (lat, lon, array[n_days, nlat, nlon]).
    Uses a .npy sidecar next to the .nc file so the expensive download+parse
    only happens once, even across process restarts on a Fly volume.
    """
    mem_key = (var, int(lv))
    with _LTM_LOCK:
        if mem_key in _LTM_MEM:
            return _LTM_MEM[mem_key]

    # Persistent cache: one .npy per (var, level) holding all 365 days
    npy_key = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        npy_key = os.path.join(cache_dir, f"ltm_full_{var}_{int(lv)}.npy")
        meta_key = npy_key + ".meta.json"
        if os.path.exists(npy_key) and os.path.exists(meta_key):
            try:
                with open(meta_key) as mf:
                    meta = json.load(mf)
                arr = np.load(npy_key)
                lat = np.array(meta["lat"])
                lon = np.array(meta["lon"])
                result = (lat, lon, arr)
                with _LTM_LOCK:
                    _LTM_MEM[mem_key] = result
                print(f"      LTM {var} {int(lv)} hPa — loaded from disk cache")
                return result
            except Exception as e:
                print(f"      LTM cache read failed ({e}), re-downloading")

    # Download the full NetCDF file
    url = _ltm_nc_url(var)
    print(f"      LTM {var}: downloading {url} …")
    nc_bytes = http_get(url, timeout=300, tries=4)
    print(f"      LTM {var}: {len(nc_bytes)/1e6:.1f} MB downloaded, parsing…")

    try:
        data, atts, shape, dims = _nc_read_var(nc_bytes, var)
        lat_data, _, _, _ = _nc_read_var(nc_bytes, "lat")
        lon_data, _, _, _ = _nc_read_var(nc_bytes, "lon")
        lev_data, _, _, _ = _nc_read_var(nc_bytes, "level")
    except Exception:
        # Fallback: use scipy if available (handles edge cases in NC files)
        try:
            from scipy.io import netcdf_file
            with netcdf_file(io.BytesIO(nc_bytes), mmap=False) as ds:
                lat_data = ds.variables["lat"][:].copy()
                lon_data = ds.variables["lon"][:].copy()
                lev_data = ds.variables["level"][:].copy()
                v = ds.variables[var]
                atts = {k: getattr(v, k) for k in v._attributes}
                data = v[:].copy()
                dims = list(v.dimensions)
        except Exception as e2:
            raise RuntimeError(f"Could not parse LTM NetCDF for {var}: {e2}") from e2

    lat = np.array(lat_data, dtype=np.float64)
    lon = np.array(lon_data, dtype=np.float64)
    lev = np.array(lev_data, dtype=np.float64)
    li  = int(np.argmin(np.abs(lev - lv)))

    # Decode packed integers
    sf = float(atts.get("scale_factor", 1.0))
    ao = float(atts.get("add_offset",   0.0))
    mv = float(atts.get("missing_value", -32767))
    arr = np.array(data, dtype=np.float64)

    # Extract the level dimension (dims typically: time, level, lat, lon)
    if "level" in dims:
        li_ax = dims.index("level")
        arr = np.take(arr, li, axis=li_ax)   # → (time, lat, lon)

    bad = np.abs(arr - mv) < max(abs(mv) * 1e-4, 1.0)
    arr = arr * sf + ao
    arr[bad] = np.nan

    # Save to disk cache
    if npy_key:
        try:
            np.save(npy_key, arr)
            with open(meta_key, "w") as mf:
                json.dump({"lat": lat.tolist(), "lon": lon.tolist()}, mf)
            print(f"      LTM {var} {int(lv)} hPa — saved to disk cache ({arr.nbytes/1e6:.1f} MB)")
        except Exception as e:
            print(f"      LTM cache write failed: {e}")

    result = (lat, lon, arr)
    with _LTM_LOCK:
        _LTM_MEM[mem_key] = result
    return result


def fetch_ltm_fields(dates, wants, cache_dir="aifs_cache"):
    """wants = [(var, level), ...]  →  (lat, lon, {(var,level): mean climo @2.5°})"""
    def one(var, lv):
        lat, lon, full_arr = _fetch_ltm_level(var, lv, cache_dir)
        n_t = full_arr.shape[0]
        out = []
        for d in dates:
            doy = d.timetuple().tm_yday
            ti  = min(doy - 1, n_t - 1)
            out.append(full_arr[ti])
        return lat, lon, np.nanmean(out, axis=0)

    res, lat, lon = {}, None, None
    with ThreadPoolExecutor(max_workers=2) as pool:
        futs = {(v, l): pool.submit(one, v, l) for (v, l) in wants}
        for k, f in futs.items():
            la, lo, field = f.result()
            res[k] = field
            lat, lon = la, lo
    return lat, lon, res


def interp_to_grid(clim, clat, clon, tlat, tlon):
    nlat, nlon = clim.shape
    dlat = clat[1] - clat[0]
    dlon = clon[1] - clon[0]
    fj = np.clip((tlat - clat[0]) / dlat, 0, nlat - 1 - 1e-9)
    j0 = np.floor(fj).astype(int)
    wj = fj - j0
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
    return A * Wi * Wj + B * (1 - Wi) * Wj + C * Wi * (1 - Wj) + D * (1 - Wi) * (1 - Wj)


# =========================================================
#  helpers
# =========================================================
def smooth2d(a, sigma):
    if sigma <= 0:
        return a
    a = gaussian_filter1d(a, sigma, axis=1, mode="wrap")
    return gaussian_filter1d(a, sigma, axis=0, mode="nearest")


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
    """forecast steps (h) from the two config knobs:

    N_DAYS >= 2 + LEAD_HOURS = L  -> N-day mean ENDING at L (24 h spacing),
                                    e.g. N=3, L=240 -> +192/+216/+240 h
    N_DAYS == 1 + LEAD_HOURS = L  -> single sample at L
    N_DAYS 0/None + LEAD_HOURS=L  -> instantaneous snapshot at L (6/12/18… ok)
    LEAD_HOURS = None             -> forward window +24…+24*N_DAYS (old default)
    LEAD_HOURS = [a, b, …]        -> explicit step list (≥2 entries), used as
                                    given; a 1-element list [L] = scalar L
    """
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


def period_text(valid, steps=None):
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


def nice_vlim(a, q=90.0):
    """colour-scale limit: ~p90 of |anomaly|, rounded to a value whose
    contour interval (vlim/4) is a clean number"""
    v = float(np.nanpercentile(np.abs(a), q))
    for step in (0.5, 1, 2, 4, 8, 10, 12, 16, 20, 24, 40, 60, 80,
                 120, 160, 200, 240, 400, 600, 800):
        if v <= step:
            return float(step)
    return float(np.ceil(v / 200.0) * 200.0)


def anomaly_cmap():
    _cdict = {
        "red":   [(0.0, 0.08, 0.08), (0.35, 0.40, 0.40), (0.50, 0.97, 0.97),
                  (0.65, 0.92, 0.92), (1.0, 0.55, 0.55)],
        "green": [(0.0, 0.38, 0.38), (0.35, 0.72, 0.72), (0.50, 0.97, 0.97),
                  (0.65, 0.78, 0.78), (1.0, 0.30, 0.30)],
        "blue":  [(0.0, 0.45, 0.45), (0.35, 0.78, 0.78), (0.50, 0.97, 0.97),
                  (0.65, 0.52, 0.52), (1.0, 0.10, 0.10)],
    }
    return LinearSegmentedColormap("chi_cmap", _cdict, N=512)


# =========================================================
#  প্লট
# =========================================================
def draw_anomaly(lat, lon, anom, coast_segs, meta, cb_label, vlim,
                 arrows=None, arrow_ref=None, arrow_spacing_deg=7.5,
                 out_file="out.png", dpi=220):
    LON2D, LAT2D = np.meshgrid(lon, lat)

    fig = plt.figure(figsize=(12, 7), facecolor="white")
    ax = fig.add_axes([0.045, 0.145, 0.910, 0.785])
    ax.set_facecolor("#f4f0e8")

    lon_min, lon_max = lon.min(), lon.max()
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(-75, 75)

    levels_fill = np.linspace(-vlim, vlim, 20)
    cf = ax.contourf(LON2D, LAT2D, anom, levels=levels_fill, cmap=anomaly_cmap(),
                     extend="both", zorder=1, alpha=0.88)

    lev = np.arange(-vlim, vlim + 1e-9, vlim / 4.0)
    lev = lev[np.abs(lev) > 1e-9]
    ax.contour(LON2D, LAT2D, anom, levels=lev[lev > 0],
               colors="#5c3d11", linewidths=0.55, alpha=0.55, zorder=2)
    ax.contour(LON2D, LAT2D, anom, levels=lev[lev < 0],
               colors="#1b4f6b", linewidths=0.55, linestyles="--", alpha=0.55, zorder=2)

    if arrows is not None:
        uq, vq = arrows
        step = max(1, int(round(arrow_spacing_deg / abs(lon[1] - lon[0]))))
        qs = slice(None, None, step)
        Xq, Yq = LON2D[qs, qs], LAT2D[qs, qs]
        Uq, Vq = uq[qs, qs], vq[qs, qs]
        mag = np.sqrt(Uq ** 2 + Vq ** 2)
        mask = ~np.isnan(mag) & (np.abs(Yq) <= 70)
        ax.quiver(Xq[mask], Yq[mask], Uq[mask], Vq[mask], color="#111111",
                  scale=50.0, scale_units="inches", width=0.0018,
                  headwidth=4.5, headlength=5.5, headaxislength=4.8,
                  minshaft=1.2, pivot="middle", zorder=6, alpha=0.92)
        if arrow_ref:
            ax.quiver(lon_max - 28, -68, arrow_ref, 0, color="#111111",
                      scale=50.0, scale_units="inches", width=0.0018,
                      headwidth=4.5, headlength=5.5, headaxislength=4.8,
                      pivot="tail", zorder=9)
            ax.text(lon_max - 28, -72, f"{arrow_ref:g} m/s", fontsize=8,
                    color="#111111", ha="center", zorder=9)

    for seg in coast_segs:
        lons = np.where(seg[:, 0] < 0, seg[:, 0] + 360.0, seg[:, 0])
        lats = seg[:, 1]
        breaks = np.where(np.abs(np.diff(lons)) > 180)[0] + 1
        for part in np.split(np.column_stack([lons, lats]), breaks):
            ax.plot(part[:, 0], part[:, 1], color="#2c2c2c", lw=0.80, zorder=7)

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

    cax = fig.add_axes([0.12, 0.057, 0.760, 0.028])
    ticks = np.arange(-vlim, vlim + 1e-9, vlim / 4.0)
    cbar = plt.colorbar(cf, cax=cax, orientation="horizontal", ticks=ticks)
    cbar.ax.tick_params(labelsize=8.5, colors="#222211", length=3.5, width=0.7)
    intv = vlim / 4.0
    fmt = "%.0f" if abs(intv - round(intv)) < 1e-9 else "%.1f"
    cbar.ax.set_xticklabels([fmt % v for v in ticks], fontsize=8.5, color="#222211")
    cbar.outline.set_edgecolor("#999988"); cbar.outline.set_linewidth(0.7)
    cax.text(0.5, -1.55, cb_label, transform=cax.transAxes, ha="center", va="top",
             fontsize=12, color="#222211", fontstyle="italic")

    fig.text(0.50, 0.985,
             f"{meta['title']}  ·  {meta['period']}  (AIFS forecast)\n",
             ha="center", va="top", fontsize=16, fontweight="bold",
             color="#111100", fontfamily="DejaVu Sans")

    ax.text(0.985, 0.016, "@XPWEATHER", transform=ax.transAxes, fontsize=11,
            va="bottom", ha="right", color="#222211", fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#ccccbb",
                      alpha=0.92, lw=0.9), zorder=10)
    ax.text(0.015, 0.016, meta["run_txt"], transform=ax.transAxes, fontsize=11,
            va="bottom", ha="left", color="#222211", fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#ccccbb",
                      alpha=0.92, lw=0.9), zorder=10)

    plt.savefig(out_file, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"\n✅  Map saved → {out_file}")
