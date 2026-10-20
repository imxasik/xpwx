"""
shp_reader.py — Pure-Python Shapefile Reader (no pyshp/fiona needed)
Reads .shp + .dbf, returns polygon segments as numpy arrays.
"""
import struct, os
import numpy as np

def _read_dbf(path):
    """Read .dbf, return list of dicts."""
    records = []
    try:
        with open(path, "rb") as f:
            header = f.read(32)
            n_records = struct.unpack_from("<I", header, 4)[0]
            hdr_size  = struct.unpack_from("<H", header, 8)[0]
            rec_size  = struct.unpack_from("<H", header, 10)[0]
            fields = []
            f.seek(32)
            while True:
                fd = f.read(32)
                if fd[0] == 0x0D or len(fd) < 32: break
                name  = fd[:11].split(b'\x00')[0].decode("latin-1").strip()
                ftype = chr(fd[11])
                flen  = fd[16]
                fields.append((name, ftype, flen))
            f.seek(hdr_size)
            for _ in range(n_records):
                raw = f.read(rec_size)
                if not raw: break
                rec = {}; pos = 1
                for name, ftype, flen in fields:
                    val = raw[pos:pos+flen].decode("latin-1", errors="replace").strip()
                    rec[name] = val
                    pos += flen
                records.append(rec)
    except Exception:
        pass
    return records


def _read_shp(path):
    """Read .shp, return list of (parts, points) for polygon/polyline types."""
    shapes = []
    try:
        with open(path, "rb") as f:
            f.seek(100)  # skip file header
            while True:
                rec_hdr = f.read(8)
                if len(rec_hdr) < 8: break
                content_len = struct.unpack_from(">I", rec_hdr, 4)[0] * 2
                content = f.read(content_len)
                if len(content) < 4: break
                shp_type = struct.unpack_from("<I", content, 0)[0]
                if shp_type == 0:  # null shape
                    shapes.append(([], []))
                    continue
                if shp_type in (3, 5, 13, 15, 23, 25):  # polyline / polygon variants
                    if len(content) < 44: shapes.append(([], [])); continue
                    n_parts  = struct.unpack_from("<I", content, 36)[0]
                    n_points = struct.unpack_from("<I", content, 40)[0]
                    parts_off = 44
                    pts_off   = parts_off + n_parts * 4
                    if len(content) < pts_off + n_points * 16:
                        shapes.append(([], [])); continue
                    parts  = [struct.unpack_from("<I", content, parts_off + i*4)[0]
                              for i in range(n_parts)]
                    points = [(struct.unpack_from("<d", content, pts_off + i*16)[0],
                               struct.unpack_from("<d", content, pts_off + i*16 + 8)[0])
                              for i in range(n_points)]
                    shapes.append((parts, points))
                else:
                    shapes.append(([], []))
    except Exception:
        pass
    return shapes


def load_country_shapes(shp_path):
    """Load country shapes, return dict with India/Bangladesh/others segments."""
    dbf_path = shp_path.replace(".shp", ".dbf")
    shapes   = _read_shp(shp_path)
    records  = _read_dbf(dbf_path)
    segs = {"India": [], "Bangladesh": [], "others": []}
    for i, (parts, points) in enumerate(shapes):
        if not points: continue
        rec  = records[i] if i < len(records) else {}
        name = rec.get("NAME", rec.get("ADMIN", rec.get("name", "")))
        end_parts = list(parts) + [len(points)]
        for j in range(len(parts)):
            seg = np.array(points[end_parts[j]:end_parts[j+1]])
            if len(seg) < 2: continue
            if   "India"      in name: segs["India"].append(seg)
            elif "Bangladesh" in name: segs["Bangladesh"].append(seg)
            else:                      segs["others"].append(seg)
    return segs


def load_coastlines(shp_path):
    """Load coastline segments as list of numpy arrays."""
    shapes = _read_shp(shp_path)
    segs   = []
    for parts, points in shapes:
        if not points: continue
        end_parts = list(parts) + [len(points)]
        for j in range(len(parts)):
            seg = np.array(points[end_parts[j]:end_parts[j+1]])
            if len(seg) >= 2: segs.append(seg)
    return segs
