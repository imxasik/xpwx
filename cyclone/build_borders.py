"""Build the country-border line layer from the bundled Natural Earth countries.

Input : map/ne_50m_admin_0_countries.shp  (Natural Earth 50 m, public domain)
Output: map/ne_50m_admin_0_borders.shp (+ .shx, .dbf, .prj)

Only the parts of each country outline that lie within BORDER_TOL_DEG of a
neighbouring country's outline are kept. Those are the land borders. The
coastlines, which touch no other country, are dropped.

Run once after changing the source data:  python -m cyclone.build_borders
"""
import os
import shutil

import shapefile
from shapely.geometry import LineString, MultiLineString, shape
from shapely.ops import linemerge, unary_union
from shapely import set_precision

BORDER_TOL_DEG = 0.03     # outlines closer than this are treated as a shared border
SIMPLIFY_DEG = 0.01       # output vertex spacing (keeps the file small)

HERE = os.path.dirname(os.path.abspath(__file__))
MAP_DIR = os.path.normpath(os.path.join(HERE, "..", "map"))
SRC = os.path.join(MAP_DIR, "ne_50m_admin_0_countries.shp")
OUT = os.path.join(MAP_DIR, "ne_50m_admin_0_borders.shp")


def _lines(geom):
    """Flatten a geometry into a list of LineStrings."""
    if geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom]
    if isinstance(geom, MultiLineString) or hasattr(geom, "geoms"):
        out = []
        for g in geom.geoms:
            out.extend(_lines(g))
        return out
    return []


def build(src=SRC, out=OUT, tol=BORDER_TOL_DEG, simplify=SIMPLIFY_DEG):
    sf = shapefile.Reader(src)
    polys = [shape(s.__geo_interface__) for s in sf.shapes()]
    bounds = [p.bounds for p in polys]
    outlines = [p.boundary for p in polys]
    buffered = [None] * len(polys)

    border_parts = []
    for i, bi in enumerate(bounds):
        pieces = []
        for j, bj in enumerate(bounds):
            if i == j:
                continue
            # cheap bbox rejection (tol is in degrees)
            if (bi[2] + tol < bj[0] or bj[2] + tol < bi[0] or
                    bi[3] + tol < bj[1] or bj[3] + tol < bi[1]):
                continue
            if buffered[j] is None:
                buffered[j] = outlines[j].buffer(tol)
            near = outlines[i].intersection(buffered[j])
            pieces.extend(_lines(near))
        border_parts.extend(pieces)

    merged = linemerge(unary_union(border_parts))
    lines = _lines(merged)
    lines = [set_precision(l.simplify(simplify), 0.001) for l in lines]
    lines = [ln for ln in lines if not ln.is_empty and len(ln.coords) >= 2]

    w = shapefile.Writer(out, shapeType=shapefile.POLYLINE)
    w.field("ID", "N", size=6)
    for k, ln in enumerate(lines):
        w.line([[list(map(float, c)) for c in ln.coords]])
        w.record(k)
    w.close()
    shutil.copyfile(os.path.join(MAP_DIR, "ne_50m_admin_0_countries.prj"),
                    os.path.splitext(out)[0] + ".prj")
    return len(lines)


if __name__ == "__main__":
    n = build()
    print(f"wrote {n} border lines to {OUT}")
