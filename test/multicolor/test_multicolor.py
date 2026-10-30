"""
Tests for the multicolor export. Needs no Earth Engine or internet:
uses the repo's sample DEM and a small hand-made features file.

    python -m pytest test/multicolor      (or)      python test/multicolor/test_multicolor.py
"""
import json
import math
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)
from touchterrain.common import multicolor as mc  # noqa: E402

DEM = os.path.join(ROOT, "stuff", "geotiff_example.tif")
FEATURES = os.path.join(HERE, "test_features.geojson")


def _job(**kw):
    j = {"tilewidth": 120, "zscale": 2.0, "basethick": 2.0, "zip_file_name": "t",
         "features_file": FEATURES}
    j.update(kw)
    return j


def _read_stl(path):
    with open(path, "rb") as f:
        f.read(80)
        n = int(np.frombuffer(f.read(4), np.uint32)[0])
        rec = np.frombuffer(f.read(), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")], count=n)
    return rec["v"].astype(np.float64)


def _edge_check(tris):
    """Every edge must be used exactly twice with opposite directions
    (closed, consistently oriented), except where 4 faces meet at a
    diagonal-only cell contact, which slicers accept."""
    v = np.round(tris.reshape(-1, 3), 5)
    _, inv = np.unique(v, axis=0, return_inverse=True)
    f = inv.reshape(-1, 3)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    fwd = {}
    for a, b in map(tuple, e):
        fwd[(a, b)] = fwd.get((a, b), 0) + 1
    bad = sum(1 for (a, b), c in fwd.items() if fwd.get((b, a), 0) != c)
    return bad, len(fwd)


def _volume(tris):
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def test_build_and_check():
    out = tempfile.mkdtemp()
    s = mc.make_multicolor(DEM, _job(), out, log=lambda *a: None)
    assert set(s["files"]) == {"terrain", "roads", "water", "towns"}
    vols = {}
    for layer, fn in s["files"].items():
        tris = _read_stl(os.path.join(out, fn))
        bad, n = _edge_check(tris)
        assert bad == 0, f"{layer}: {bad} of {n} edges not closed/consistent"
        vols[layer] = _volume(tris)
        assert vols[layer] > 0, f"{layer}: inside-out (negative volume)"
    # toolhead numbers in file names follow the default mapping
    assert s["files"]["roads"].endswith("_T2.stl") and s["files"]["water"].endswith("_T3.stl")
    return s, vols


def test_alignment_same_footprint():
    """All parts share one coordinate system: every feature part lies
    inside the terrain's xy extent."""
    out = tempfile.mkdtemp()
    s = mc.make_multicolor(DEM, _job(), out, log=lambda *a: None)
    terr = _read_stl(os.path.join(out, s["files"]["terrain"])).reshape(-1, 3)
    lo, hi = terr.min(0), terr.max(0)
    for layer in ("roads", "water", "towns"):
        p = _read_stl(os.path.join(out, s["files"][layer])).reshape(-1, 3)
        assert (p[:, :2] >= lo[:2] - 1e-4).all() and (p[:, :2] <= hi[:2] + 1e-4).all(), layer


def _widths_along_lines(mask, feats, ds, cell_mm):
    """Measure the printed width (mm) of a rasterized line layer at many
    points, perpendicular to each line's own direction. The cells are
    treated as the squares they print as."""
    from osgeo import osr
    gt = ds.GetGeoTransform()
    proj = osr.SpatialReference(wkt=ds.GetProjection())
    proj.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    tr = osr.CoordinateTransformation(mc._wgs84(), proj)
    ny, nx = mask.shape
    widths = []
    for f in feats:
        g = f["geom"].Clone()
        g.Transform(tr)
        pts = [((x - gt[0]) / gt[1], (y - gt[3]) / gt[5]) for x, y in g.GetPoints()]  # pixel col,row
        for (c0, r0), (c1, r1) in zip(pts[:-1], pts[1:]):
            L = math.hypot(c1 - c0, r1 - r0)
            if L == 0:
                continue
            dc, dr = (c1 - c0) / L, (r1 - r0) / L
            pc, pr = -dr, dc
            for u in np.arange(3, L - 3, 0.5):        # skip the segment ends (bends, joins)
                c, r = c0 + u * dc, r0 + u * dr
                if not (8 < c < nx - 8 and 8 < r < ny - 8):
                    continue
                span = []
                for sgn in (1, -1):
                    t = 0.0
                    while True:
                        cc, rr = c + sgn * t * pc, r + sgn * t * pr
                        if not mask[int(math.floor(rr)), int(math.floor(cc))]:
                            break
                        t += 0.02
                    span.append(t)
                widths.append(sum(span) * cell_mm)
    return np.array(widths)


def test_minimum_width_scales():
    """Roads are never narrower than road_min_width_mm on the print: widened
    at small scale (1:125,000), true width at large scale (1:31,000)."""
    from osgeo import gdal
    for tilewidth in (120, 480):
        out = tempfile.mkdtemp()
        job = _job(tilewidth=tilewidth, layers=["roads"])
        s = mc.settings_with_defaults(job)
        src = gdal.Open(DEM)
        m_per_mm = src.RasterXSize * abs(src.GetGeoTransform()[1]) / tilewidth
        rs = mc.resample_dem(DEM, s["feature_res_mm"] * m_per_mm, os.path.join(out, "r.tif"))
        ds, elev = mc.read_dem(rs)
        feats = [f for f in mc.load_features_file(FEATURES, s) if f["kind"] == "line"]
        masks, _ = mc.features_to_masks(feats, ds, s, m_per_mm, log=lambda *a: None)
        cell_mm = abs(ds.GetGeoTransform()[1]) / m_per_mm
        w = _widths_along_lines(masks["roads"], feats, ds, cell_mm)
        assert len(w) > 50
        assert w.min() >= s["road_min_width_mm"] - 1e-6, \
            f"tilewidth {tilewidth}: narrowest road {w.min():.3f} mm < {s['road_min_width_mm']} mm"
        print(f"   1:{m_per_mm * 1000:,.0f}  road width min {w.min():.2f} / median {np.median(w):.2f} mm "
              f"over {len(w)} samples")


def test_settings_rules():
    s = mc.settings_with_defaults({"feature_res_mm": 1.0})
    w = mc.check_settings(s, basethick=0.5, log=lambda *a: None)
    assert s["feature_res_mm"] <= 0.8 / 3 + 1e-9
    assert any("basethick" in x for x in w)
    s = mc.settings_with_defaults({"road_min_width_mm": 0.5})
    w = mc.check_settings(s, basethick=2, log=lambda *a: None)
    assert any("nozzle" in x for x in w)


def test_overpass_parsing():
    """OSM json (as Overpass returns it) is classified into layers."""
    osm = {"elements": [
        {"type": "way", "tags": {"highway": "primary"},
         "geometry": [{"lat": 44.55, "lon": -108.2}, {"lat": 44.56, "lon": -108.1}]},
        {"type": "way", "tags": {"highway": "residential"},
         "geometry": [{"lat": 44.55, "lon": -108.2}, {"lat": 44.56, "lon": -108.1}]},
        {"type": "way", "tags": {"waterway": "river", "width": "45"},
         "geometry": [{"lat": 44.55, "lon": -108.2}, {"lat": 44.56, "lon": -108.1}]},
        {"type": "relation", "tags": {"natural": "water", "type": "multipolygon"}, "members": [
            {"type": "way", "role": "outer", "geometry": [
                {"lat": 44.55, "lon": -108.10}, {"lat": 44.55, "lon": -108.05}, {"lat": 44.58, "lon": -108.05}]},
            {"type": "way", "role": "outer", "geometry": [
                {"lat": 44.58, "lon": -108.05}, {"lat": 44.58, "lon": -108.10}, {"lat": 44.55, "lon": -108.10}]}]},
        {"type": "way", "tags": {"landuse": "retail"},
         "geometry": [{"lat": 44.55, "lon": -108.2}, {"lat": 44.56, "lon": -108.2},
                      {"lat": 44.56, "lon": -108.19}, {"lat": 44.55, "lon": -108.2}]},
    ]}
    s = mc.settings_with_defaults({})
    f = mc.osm_to_features(osm, s)
    kinds = sorted((x["layer"], x["kind"]) for x in f)
    assert kinds == [("roads", "line"), ("towns", "area"), ("water", "area"), ("water", "line")]
    river = [x for x in f if x["kind"] == "line" and x["layer"] == "water"][0]
    assert river["width_m"] == 45
    lake = [x for x in f if x["kind"] == "area" and x["layer"] == "water"][0]
    assert lake["geom"].GetArea() > 0
    assert "highway" in mc.overpass_query((44.5, -108.3, 44.7, -108.0), s)


if __name__ == "__main__":
    s, vols = test_build_and_check()
    print("watertight + oriented: OK;  volumes (mm3):", {k: round(v) for k, v in vols.items()})
    test_alignment_same_footprint(); print("alignment: OK")
    test_minimum_width_scales(); print("minimum width at two scales: OK")
    test_settings_rules(); print("settings rules: OK")
    test_overpass_parsing(); print("OpenStreetMap parsing: OK")
