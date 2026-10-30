"""
Multi-color (multi-extruder) export for TouchTerrain.

Builds the terrain plus feature layers (water, roads, towns) as separate,
perfectly aligned solids so a multi-toolhead printer can print each layer
in its own color. Layers sit in shallow pockets ("inlays") cut into the
terrain, so the parts touch but never overlap.

Pipeline:
    elevation GeoTIFF (projected, meters)
      -> resample to the feature grid (feature_res_mm on the print)
      -> load features (OpenStreetMap via Overpass, or a local GeoJSON file)
      -> widen thin features to their minimum printed width
      -> rasterize one mask per layer, settle overlaps by priority
      -> build one watertight mesh per layer on a shared height field
      -> write one binary STL per layer (+ summary json)

Part of TouchTerrain_for_MPH (a fork of TouchTerrain), GPL v3.
"""

import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
from osgeo import gdal, ogr, osr

gdal.UseExceptions()
ogr.UseExceptions()

# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------

LAYERS = ("roads", "water", "towns")

DEFAULTS = {
    "multicolor": True,
    "layers": ["roads", "water", "towns"],     # which feature layers to build
    "road_min_width_mm": 0.8,                  # narrowest a road may print
    "water_min_width_mm": 1.0,                 # narrowest a river/stream may print
    "min_feature_size_mm": 2.0,                # lakes/towns smaller than this across are dropped
    "road_classes": ["motorway", "trunk", "primary", "secondary", "tertiary"],
    "town_landuse": ["residential", "commercial", "retail", "industrial"],
    "inlay_depth_mm": 0.6,                     # how deep each layer sits into the terrain
    "road_height_mm": 0.4,                     # roads stand this far above the terrain
    "water_height_mm": 0.0,                    # water surface relative to terrain (negative = recessed)
    "town_height_mm": 0.0,                     # town surface relative to terrain
    "feature_res_mm": 0.25,                    # grid size on the print when features are on
    "feature_priority": ["roads", "water", "towns"],  # first wins where layers overlap
    "toolheads": {"terrain": 1, "roads": 2, "water": 3, "towns": 4},
    "nozzle_mm": 0.4,                          # only used for warnings
    "features_file": None,                     # optional local GeoJSON instead of OpenStreetMap
    "overpass_url": "https://overpass-api.de/api/interpreter",
    "output": "stl",
}

# typical real-world widths (m) when OSM has no usable width tag
ROAD_WIDTH_M = {"motorway": 25, "trunk": 20, "primary": 15, "secondary": 12,
                "tertiary": 10, "unclassified": 7, "residential": 7, "service": 5}
WATERWAY_WIDTH_M = {"river": 30, "canal": 15, "stream": 4}


def settings_with_defaults(job):
    """Return a dict of all multicolor settings, job values over defaults."""
    s = dict(DEFAULTS)
    for k in DEFAULTS:
        if k in job and job[k] is not None:
            s[k] = job[k]
    s["toolheads"] = {**DEFAULTS["toolheads"], **(job.get("toolheads") or {})}
    bad = [l for l in s["layers"] if l not in LAYERS]
    if bad:
        raise ValueError(f"unknown layer(s) {bad}; allowed: {LAYERS}")
    return s


def check_settings(s, basethick, log=print):
    """Apply the sanity rules; return list of warnings (also logged)."""
    warns = []
    mins = []
    if "roads" in s["layers"]:
        mins.append(s["road_min_width_mm"])
    if "water" in s["layers"]:
        mins.append(s["water_min_width_mm"])
    if mins:
        cap = min(mins) / 3.0
        if s["feature_res_mm"] > cap:
            warns.append(f"feature_res_mm {s['feature_res_mm']} is coarser than a third of the "
                         f"smallest minimum width; using {cap:.3f}")
            s["feature_res_mm"] = cap
        for m in mins:
            if m < 2 * s["nozzle_mm"]:
                warns.append(f"minimum width {m} mm is under twice the {s['nozzle_mm']} mm nozzle; "
                             f"it may not print as a solid line")
    if basethick <= s["inlay_depth_mm"] + 0.2:
        warns.append(f"basethick {basethick} mm is too thin for a {s['inlay_depth_mm']} mm inlay; "
                     f"use at least {s['inlay_depth_mm'] + 0.6:.1f} mm")
    for w in warns:
        log("WARNING:", w)
    return warns


# --------------------------------------------------------------------------
# features: OpenStreetMap (Overpass) or local GeoJSON
# A feature is a dict: {"layer": str, "geom": ogr.Geometry (WGS84 lon/lat),
#                       "kind": "line"|"area", "width_m": float|None}
# --------------------------------------------------------------------------

def _wgs84():
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(4326)
    sr.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return sr


def _parse_width(tags):
    w = tags.get("width")
    if not w:
        return None
    try:
        return float(str(w).replace("m", "").strip().split(";")[0])
    except ValueError:
        return None


def overpass_query(bbox, s):
    """bbox = (south, west, north, east) in degrees."""
    b = "{:.6f},{:.6f},{:.6f},{:.6f}".format(*bbox)
    parts = []
    if "roads" in s["layers"]:
        cls = "|".join(s["road_classes"])
        parts.append(f'way["highway"~"^({cls})(_link)?$"]({b});')
    if "water" in s["layers"]:
        parts += [f'way["natural"="water"]({b});', f'relation["natural"="water"]({b});',
                  f'way["waterway"~"^(river|stream|canal|riverbank)$"]({b});',
                  f'relation["waterway"="riverbank"]({b});']
    if "towns" in s["layers"]:
        lu = "|".join(s["town_landuse"])
        parts += [f'way["landuse"~"^({lu})$"]({b});', f'relation["landuse"~"^({lu})$"]({b});']
    return "[out:json][timeout:180];(" + "".join(parts) + ");out geom;"


def fetch_overpass(bbox, s, log=print):
    q = overpass_query(bbox, s)
    data = urllib.parse.urlencode({"data": q}).encode()
    req = urllib.request.Request(s["overpass_url"], data=data,
                                 headers={"User-Agent": "TouchTerrain_for_MPH multicolor"})
    log("downloading OpenStreetMap features from", s["overpass_url"])
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=240) as r:
        osm = json.load(r)
    log(f" got {len(osm.get('elements', []))} OSM elements in {time.time() - t0:.1f} s")
    return osm_to_features(osm, s)


def _line_from_coords(coords):
    g = ogr.Geometry(ogr.wkbLineString)
    for lon, lat in coords:
        g.AddPoint_2D(lon, lat)
    return g


def _poly_from_ring(coords):
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for lon, lat in coords:
        ring.AddPoint_2D(lon, lat)
    ring.CloseRings()
    p = ogr.Geometry(ogr.wkbPolygon)
    p.AddGeometry(ring)
    return p


def _classify(tags, s):
    """Return (layer, kind, width_m) for an OSM tag set, or None."""
    hw = tags.get("highway")
    if hw:
        base = hw.replace("_link", "")
        if base in s["road_classes"]:
            w = _parse_width(tags) or ROAD_WIDTH_M.get(base, 8) * (0.6 if hw.endswith("_link") else 1)
            return "roads", "line", w
    if tags.get("natural") == "water" or tags.get("waterway") == "riverbank":
        return "water", "area", None
    ww = tags.get("waterway")
    if ww in WATERWAY_WIDTH_M:
        return "water", "line", _parse_width(tags) or WATERWAY_WIDTH_M[ww]
    if tags.get("landuse") in s["town_landuse"]:
        return "towns", "area", None
    return None


def osm_to_features(osm, s):
    feats = []
    for el in osm.get("elements", []):
        c = _classify(el.get("tags", {}), s)
        if not c or c[0] not in s["layers"]:
            continue
        layer, kind, width = c
        try:
            if el["type"] == "way" and "geometry" in el:
                coords = [(p["lon"], p["lat"]) for p in el["geometry"]]
                if len(coords) < 2:
                    continue
                if kind == "area":
                    if len(coords) < 4 or coords[0] != coords[-1]:
                        continue
                    geom = _poly_from_ring(coords)
                else:
                    geom = _line_from_coords(coords)
            elif el["type"] == "relation" and kind == "area":
                geom = _relation_polygon(el)
                if geom is None:
                    continue
            else:
                continue
        except Exception:
            continue  # skip broken geometry rather than failing the whole job
        feats.append({"layer": layer, "geom": geom, "kind": kind, "width_m": width})
    return feats


def _relation_polygon(el):
    """Assemble an OSM multipolygon relation (out geom) into an OGR polygon."""
    outer = ogr.Geometry(ogr.wkbMultiLineString)
    inner = ogr.Geometry(ogr.wkbMultiLineString)
    for m in el.get("members", []):
        if m.get("type") != "way" or "geometry" not in m:
            continue
        line = _line_from_coords([(p["lon"], p["lat"]) for p in m["geometry"]])
        (inner if m.get("role") == "inner" else outer).AddGeometry(line)
    if outer.GetGeometryCount() == 0:
        return None
    o = outer.Polygonize()
    if o is None or o.IsEmpty():
        return None
    o = o.UnionCascaded() if o.GetGeometryCount() > 1 else o.GetGeometryRef(0).Clone()
    if inner.GetGeometryCount():
        i = inner.Polygonize()
        if i is not None and not i.IsEmpty():
            o = o.Difference(i)
    return o


def load_features_file(path, s):
    """Local GeoJSON in lon/lat. Each feature needs either a 'layer' property
    (roads/water/towns, optional 'width_m') or OSM-style tags as properties."""
    with open(path) as f:
        gj = json.load(f)
    feats = []
    for ft in gj["features"]:
        props = ft.get("properties") or {}
        geom = ogr.CreateGeometryFromJson(json.dumps(ft["geometry"]))
        gtype = ogr.GT_Flatten(geom.GetGeometryType())
        kind = "area" if gtype in (ogr.wkbPolygon, ogr.wkbMultiPolygon) else "line"
        if "layer" in props:
            layer, width = props["layer"], props.get("width_m")
            if width is None and kind == "line":
                width = ROAD_WIDTH_M["tertiary"] if layer == "roads" else WATERWAY_WIDTH_M["stream"]
        else:
            c = _classify(props, s)
            if not c:
                continue
            layer, _, width = c
        if layer in s["layers"]:
            feats.append({"layer": layer, "geom": geom, "kind": kind, "width_m": width})
    return feats


# --------------------------------------------------------------------------
# raster helpers
# --------------------------------------------------------------------------

def read_dem(path):
    ds = gdal.Open(path)
    band = ds.GetRasterBand(1)
    a = band.ReadAsArray().astype(np.float64)
    nd = band.GetNoDataValue()
    if nd is not None:
        a[a == nd] = np.nan
    a[a < -1e30] = np.nan
    return ds, a


def resample_dem(src_path, cell_m, out_path):
    """Resample the DEM to a square grid of cell_m meters (bilinear)."""
    gdal.Warp(out_path, src_path, xRes=cell_m, yRes=cell_m, resampleAlg="bilinear",
              dstNodata=-3.4e38, format="GTiff", creationOptions=["COMPRESS=DEFLATE"])
    return out_path


def dem_bbox_wgs84(ds):
    """(south, west, north, east) of a projected dataset."""
    gt = ds.GetGeoTransform()
    nx, ny = ds.RasterXSize, ds.RasterYSize
    src = osr.SpatialReference(wkt=ds.GetProjection())
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    tr = osr.CoordinateTransformation(src, _wgs84())
    xs, ys = [], []
    for px, py in ((0, 0), (nx, 0), (0, ny), (nx, ny), (nx / 2, 0), (nx / 2, ny), (0, ny / 2), (nx, ny / 2)):
        x = gt[0] + px * gt[1] + py * gt[2]
        y = gt[3] + px * gt[4] + py * gt[5]
        lon, lat, _ = tr.TransformPoint(x, y)
        xs.append(lon)
        ys.append(lat)
    return (min(ys), min(xs), max(ys), max(xs))


def features_to_masks(feats, ds, s, m_per_mm, log=print):
    """Widen features to their minimum printed width and rasterize one
    boolean mask per layer on the dataset's grid.

    Rasterizing keeps a cell only when its centre is inside a shape, which
    can shave up to one cell off a narrow strip (more on diagonals). Widened
    features therefore get half a cell diagonal of extra margin on each side:
    any point within the minimum half-width then lies in a cell whose centre
    is inside the shape, so the printed width never falls under the minimum
    (on average it comes out about one cell wider)."""
    margin = 0.5 * math.sqrt(2) * abs(ds.GetGeoTransform()[1])
    proj = osr.SpatialReference(wkt=ds.GetProjection())
    proj.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    tr = osr.CoordinateTransformation(_wgs84(), proj)
    min_w = {"roads": s["road_min_width_mm"] * m_per_mm,
             "water": s["water_min_width_mm"] * m_per_mm}
    stats = {l: {"features": 0, "widened": 0} for l in s["layers"]}
    shapes = {l: [] for l in s["layers"]}
    for f in feats:
        g = f["geom"].Clone()
        g.Transform(tr)
        layer = f["layer"]
        stats[layer]["features"] += 1
        if f["kind"] == "line":
            true_w = f["width_m"] or 0
            mw = min_w.get(layer, 0)
            if mw > true_w:
                stats[layer]["widened"] += 1
                g = g.Buffer(mw / 2.0 + margin, 4)
            else:
                g = g.Buffer(true_w / 2.0, 4)
        elif layer == "water":
            # an area too thin everywhere for the minimum (a narrow river polygon
            # with no centerline) is grown to the minimum width
            mw = min_w["water"]
            thin = g.Buffer(-mw / 2.0, 2)
            if thin is None or thin.IsEmpty():
                g = g.Buffer(mw / 2.0 + margin, 4)
                stats[layer]["widened"] += 1
        if g is not None and not g.IsEmpty():
            shapes[layer].append(g)

    nx, ny = ds.RasterXSize, ds.RasterYSize
    masks = {}
    mem = ogr.GetDriverByName("Memory")
    for layer in s["layers"]:
        tgt = gdal.GetDriverByName("MEM").Create("", nx, ny, 1, gdal.GDT_Byte)
        tgt.SetGeoTransform(ds.GetGeoTransform())
        tgt.SetProjection(ds.GetProjection())
        src = mem.CreateDataSource("m")
        lyr = src.CreateLayer("f", srs=proj, geom_type=ogr.wkbUnknown)
        for g in shapes[layer]:
            ft = ogr.Feature(lyr.GetLayerDefn())
            ft.SetGeometry(g)
            lyr.CreateFeature(ft)
        gdal.RasterizeLayer(tgt, [1], lyr, burn_values=[1])
        masks[layer] = tgt.GetRasterBand(1).ReadAsArray().astype(bool)
    for layer in s["layers"]:
        log(f" {layer}: {stats[layer]['features']} features, "
            f"{stats[layer]['widened']} widened to the minimum, {int(masks[layer].sum())} cells")
    return masks, stats


def drop_small(mask, min_cells):
    """Remove connected blobs smaller than min_cells."""
    from scipy import ndimage
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_cells
    keep[0] = False
    return keep[lab]


def build_labels(masks, valid, s, feature_res_mm, log=print):
    """Combine masks into one label grid: 0 = terrain, 1..K = layer index in
    s['layers'] order. Priority list decides overlaps (first wins)."""
    lab = np.zeros(valid.shape, np.int16)
    min_cells = (s["min_feature_size_mm"] / feature_res_mm) ** 2 * 0.785  # circle of that diameter
    order = [l for l in s["feature_priority"] if l in s["layers"]] + \
            [l for l in s["layers"] if l not in s["feature_priority"]]
    for layer in reversed(order):              # lowest priority first, higher overwrite
        m = masks[layer] & valid
        if layer in ("water", "towns"):
            before = int(m.sum())
            m = drop_small(m, min_cells)
            if before - int(m.sum()):
                log(f" {layer}: dropped {before - int(m.sum())} cells in features under "
                    f"{s['min_feature_size_mm']} mm across")
        lab[m] = s["layers"].index(layer) + 1
    return lab


# --------------------------------------------------------------------------
# meshing
# --------------------------------------------------------------------------

def corner_heights(z):
    """Cell-centre heights (ny,nx, NaN outside) -> corner heights (ny+1,nx+1),
    each the mean of the valid cells touching that corner."""
    ny, nx = z.shape
    p = np.full((ny + 2, nx + 2), np.nan)
    p[1:-1, 1:-1] = z
    stack = np.stack([p[:-1, :-1], p[:-1, 1:], p[1:, :-1], p[1:, 1:]])
    with np.errstate(invalid="ignore"):
        cnt = np.sum(~np.isnan(stack), axis=0)
        s = np.nansum(stack, axis=0)
        h = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    return h


def body_mesh(occ, t, hc, cell_mm, bottom=None, u=None):
    """Watertight triangle soup for one solid.

    occ    (ny,nx) bool  cells this body occupies
    t      (ny,nx) float top offset per cell; top corner z = hc + t
    hc     (ny+1,nx+1)   shared corner height field (mm)
    bottom float or None flat bottom z (terrain), else bottom = hc + u
    u      float         bottom offset when bottom is None
    Returns (n,3,3) float32 triangles, counter-clockwise seen from outside.
    """
    ny, nx = occ.shape
    I, J = np.nonzero(occ)
    if I.size == 0:
        return np.zeros((0, 3, 3), np.float32)
    tc = t[I, J]

    # corner (row, col) -> xy ; row 0 is north, y grows north
    def xy(ci, cj):
        return cj * cell_mm, (ny - ci) * cell_mm

    def top(ci, cj, toff):
        return hc[ci, cj] + toff

    def bot(ci, cj):
        if bottom is not None:
            return np.full(ci.shape, float(bottom))
        return hc[ci, cj] + u

    def P(ci, cj, z):
        x, y = xy(ci, cj)
        return np.stack([x, y, z], axis=-1)

    # corners: A top-left (I,J), B top-right (I,J+1), C bottom-right (I+1,J+1), D bottom-left (I+1,J)
    A = (I, J); B = (I, J + 1); C = (I + 1, J + 1); D = (I + 1, J)
    At, Bt, Ct, Dt = (P(*k, top(*k, tc)) for k in (A, B, C, D))
    Ab, Bb, Cb, Db = (P(*k, bot(*k)) for k in (A, B, C, D))

    tris = [np.stack([At, Dt, Ct], 1), np.stack([At, Ct, Bt], 1),     # top, normal +z
            np.stack([Ab, Cb, Db], 1), np.stack([Ab, Bb, Cb], 1)]     # bottom, normal -z

    # walls: edges in CCW order (from above) P->Q, neighbour across that edge
    edges = [(A, D, (I, J - 1)),        # west
             (D, C, (I + 1, J)),        # south
             (C, B, (I, J + 1)),        # east
             (B, A, (I - 1, J))]        # north
    # lowest top offset of the occupied cells around each corner: where a
    # higher cell's outer wall meets a step down, the wall needs a vertex at
    # the lower level too, or the mesh gets a T-junction (an open edge)
    tp = np.full((ny + 2, nx + 2), np.inf)
    tp[1:-1, 1:-1] = np.where(occ, t, np.inf)
    tmin_c = np.minimum.reduce([tp[:-1, :-1], tp[:-1, 1:], tp[1:, :-1], tp[1:, 1:]])

    for (p, q, (ni, nj)) in edges:
        inside = (ni >= 0) & (ni < ny) & (nj >= 0) & (nj < nx)
        nocc = np.zeros(I.shape, bool)
        nt = np.zeros(I.shape)
        nocc[inside] = occ[ni[inside], nj[inside]]
        nt[inside] = t[ni[inside], nj[inside]]
        # case 1: neighbour outside the body -> wall from top down to bottom,
        # as a fan around its centre: pb, qb, qmid, qt, pt, pmid
        k = ~nocc
        if k.any():
            pi, pj, qi, qj, tk = p[0][k], p[1][k], q[0][k], q[1][k], tc[k]
            pt = P(pi, pj, top(pi, pj, tk))
            qt = P(qi, qj, top(qi, qj, tk))
            pb = P(pi, pj, bot(pi, pj))
            qb = P(qi, qj, bot(qi, qj))
            pm = P(pi, pj, top(pi, pj, np.minimum(tk, tmin_c[pi, pj])))
            qm = P(qi, qj, top(qi, qj, np.minimum(tk, tmin_c[qi, qj])))
            c = (pt + qt + pb + qb) / 4.0
            ring = [pb, qb, qm, qt, pt, pm]
            for a in range(6):
                tris.append(np.stack([c, ring[a], ring[(a + 1) % 6]], 1))
        # case 2: neighbour in the body but lower top -> step wall down to its top
        k = nocc & (tc > nt + 1e-9)
        if k.any():
            pt = P(p[0][k], p[1][k], top(p[0][k], p[1][k], tc[k]))
            qt = P(q[0][k], q[1][k], top(q[0][k], q[1][k], tc[k]))
            pl = P(p[0][k], p[1][k], top(p[0][k], p[1][k], nt[k]))
            ql = P(q[0][k], q[1][k], top(q[0][k], q[1][k], nt[k]))
            tris += [np.stack([pl, ql, qt], 1), np.stack([pl, qt, pt], 1)]
    out = np.concatenate(tris).astype(np.float32)
    # drop zero-height wall triangles
    e1 = out[:, 1] - out[:, 0]
    e2 = out[:, 2] - out[:, 0]
    area2 = np.linalg.norm(np.cross(e1, e2), axis=1)
    return out[area2 > 1e-9]


def write_stl(path, tris, name="touchterrain"):
    tris = np.asarray(tris, np.float32)
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    n = np.cross(e1, e2)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = (n / np.where(ln == 0, 1, ln)).astype(np.float32)
    rec = np.zeros(len(tris), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"] = n
    rec["v"] = tris
    with open(path, "wb") as f:
        f.write(name.encode()[:80].ljust(80, b" "))
        f.write(np.uint32(len(tris)).tobytes())
        f.write(rec.tobytes())


# --------------------------------------------------------------------------
# main entry
# --------------------------------------------------------------------------

def make_multicolor(dem_path, job, out_dir, log=print):
    """Build multicolor STLs from a projected (meters) DEM GeoTIFF.

    job: TouchTerrain-style settings (tilewidth, zscale, basethick, ...) plus
         the multicolor settings in DEFAULTS.
    Returns a summary dict; files are written into out_dir.
    """
    s = settings_with_defaults(job)
    tilewidth = float(job.get("tilewidth", 120))
    zscale = float(job.get("zscale", 1.0))
    basethick = float(job.get("basethick", 2.0))
    warns = check_settings(s, basethick, log)
    os.makedirs(out_dir, exist_ok=True)

    src = gdal.Open(dem_path)
    gt = src.GetGeoTransform()
    if osr.SpatialReference(wkt=src.GetProjection()).IsGeographic():
        raise ValueError("DEM must be in a projected (meter) coordinate system, e.g. UTM")
    width_m = src.RasterXSize * abs(gt[1])
    m_per_mm = width_m / tilewidth
    cell_m = s["feature_res_mm"] * m_per_mm
    log(f"model {tilewidth} mm wide = {width_m:.0f} m -> scale 1:{m_per_mm * 1000:,.0f}")
    log(f"feature grid {s['feature_res_mm']} mm = {cell_m:.1f} m cells")
    src = None

    rs_path = os.path.join(out_dir, "_dem_resampled.tif")
    resample_dem(dem_path, cell_m, rs_path)
    ds, elev = read_dem(rs_path)
    valid = ~np.isnan(elev)
    if not valid.any():
        raise ValueError("DEM has no valid cells")
    cell_mm = abs(ds.GetGeoTransform()[1]) / m_per_mm
    log(f"grid {elev.shape[1]} x {elev.shape[0]} cells")

    # features
    if s["features_file"]:
        log("reading features from", s["features_file"])
        feats = load_features_file(s["features_file"], s)
    else:
        feats = fetch_overpass(dem_bbox_wgs84(ds), s, log)
    masks, fstats = features_to_masks(feats, ds, s, m_per_mm, log)
    labels = build_labels(masks, valid, s, s["feature_res_mm"], log)

    # heights in mm on the print
    min_el = float(job["min_elev"]) if job.get("min_elev") is not None else float(np.nanmin(elev))
    z = (elev - min_el) / m_per_mm * zscale + basethick
    hc = corner_heights(z)

    inlay = s["inlay_depth_mm"]
    offsets = {"roads": s["road_height_mm"], "water": s["water_height_mm"],
               "towns": s["town_height_mm"]}
    bodies = {}
    t_terrain = np.where(labels > 0, -inlay, 0.0)
    bodies["terrain"] = body_mesh(valid, t_terrain, hc, cell_mm, bottom=0.0)
    for k, layer in enumerate(s["layers"], start=1):
        occ = labels == k
        if occ.any():
            t = np.full(labels.shape, float(offsets[layer]))
            bodies[layer] = body_mesh(occ, t, hc, cell_mm, u=-inlay)
        else:
            log(f" {layer}: nothing in this area, no file written")

    base = job.get("zip_file_name", "terrain")
    files = {}
    for name, tris in bodies.items():
        th = s["toolheads"].get(name, "")
        fn = f"{base}_{name}_T{th}.stl" if th else f"{base}_{name}.stl"
        path = os.path.join(out_dir, fn)
        write_stl(path, tris, name)
        files[name] = fn
        log(f" wrote {fn}: {len(tris):,} triangles, {os.path.getsize(path) / 1048576:.1f} MB")

    os.remove(rs_path)
    summary = {
        "scale": f"1:{m_per_mm * 1000:,.0f}",
        "model_mm": [round(elev.shape[1] * cell_mm, 2), round(elev.shape[0] * cell_mm, 2)],
        "feature_res_mm": round(cell_mm, 4),
        "settings": {k: s[k] for k in DEFAULTS if k not in ("overpass_url",)},
        "features": fstats,
        "cells": {name: int((labels == k).sum()) for k, name in enumerate(["terrain"] + s["layers"])},
        "files": files,
        "warnings": warns,
    }
    with open(os.path.join(out_dir, f"{base}_multicolor_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary
