#!/usr/bin/env python3
"""
Riyadh obstacles - best available heights for buildings and structures around all Riyadh runways.

Rules (defaults):  within NEAR_KM of any runway: height > NEAR_MIN   (10 km, > 15 m)
                   NEAR_KM .. FAR_KM:             height >= FAR_MIN  (10-45 km, >= 46 m)

Height sources, best first (one height per object; overlapping footprints merged, tallest part kept):
  1  OSM explicit height tag (building, building:part, tower, mast, chimney ...)
  2  Overture Maps building height (non-OSM contributors)
  3  OSM / Overture floor count x FLOOR_M (estimate)
  4  GlobalBuildingAtlas LoD1 height (satellite estimate, underestimates towers)
Runways: OSM aeroway=runway in the Riyadh region, plus OurAirports runways for any airfield OSM lacks.
"""
import argparse, csv, json, math, os, re, subprocess, sys, time
from pathlib import Path

import logging
import requests
import shapefile
logging.getLogger('shapefile').setLevel(logging.ERROR)
from shapely.geometry.polygon import orient
from shapely.geometry import Polygon, MultiPolygon, Point, LineString, shape, mapping
from shapely.strtree import STRtree
from shapely import wkb as shp_wkb

R = 6371008.8
REGION = (45.9, 24.1, 47.5, 25.6)          # lon_min, lat_min, lon_max, lat_max: where to look for runways
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]
CHECKS = {  # known structures for the validation report (name: lat, lon, published height m)
    "Kingdom Centre": (24.71139, 46.67444, 302), "Al Faisaliah Tower": (24.69047, 46.68516, 267),
    "PIF Tower (KAFD)": (24.7635, 46.6406, 385), "Burj Rafal": (24.7909, 46.6600, 308),
    "Riyadh TV Tower": (24.6501, 46.7108, 170),
}
OURAIRPORTS_RUNWAYS = [  # used only for airfields with no OSM runway
    ("OERK King Khalid International", "15L/33R", 24.976900, 46.702000, 24.944000, 46.722801),
    ("OERK King Khalid International", "15R/33L", 24.970900, 46.674900, 24.938000, 46.695701),
    ("OETH Thumamah", "17/35", 25.230900, 46.637798, 25.195101, 46.644101),
    ("OE53 King Abdul Aziz Military Academy", "15/33", 24.951200, 46.385700, 24.933100, 46.396801),
    ("OERY Riyadh Air Base (old airport)", "01/19", 24.705099, 46.724500, 24.741800, 46.730202),
    ("OERY Riyadh Air Base (old airport)", "15/33", 24.722700, 46.697800, 24.707800, 46.729301),
]


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ helpers
def parse_height(v):
    if v is None:
        return None
    s = str(v).strip().lower().replace(",", ".")
    m = re.match(r"^~?\s*(-?\d+(?:\.\d+)?)\s*(m|meter|meters|metre|metres|ft|feet|')?\s*$", s)
    if not m:
        m = re.match(r"^(\d+(?:\.\d+)?)", s)
        if not m:
            return None
        val, unit = float(m.group(1)), None
    else:
        val, unit = float(m.group(1)), m.group(2)
    if unit in ("ft", "feet", "'"):
        val *= 0.3048
    return val if 0 < val < 1000 else None


def parse_levels(v):
    try:
        x = float(str(v).split(";")[0].replace(",", "."))
        return x if 0 < x < 200 else None
    except (TypeError, ValueError):
        return None


class Proj:
    """Local equirectangular metres around the area centre (for overlaps/areas only)."""
    def __init__(self, lat0, lon0):
        self.lat0, self.lon0, self.k = math.radians(lat0), lon0, math.cos(math.radians(lat0))

    def fwd(self, lon, lat):
        return math.radians(lon - self.lon0) * self.k * R, (math.radians(lat) - self.lat0) * R

    def geom(self, g):
        import shapely, numpy as np
        k, lon0, lat0 = self.k, self.lon0, self.lat0
        return shapely.transform(g, lambda a: np.column_stack([np.radians(a[:, 0] - lon0) * k * R, (np.radians(a[:, 1]) - lat0) * R]))


class RunwayDistance:
    def __init__(self, runways):
        self.rw = []
        for r in runways:
            lat0 = math.radians((r["lat1"] + r["lat2"]) / 2); lon0 = (r["lon1"] + r["lon2"]) / 2; k = math.cos(lat0)
            f = lambda la, lo: (math.radians(lo - lon0) * k * R, (math.radians(la) - lat0) * R)
            self.rw.append((lat0, lon0, k, *f(r["lat1"], r["lon1"]), *f(r["lat2"], r["lon2"]), f'{r["airport"]} {r["runway"]}'))

    def nearest(self, pts):
        best, name = float("inf"), ""
        for lat0, lon0, k, ax, ay, bx, by, nm in self.rw:
            dx, dy = bx - ax, by - ay; L2 = dx * dx + dy * dy
            for lon, lat in pts:
                px, py = math.radians(lon - lon0) * k * R, (math.radians(lat) - lat0) * R
                t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
                d = math.hypot(px - (ax + t * dx), py - (ay + t * dy))
                if d < best:
                    best, name = d, nm
        return best, name


def overpass(query, label):
    for attempt in range(3):
        for url in OVERPASS:
            try:
                log(f"  Overpass ({label}) -> {url}")
                r = requests.post(url, data={"data": query}, timeout=1200, headers={"User-Agent": "riyadh-obstacles/1.0 (GitHub Actions)"})
                if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json"):
                    js = r.json()
                    log(f"    {len(js.get('elements', [])):,} elements")
                    return js
                log(f"    HTTP {r.status_code}: {r.text[:200]}")
            except Exception as e:
                log(f"    failed: {e}")
        time.sleep(30)
    sys.exit(f"Overpass query '{label}' failed on all servers")


def osm_geom(el):
    """Shapely geometry (lon/lat) of an Overpass element returned with 'out geom'."""
    t = el["type"]
    if t == "node":
        return Point(el["lon"], el["lat"])
    if t == "way":
        pts = [(p["lon"], p["lat"]) for p in el.get("geometry", []) if p]
        if len(pts) >= 4 and pts[0] == pts[-1]:
            g = Polygon(pts)
            return g if g.is_valid else g.buffer(0)
        return LineString(pts) if len(pts) >= 2 else (Point(*pts[0]) if pts else None)
    if t == "relation":
        outers = []
        for m in el.get("members", []):
            if m.get("type") == "way" and m.get("role", "outer") in ("outer", "") and m.get("geometry"):
                pts = [(p["lon"], p["lat"]) for p in m["geometry"] if p]
                if len(pts) >= 4 and pts[0] == pts[-1]:
                    outers.append(Polygon(pts))
        if outers:
            g = MultiPolygon(outers) if len(outers) > 1 else outers[0]
            return g if g.is_valid else g.buffer(0)
        b = el.get("bounds")
        if b:
            return Point((b["minlon"] + b["maxlon"]) / 2, (b["minlat"] + b["maxlat"]) / 2)
    return None


# ------------------------------------------------------------------ OpenStreetMap (Geofabrik extract + osmium)
OSM_FEATURES = None


def load_osm(pbf):
    """Filter the Geofabrik extract to the Riyadh region with osmium and return GeoJSON features."""
    global OSM_FEATURES
    if OSM_FEATURES is not None:
        return OSM_FEATURES
    w, s, e, n = REGION[0] - 0.6, REGION[1] - 0.6, REGION[2] + 0.6, REGION[3] + 0.6
    work = Path("osm_work"); work.mkdir(exist_ok=True)
    cut, flt, gj = work / "region.osm.pbf", work / "filtered.osm.pbf", work / "features.geojsonseq"
    run = lambda cmd: (log("   $ " + " ".join(cmd)), subprocess.run(cmd, check=True))
    run(["osmium", "extract", "-b", f"{w},{s},{e},{n}", str(pbf), "-o", str(cut), "--overwrite", "-s", "smart"])
    run(["osmium", "tags-filter", str(cut), "nwr/building:levels", "nwr/height", "nwr/est_height",
         "nwr/man_made=tower,mast,chimney,water_tower,communications_tower,antenna,crane,silo,flagpole",
         "nwr/tower:type", "n/power=tower", "nwr/aerialway=pylon", "nwr/aeroway=runway,aerodrome",
         "-o", str(flt), "--overwrite"])
    run(["osmium", "export", str(flt), "-f", "geojsonseq", "-a", "type,id", "-o", str(gj), "--overwrite"])
    feats = []
    with open(gj, encoding="utf-8") as f:
        for line in f:
            line = line.strip().lstrip("\x1e")
            if line:
                try:
                    feats.append(json.loads(line))
                except ValueError:
                    pass
    best = {}
    rank = {"Polygon": 0, "MultiPolygon": 0, "Point": 1, "LineString": 2, "MultiLineString": 2}
    for ft in feats:
        k = (ft["properties"].get("@type"), ft["properties"].get("@id"))
        if k not in best or rank.get(ft["geometry"]["type"], 3) < rank.get(best[k]["geometry"]["type"], 3):
            best[k] = ft
    log(f"   {len(feats):,} OSM features exported, {len(best):,} distinct objects")
    feats = list(best.values())
    OSM_FEATURES = feats
    return feats


def fgeom(ft):
    try:
        g = shape(ft["geometry"])
        return g if g.is_valid else g.buffer(0)
    except Exception:
        return None


def fref(ft):
    p = ft["properties"]
    return f'osm {p.get("@type", "")}/{p.get("@id", "")}'


# ------------------------------------------------------------------ runways
def get_runways(pbf, min_len=800):
    feats = load_osm(pbf)
    w, s, e, n = REGION
    aerodromes = []
    for ft in feats:
        p = ft["properties"]
        if p.get("aeroway") == "aerodrome":
            g = fgeom(ft)
            if g is not None:
                aerodromes.append((g, p.get("name:en") or p.get("name") or "", p.get("icao") or ""))
    rws, short = [], 0
    for ft in feats:
        p = ft["properties"]
        if p.get("aeroway") != "runway":
            continue
        g = fgeom(ft)
        if g is None:
            continue
        if g.geom_type in ("Polygon", "MultiPolygon"):
            ring = list((g.geoms[0] if g.geom_type == "MultiPolygon" else g).exterior.coords)
            pts = list(max(((a, b) for i, a in enumerate(ring) for b in ring[i + 1:]),
                           key=lambda ab: (ab[0][0] - ab[1][0]) ** 2 + (ab[0][1] - ab[1][1]) ** 2))
        elif g.geom_type == "LineString":
            pts = list(g.coords)
        else:
            continue
        (lo1, la1), (lo2, la2) = pts[0][:2], pts[-1][:2]
        mid = Point((lo1 + lo2) / 2, (la1 + la2) / 2)
        if not (w <= mid.x <= e and s <= mid.y <= n):
            continue
        length = 2 * R * math.asin(math.sqrt(math.sin(math.radians(la2 - la1) / 2) ** 2 +
                                             math.cos(math.radians(la1)) * math.cos(math.radians(la2)) * math.sin(math.radians(lo2 - lo1) / 2) ** 2))
        if length < min_len:
            short += 1
            continue
        name, icao = "", ""
        for ag, an, ai in aerodromes:
            if ag.geom_type in ("Polygon", "MultiPolygon") and ag.contains(mid):
                name, icao = an, ai; break
        if not name:
            cand = [(ag.distance(mid), an, ai) for ag, an, ai in aerodromes]
            if cand:
                d, an, ai = min(cand)
                if d < 0.05:
                    name, icao = an, ai
        status = "closed" if any(k in p for k in ("disused", "abandoned", "disused:aeroway")) or p.get("operational_status") == "closed" else "open"
        rws.append(dict(airport=(f"{icao} " if icao else "") + (name or "unnamed airfield"), runway=p.get("ref", "?"), status=status,
                        lat1=la1, lon1=lo1, lat2=la2, lon2=lo2, length_m=round(length), source=fref(ft)))
    log(f"   {len(rws)} runways of {min_len} m or longer from OSM ({short} shorter strips ignored)")
    for ap, ref, la1, lo1, la2, lo2 in OURAIRPORTS_RUNWAYS:
        mid = ((la1 + la2) / 2, (lo1 + lo2) / 2)
        if not any(r["source"].startswith("osm") and abs((r["lat1"] + r["lat2"]) / 2 - mid[0]) < 0.015 and abs((r["lon1"] + r["lon2"]) / 2 - mid[1]) < 0.015 for r in rws):
            rws.append(dict(airport=ap, runway=ref, status="closed" if "15/33" in ref and "OERY" in ap else "open", lat1=la1, lon1=lo1, lat2=la2, lon2=lo2, length_m=0, source="OurAirports"))
    return rws


# ------------------------------------------------------------------ sources
def osm_objects(pbf, bbox):
    w, s, e, n = bbox
    out = []
    for ft in load_osm(pbf):
        t = ft["properties"]
        if t.get("aeroway") in ("runway", "aerodrome") and not ("building" in t or "height" in t):
            continue
        g = fgeom(ft)
        if g is None or g.is_empty:
            continue
        c = g.representative_point()
        if not (w <= c.x <= e and s <= c.y <= n):
            continue
        h = parse_height(t.get("height")) or parse_height(t.get("est_height"))
        lv = parse_levels(t.get("building:levels"))
        kind = ("building part" if "building:part" in t else "building" if "building" in t else
                "power pylon" if t.get("power") == "tower" else t.get("man_made") or t.get("tower:type") or "structure")
        if kind in ("building", "building part") and not (h or lv):
            continue
        name = t.get("name:en") or t.get("name") or ""
        if h:
            out.append(dict(geom=g, height=h, prio=1, source="OSM height tag", kind=kind, name=name, ref=fref(ft)))
        elif lv:
            out.append(dict(geom=g, height=lv * FLOOR_M, prio=3, source=f"OSM {lv:g} floors x {FLOOR_M} m", kind=kind, name=name, ref=fref(ft)))
        else:
            out.append(dict(geom=g, height=None, prio=9, source="OSM (no height)", kind=kind, name=name, ref=fref(ft)))
    return out


def overture_objects(bbox):
    import duckdb
    w, s, e, n = bbox
    rel = os.environ.get("OVERTURE_RELEASE")
    if not rel:
        try:
            ls = subprocess.run(["aws", "s3", "ls", "--no-sign-request", "s3://overturemaps-us-west-2/release/"],
                                capture_output=True, text=True, timeout=120).stdout
            rels = sorted(re.findall(r"PRE (\d{4}-\d{2}-\d{2}[^/]*)/", ls))
            rel = rels[-1]
        except Exception as ex:
            log(f"  Overture release listing failed ({ex}); skipping Overture")
            return []
    log(f"  Overture release {rel}")
    con = duckdb.connect()
    for sql in ("INSTALL httpfs", "LOAD httpfs", "INSTALL spatial", "LOAD spatial", "SET s3_region='us-west-2'"):
        con.execute(sql)
    path = f"s3://overturemaps-us-west-2/release/{rel}/theme=buildings/type=building/*"
    q = f"""SELECT id, height, num_floors, names.primary AS name, sources[1].dataset AS src, ST_AsWKB(geometry) AS g
            FROM read_parquet('{path}', hive_partitioning=1)
            WHERE bbox.xmin > {w} AND bbox.xmax < {e} AND bbox.ymin > {s} AND bbox.ymax < {n}
              AND (height IS NOT NULL OR num_floors IS NOT NULL)"""
    try:
        rows = con.execute(q).fetchall()
    except Exception as ex:
        log(f"  Overture ST_AsWKB failed ({str(ex)[:120]}); retrying with raw geometry")
        rows = con.execute(q.replace("ST_AsWKB(geometry)", "geometry")).fetchall()
    out = []
    for oid, h, nf, name, src, g in rows:
        try:
            geom = shp_wkb.loads(bytes(g))
        except Exception:
            continue
        if h and h > 0:
            if src and "OpenStreetMap" in src:
                continue      # OSM heights already taken directly from OSM
            out.append(dict(geom=geom, height=float(h), prio=2, source=f"Overture height ({src})", kind="building", name=name or "", ref=f"overture {oid}"))
        elif nf and nf > 0:
            if src and "OpenStreetMap" in src:
                continue
            out.append(dict(geom=geom, height=float(nf) * FLOOR_M, prio=3, source=f"Overture {nf} floors x {FLOOR_M} m ({src})", kind="building", name=name or "", ref=f"overture {oid}"))
    log(f"    {len(rows):,} Overture buildings with height/floors in area, {len(out):,} non-OSM kept")
    return out


def gba_objects(path, bbox):
    out = []
    if not path or not Path(path).exists():
        log("  GBA file not found - skipping")
        return out
    r = shapefile.Reader(str(path))
    flds = [f[0] for f in r.fields[1:]]
    for sr in r.iterShapeRecords():
        rec = dict(zip(flds, sr.record))
        try:
            g = shape(sr.shape.__geo_interface__)
        except Exception:
            continue
        out.append(dict(geom=g, height=float(rec.get("HEIGHT_M") or 0), prio=4, source="GlobalBuildingAtlas LoD1 estimate",
                        kind="building", name="", ref=f"gba {rec.get('SOURCE','')} {rec.get('GBA_ID','')}"))
    log(f"  {len(out):,} GBA buildings (above 15 m) loaded from {path}")
    return out


# ------------------------------------------------------------------ merge
def merge(objs, proj):
    """Group overlapping footprints; per group keep the best-priority source, tallest height within it."""
    polys = [o for o in objs if o["geom"].geom_type in ("Polygon", "MultiPolygon")]
    others = [o for o in objs if o["geom"].geom_type not in ("Polygon", "MultiPolygon")]
    for o in polys:
        o["m"] = proj.geom(o["geom"])
        o["area"] = o["m"].area
    tree = STRtree([o["m"] for o in polys])
    parent = list(range(len(polys)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, o in enumerate(polys):
        for j in tree.query(o["m"]):
            j = int(j)
            if j <= i:
                continue
            p = polys[j]
            try:
                inter = o["m"].intersection(p["m"]).area
            except Exception:
                continue
            if inter > 0.3 * min(o["area"], p["area"]) and min(o["area"], p["area"]) > 0:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(len(polys)):
        groups.setdefault(find(i), []).append(polys[i])
    merged = []
    for g in groups.values():
        with_h = [o for o in g if o["height"]]
        if not with_h:
            best = max(g, key=lambda o: o["area"]); best = dict(best); best["n_sources"] = len(g); merged.append(best); continue
        pr = min(o["prio"] for o in with_h)
        top = max((o for o in with_h if o["prio"] == pr), key=lambda o: o["height"])
        outline = max(g, key=lambda o: o["area"])
        res = dict(top); res["geom"] = outline["geom"] if outline["area"] >= top.get("area", 0) else top["geom"]
        res["name"] = top["name"] or next((o["name"] for o in g if o["name"]), "")
        others_h = sorted({round(o["height"], 1) for o in with_h if o is not top})
        res["alt_heights"] = "; ".join(f"{h:g}" for h in others_h[:6])
        res["n_sources"] = len(g)
        merged.append(res)
    # point structures (masts, pylons): drop if inside a merged footprint that already has a height >= theirs
    for o in others:
        o["n_sources"] = 1; o["alt_heights"] = ""
        merged.append(o)
    return merged


# ------------------------------------------------------------------ output
def write(results, unknown, runways, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    prj = ('GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],'
           'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]')
    cols = ["obj_no", "zone", "height_m", "height_source", "kind", "name", "dist_km", "nearest_runway", "lat", "lon",
            "other_heights_m", "ref"]
    with open(out / "riyadh_obstacles.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in results:
            w.writerow([r["obj_no"], r["zone"], round(r["height"], 1), r["source"], r["kind"], r["name"], round(r["dist_km"], 3),
                        r["runway"], round(r["lat"], 7), round(r["lon"], 7), r.get("alt_heights", ""), r["ref"]])
    with open(out / "pylons_towers_unknown_height_within_near_zone.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["kind", "name", "dist_km", "nearest_runway", "lat", "lon", "ref"])
        for r in unknown:
            w.writerow([r["kind"], r["name"], round(r["dist_km"], 3), r["runway"], round(r["lat"], 7), round(r["lon"], 7), r["ref"]])

    fields = (("OBJ_NO", "N", 9, 0), ("ZONE", "C", 24, 0), ("HEIGHT_M", "F", 9, 1), ("H_SOURCE", "C", 80, 0), ("KIND", "C", 30, 0),
              ("NAME", "C", 80, 0), ("DIST_KM", "F", 9, 3), ("RUNWAY", "C", 80, 0), ("LAT", "F", 12, 7), ("LON", "F", 12, 7), ("REF", "C", 60, 0))

    def rec(r):
        return [r["obj_no"], r["zone"], round(r["height"], 1), r["source"][:80], r["kind"][:30], (r["name"] or "")[:80],
                round(r["dist_km"], 3), r["runway"][:80], round(r["lat"], 7), round(r["lon"], 7), r["ref"][:60]]
    pw = shapefile.Writer(str(out / "obstacles_footprints"), shapeType=shapefile.POLYGON, encoding="utf-8")
    pt = shapefile.Writer(str(out / "obstacles_points"), shapeType=shapefile.POINT, encoding="utf-8")
    for wr in (pw, pt):
        for fd in fields:
            wr.field(*fd)
    for r in results:
        g = r["geom"]
        if g.geom_type in ("Polygon", "MultiPolygon"):
            rings = []
            for p in (g.geoms if g.geom_type == "MultiPolygon" else [g]):
                p = orient(p, sign=-1.0)   # shapefile: clockwise outer ring
                rings.append([list(c) for c in p.exterior.coords])
                rings += [[list(c) for c in i.coords] for i in p.interiors]
            pw.poly(rings); pw.record(*rec(r))
        pt.point(r["lon"], r["lat"]); pt.record(*rec(r))
    pw.close(); pt.close()
    rw = shapefile.Writer(str(out / "runways"), shapeType=shapefile.POLYLINE, encoding="utf-8")
    for fd in (("AIRPORT", "C", 80, 0), ("RUNWAY", "C", 16, 0), ("STATUS", "C", 8, 0), ("LENGTH_M", "N", 7, 0), ("SOURCE", "C", 30, 0)):
        rw.field(*fd)
    for r in runways:
        rw.line([[[r["lon1"], r["lat1"]], [r["lon2"], r["lat2"]]]]); rw.record(r["airport"][:80], r["runway"][:16], r["status"], r["length_m"], r["source"][:30])
    rw.close()
    for nm in ("obstacles_footprints", "obstacles_points", "runways"):
        (out / f"{nm}.prj").write_text(prj); (out / f"{nm}.cpg").write_text("UTF-8")
    with open(out / "runways.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(runways[0].keys())); w.writeheader(); w.writerows(runways)

    esc = lambda s: str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    with open(out / "riyadh_obstacles.kml", "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>Riyadh obstacles</name>'
                '<Style id="near"><LineStyle><color>ff0000ff</color></LineStyle><PolyStyle><color>b30000ff</color></PolyStyle><IconStyle><color>ff0000ff</color></IconStyle></Style>'
                '<Style id="far"><LineStyle><color>ff00a5ff</color></LineStyle><PolyStyle><color>b300a5ff</color></PolyStyle><IconStyle><color>ff00a5ff</color></IconStyle></Style>'
                '<Style id="rw"><LineStyle><color>ffffff00</color><width>4</width></LineStyle></Style>\n<Folder><name>Runways</name>')
        for r in runways:
            f.write(f'<Placemark><name>{esc(r["airport"])} {esc(r["runway"])}</name><styleUrl>#rw</styleUrl><LineString><tessellate>1</tessellate>'
                    f'<coordinates>{r["lon1"]},{r["lat1"]},0 {r["lon2"]},{r["lat2"]},0</coordinates></LineString></Placemark>')
        f.write("</Folder>\n")
        for zid, title in (("near", "Within near zone"), ("far", "Outer zone")):
            f.write(f"<Folder><name>{title}</name>\n")
            for r in results:
                if (r["zone_id"] == "near") != (zid == "near"):
                    continue
                desc = esc(f'{r["height"]:.1f} m ({r["source"]}); {r["kind"]}; {r["dist_km"]:.2f} km from {r["runway"]}; {r["ref"]}')
                f.write(f'<Placemark><name>{esc((r["name"] + " ") if r["name"] else "")}{r["height"]:.0f} m</name><description>{desc}</description><styleUrl>#{zid}</styleUrl>')
                g = r["geom"]
                if g.geom_type in ("Polygon", "MultiPolygon"):
                    f.write("<MultiGeometry>")
                    for p in (g.geoms if g.geom_type == "MultiPolygon" else [g]):
                        cs = " ".join(f"{x:.7f},{y:.7f},{r['height']:.1f}" for x, y in p.exterior.coords)
                        f.write(f'<Polygon><extrude>1</extrude><altitudeMode>relativeToGround</altitudeMode><outerBoundaryIs><LinearRing><coordinates>{cs}</coordinates></LinearRing></outerBoundaryIs></Polygon>')
                    f.write("</MultiGeometry>")
                else:
                    f.write(f'<Point><extrude>1</extrude><altitudeMode>relativeToGround</altitudeMode><coordinates>{r["lon"]:.7f},{r["lat"]:.7f},{r["height"]:.1f}</coordinates></Point>')
                f.write("</Placemark>\n")
            f.write("</Folder>\n")
        f.write("</Document></kml>\n")


# ------------------------------------------------------------------ main
FLOOR_M = 3.5


def main():
    global FLOOR_M
    ap = argparse.ArgumentParser()
    ap.add_argument("--near-km", type=float, default=10); ap.add_argument("--near-min", type=float, default=15)
    ap.add_argument("--far-km", type=float, default=45); ap.add_argument("--far-min", type=float, default=46)
    ap.add_argument("--floor-m", type=float, default=3.5, help="metres per floor for floor-count estimates")
    ap.add_argument("--gba", default="../../results/gba_obstacles.shp")
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--no-overture", action="store_true")
    ap.add_argument("--pbf", default="gcc-states-latest.osm.pbf", help="Geofabrik OSM extract")
    ap.add_argument("--min-runway-m", type=float, default=800)
    a = ap.parse_args()
    FLOOR_M = a.floor_m

    log("1) Runways")
    runways = get_runways(a.pbf, a.min_runway_m)
    for r in runways:
        log(f"   {r['airport']} {r['runway']} {r['status']} {r['length_m']} m [{r['source']}]")
    lats = [r[k] for r in runways for k in ("lat1", "lat2")]; lons = [r[k] for r in runways for k in ("lon1", "lon2")]
    dlat = a.far_km / 111.0; dlon = dlat / math.cos(math.radians(max(abs(min(lats)), abs(max(lats))) + dlat))
    bbox = (min(lons) - dlon, min(lats) - dlat, max(lons) + dlon, max(lats) + dlat)
    log(f"   search box: lon {bbox[0]:.3f}..{bbox[2]:.3f}, lat {bbox[1]:.3f}..{bbox[3]:.3f}")
    rd = RunwayDistance(runways)
    proj = Proj((bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2)

    log("2) OpenStreetMap buildings and structures")
    objs = osm_objects(a.pbf, bbox)
    log(f"   {len(objs):,} OSM objects ({sum(1 for o in objs if o['prio'] == 1):,} with height tag, "
        f"{sum(1 for o in objs if o['prio'] == 3):,} with floors only)")
    if not a.no_overture:
        log("3) Overture Maps buildings")
        try:
            objs += overture_objects(bbox)
        except Exception as ex:
            log(f"   Overture failed: {ex}")
    log("4) GlobalBuildingAtlas (fallback)")
    objs += gba_objects(a.gba, bbox)

    log("5) Merging overlapping footprints")
    merged = merge(objs, proj)
    log(f"   {len(merged):,} distinct objects")

    log("6) Applying runway rules")
    results, unknown = [], []
    for o in merged:
        g = o["geom"]
        c = g.representative_point() if g.geom_type in ("Polygon", "MultiPolygon") else g.centroid
        if g.geom_type == "Polygon":
            pts = list(g.exterior.coords)
        elif g.geom_type == "MultiPolygon":
            pts = [p for poly in g.geoms for p in poly.exterior.coords]
        elif g.geom_type == "LineString":
            pts = list(g.coords)
        else:
            pts = [(c.x, c.y)]
        if len(pts) > 60:
            pts = pts[:: max(1, len(pts) // 60)]
        d, rwn = rd.nearest(pts + [(c.x, c.y)])
        dkm = d / 1000
        o.update(lat=c.y, lon=c.x, dist_km=dkm, runway=rwn)
        h = o["height"]
        if h is None:
            if dkm <= a.near_km and o["kind"] not in ("building", "building part"):
                unknown.append(o)
            continue
        if dkm <= a.near_km and h > a.near_min:
            o["zone"], o["zone_id"] = f"0-{a.near_km:g} km, >{a.near_min:g} m", "near"
        elif a.near_km < dkm <= a.far_km and h >= a.far_min:
            o["zone"], o["zone_id"] = f"{a.near_km:g}-{a.far_km:g} km, >={a.far_min:g} m", "far"
        else:
            continue
        results.append(o)
    results.sort(key=lambda r: (r["zone_id"] != "near", -r["height"]))
    for i, r in enumerate(results, 1):
        r["obj_no"] = i
    write(results, unknown, runways, a.out)

    near = [r for r in results if r["zone_id"] == "near"]; far = [r for r in results if r["zone_id"] == "far"]
    log(f"\nRESULT: {len(results):,} obstacles - {len(near):,} within {a.near_km:g} km above {a.near_min:g} m, "
        f"{len(far):,} at {a.near_km:g}-{a.far_km:g} km of {a.far_min:g} m and above; "
        f"{len(unknown):,} pylons/towers without height inside {a.near_km:g} km listed separately")
    from collections import Counter
    log("   by height source: " + ", ".join(f"{k.split(' (')[0]}: {v}" for k, v in Counter(r["source"].split(" (")[0] for r in results).most_common()))
    log("\nValidation against known structures (nearest selected object within 250 m):")
    for nm, (la, lo, true) in CHECKS.items():
        best = None
        for r in results:
            dd = math.hypot((r["lat"] - la) * 111000, (r["lon"] - lo) * 111000 * math.cos(math.radians(la)))
            if dd < 250 and (best is None or r["height"] > best[0]["height"]):
                best = (r, dd)
        if best:
            r = best[0]
            log(f"   {nm:22s} published {true} m -> {r['height']:.1f} m [{r['source']}] {r['name']!r} {r['zone']}")
        else:
            log(f"   {nm:22s} published {true} m -> not in the selection")
    log("\nTop 25:")
    for r in sorted(results, key=lambda r: -r["height"])[:25]:
        log(f"   {r['height']:6.1f} m  {r['dist_km']:6.2f} km  {r['source'][:40]:40s} {r['kind'][:14]:14s} {r['name'][:40]}")


if __name__ == "__main__":
    main()
