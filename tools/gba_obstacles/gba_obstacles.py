#!/usr/bin/env python3
"""
GBA Obstacles - tall buildings around runways from the GlobalBuildingAtlas (GBA.LoD1)

Selects buildings from zhu-xlab GlobalBuildingAtlas (Hugging Face: GBA.LoD1 + GBA.ODbLPolygon):
  * within NEAR_KM of any runway centreline and taller than NEAR_MIN metres   (default 10 km, > 15 m)
  * between NEAR_KM and FAR_KM from any runway and at least FAR_MIN metres    (default 10-45 km, >= 46 m)
Writes CSV, polygon shapefile, runway shapefile and a 3D KML (extruded by height).

Install once:   pip install huggingface_hub ijson pyshp
Run:            python gba_obstacles.py
Options:        python gba_obstacles.py --help

Data licence: GBA.LoD1 / GBA.Polygon / GBA.Height are CC BY-NC 4.0, GBA.ODbLPolygon is ODbL.
By downloading you accept the GBA terms of use: https://tubvsig-so2sat-vm1.srv.mwn.de/terms_of_use.html
"""
import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

R_EARTH = 6371008.8
MERC_R = 6378137.0
LOD1_REPO = "zhu-xlab/GBA.LoD1"
ODBL_REPO = "zhu-xlab/GBA.ODbLPolygon"

DEFAULT_RUNWAYS = [
    # airport, runway, status, lat1, lon1, lat2, lon2   (source: OurAirports runways.csv, Nov 2026)
    ("OERK King Khalid International", "15L/33R", "open", 24.976900, 46.702000, 24.944000, 46.722801),
    ("OERK King Khalid International", "15R/33L", "open", 24.970900, 46.674900, 24.938000, 46.695701),
    ("OETH Thumamah", "17/35", "open", 25.230900, 46.637798, 25.195101, 46.644101),
    ("OE53 King Abdul Aziz Military Academy", "15/33", "open", 24.951200, 46.385700, 24.933100, 46.396801),
    ("OERY Riyadh Air Base (old airport)", "01/19", "open", 24.705099, 46.724500, 24.741800, 46.730202),
    ("OERY Riyadh Air Base (old airport)", "15/33", "closed", 24.722700, 46.697800, 24.707800, 46.729301),
    # King Salman Air Base (OESN) and Khashm Alan Air Base: runway ends not in OurAirports.
    # Add them to runways.csv (lat1,lon1 = one end, lat2,lon2 = other end) from Google Earth.
]


# ----------------------------------------------------------------------------- runways
def load_runways(path: Path, include_closed: bool):
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["airport", "runway", "status", "lat1", "lon1", "lat2", "lon2"])
            w.writerows(DEFAULT_RUNWAYS)
        print(f"Created {path} with {len(DEFAULT_RUNWAYS)} Riyadh runways. Edit it to add or remove runways.")
    rws = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                rw = dict(airport=r["airport"].strip(), runway=r["runway"].strip(),
                          status=(r.get("status") or "open").strip().lower(),
                          lat1=float(r["lat1"]), lon1=float(r["lon1"]), lat2=float(r["lat2"]), lon2=float(r["lon2"]))
            except (KeyError, ValueError):
                print(f"  skipped runway row (missing coordinates): {dict(r)}")
                continue
            if rw["status"] == "closed" and not include_closed:
                continue
            rws.append(rw)
    if not rws:
        sys.exit("No runways with coordinates in " + str(path))
    return rws


class RunwayDistance:
    """Distance (m) from a point to the nearest runway centreline segment (local tangent plane per runway)."""

    def __init__(self, runways):
        self.rw = []
        for r in runways:
            lat0 = math.radians((r["lat1"] + r["lat2"]) / 2)
            lon0 = (r["lon1"] + r["lon2"]) / 2
            k = math.cos(lat0)
            ax, ay = self._xy(r["lat1"], r["lon1"], lat0, lon0, k)
            bx, by = self._xy(r["lat2"], r["lon2"], lat0, lon0, k)
            self.rw.append((lat0, lon0, k, ax, ay, bx, by, f'{r["airport"]} {r["runway"]}'))

    @staticmethod
    def _xy(lat, lon, lat0, lon0, k):
        return math.radians(lon - lon0) * k * R_EARTH, (math.radians(lat) - lat0) * R_EARTH

    def nearest(self, pts):
        """pts: list of (lon, lat). Returns (distance_m, runway_name) of the closest point to any runway."""
        best, name = float("inf"), ""
        for lat0, lon0, k, ax, ay, bx, by, nm in self.rw:
            dx, dy = bx - ax, by - ay
            L2 = dx * dx + dy * dy
            for lon, lat in pts:
                px, py = self._xy(lat, lon, lat0, lon0, k)
                t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
                d = math.hypot(px - (ax + t * dx), py - (ay + t * dy))
                if d < best:
                    best, name = d, nm
        return best, name


# ----------------------------------------------------------------------------- tiles
def aoi_bbox(runways, far_km):
    lats = [r[k] for r in runways for k in ("lat1", "lat2")]
    lons = [r[k] for r in runways for k in ("lon1", "lon2")]
    dlat = far_km * 1000 / R_EARTH * 180 / math.pi
    latmax = max(abs(min(lats)), abs(max(lats))) + dlat
    dlon = dlat / math.cos(math.radians(min(latmax, 89)))
    return min(lons) - dlon, min(lats) - dlat, max(lons) + dlon, max(lats) + dlat


def tile_names(bbox):
    """GBA 5-degree tile names, e.g. e045_n25_e050_n20 (west, north, east, south)."""
    w, s, e, n = bbox
    names = []
    lon = math.floor(w / 5) * 5
    while lon < e:
        lat = math.floor(s / 5) * 5
        while lat < n:
            def ew(v):
                return ("e" if v >= 0 else "w") + f"{abs(v):03d}"

            def ns(v):
                return ("n" if v >= 0 else "s") + f"{abs(v):02d}"
            names.append(f"{ew(lon)}_{ns(lat + 5)}_{ew(lon + 5)}_{ns(lat)}")
            lat += 5
        lon += 5
    return names


def download_tiles(tiles, data_dir: Path, api=None, downloader=None):
    """Find and download, per tile: LoD1/<region>/<tile>.json, Polygon/<region>/<tile>.geojson (GBA.LoD1)
    and the matching <tile>.geojson from GBA.ODbLPolygon. Returns list of (tile, json, poly_geojson, odbl_geojson)."""
    if api is None or downloader is None:
        try:
            from huggingface_hub import HfApi, hf_hub_download
        except ImportError:
            sys.exit("Missing package: run  pip install huggingface_hub ijson pyshp")
        api = api or HfApi()
        downloader = downloader or hf_hub_download
    print("Listing GBA files on Hugging Face ...")
    lod1_files = api.list_repo_files(LOD1_REPO, repo_type="dataset")
    odbl_files = api.list_repo_files(ODBL_REPO, repo_type="dataset")
    out = []
    for t in tiles:
        js_all = [f for f in lod1_files if f.startswith("LoD1/") and f.endswith("/" + t + ".json")]
        if not js_all:
            print(f"  tile {t}: no LoD1 heights in GBA (no buildings or not covered) - skipped")
            continue
        # the same tile can exist in several regional folders (e.g. africa and asiawest): use every copy
        for js in js_all:
            region = js.split("/")[1]
            pg = [f for f in lod1_files if f == f"Polygon/{region}/{t}.geojson"]
            od = [f for f in odbl_files if f.endswith(f"{region}/{t}.geojson")]
            got = []
            for repo, lst, sub in ((LOD1_REPO, [js], ""), (LOD1_REPO, pg, ""), (ODBL_REPO, od, "ODbLPolygon")):
                if not lst:
                    got.append(None)
                    continue
                f = lst[0]
                print(f"  downloading {repo}/{f} (large files take a while) ...")
                p = downloader(repo_id=repo, filename=f, repo_type="dataset", local_dir=str(data_dir / sub) if sub else str(data_dir))
                got.append(Path(p))
            out.append((f"{t} [{region}]", got[0], got[1], got[2]))
    return out


def local_tiles(tiles, data_dir: Path):
    """Use files already on disk (same layout as Hugging Face) - for --no-download."""
    out = []
    for t in tiles:
        found = sorted(data_dir.glob(f"LoD1/*/{t}.json"))
        if not found:
            print(f"  tile {t}: no LoD1/<region>/{t}.json under {data_dir} - skipped")
        for js in found:
            region = js.parent.name
            pg = data_dir / "Polygon" / region / f"{t}.geojson"
            od = data_dir / "ODbLPolygon" / region / f"{t}.geojson"
            out.append((f"{t} [{region}]", js, pg if pg.exists() else None, od if od.exists() else None))
    return out


# ----------------------------------------------------------------------------- geometry
def merc_to_ll(x, y):
    return x / MERC_R * 180 / math.pi, (2 * math.atan(math.exp(y / MERC_R)) - math.pi / 2) * 180 / math.pi


def to_ll(coord):
    x, y = float(coord[0]), float(coord[1])
    if abs(x) > 180 or abs(y) > 90:   # GBA polygons are EPSG:3857 even when labelled 4326
        return merc_to_ll(x, y)
    return x, y


def polygons_ll(geom):
    """Returns list of polygons, each a list of rings [(lon,lat),...]."""
    t = geom.get("type")
    c = geom.get("coordinates") or []
    if t == "Polygon":
        polys = [c]
    elif t == "MultiPolygon":
        polys = c
    else:
        return []
    return [[[to_ll(p) for p in ring] for ring in poly] for poly in polys if poly]


def ring_centroid(ring):
    a = cx = cy = 0.0
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
        f = x1 * y2 - x2 * y1
        a += f
        cx += (x1 + x2) * f
        cy += (y1 + y2) * f
    if abs(a) < 1e-18:
        return sum(p[0] for p in ring) / len(ring), sum(p[1] for p in ring) / len(ring)
    return cx / (3 * a), cy / (3 * a)


# ----------------------------------------------------------------------------- main processing
def read_heights(json_path: Path, min_h: float):
    import ijson
    keep = {}
    n = 0
    t0 = time.time()
    with open(json_path, "rb") as f:
        for key, v in ijson.kvitems(f, ""):
            n += 1
            h = v.get("height") if isinstance(v, dict) else None
            if h is not None and float(h) > min_h:
                var = v.get("var")
                keep[key] = (float(h), float(var) if var is not None else None)
            if n % 2_000_000 == 0:
                print(f"    {n:,} heights read, {len(keep):,} above {min_h} m ({time.time() - t0:.0f}s)")
    hs = sorted(v[0] for v in keep.values())
    print(f"    {n:,} heights read, {len(keep):,} above {min_h} m; >=46 m: {sum(1 for h in hs if h >= 46):,}; max {hs[-1] if hs else 0:.1f} m")
    return keep


def scan_polygons(gj_path: Path, heights, bbox, rd: RunwayDistance, a):
    import ijson
    w, s, e, n = bbox
    hits = []
    cnt = matched = inside = 0
    with open(gj_path, "rb") as f:
        for feat in ijson.items(f, "features.item"):
            cnt += 1
            p = feat.get("properties") or {}
            key = str(p.get("source", "")) + str(p.get("id", "")) + str(p.get("region", ""))
            hv = heights.get(key)
            if hv is None:
                continue
            h, var = hv
            matched += 1
            polys = polygons_ll(feat.get("geometry") or {})
            if not polys:
                continue
            outer = polys[0][0]
            clon, clat = ring_centroid(outer[:-1] if outer[0] == outer[-1] else outer)
            if not (w <= clon <= e and s <= clat <= n):
                continue
            inside += 1
            pts = [pt for poly in polys for pt in poly[0]] + [(clon, clat)]
            d, rwname = rd.nearest(pts)
            dkm = d / 1000
            if dkm <= a.near_km and h > a.near_min:
                zone = f"0-{a.near_km:g} km, >{a.near_min:g} m"
            elif a.near_km < dkm <= a.far_km and h >= a.far_min:
                zone = f"{a.near_km:g}-{a.far_km:g} km, >={a.far_min:g} m"
            else:
                continue
            hits.append(dict(source=str(p.get("source", "")), gba_id=str(p.get("id", "")), region=str(p.get("region", "")),
                             height_m=round(h, 2), height_var=None if var is None else round(var, 3),
                             lon=round(clon, 7), lat=round(clat, 7), dist_km=round(dkm, 3),
                             nearest_runway=rwname, zone=zone, polys=polys))
            if cnt % 1_000_000 == 0:
                print(f"    {cnt:,} footprints scanned, {len(hits):,} selected")
    print(f"    {cnt:,} footprints scanned in {gj_path.name}, {matched:,} had a height above the minimum, "
          f"{inside:,} of those inside the 45 km box, {len(hits):,} selected")
    return hits


def write_outputs(hits, runways, out: Path, near_km=10.0):
    import shapefile
    out.mkdir(parents=True, exist_ok=True)
    prj = ('GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],'
           'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]')
    hits.sort(key=lambda r: (r["dist_km"] > near_km, -r["height_m"]))
    cols = ["obj_no", "zone", "height_m", "height_var", "dist_km", "nearest_runway", "lat", "lon", "source", "gba_id", "region"]
    with open(out / "gba_obstacles.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i, r in enumerate(hits, 1):
            r["obj_no"] = i
            w.writerow([r[c] for c in cols])

    sw = shapefile.Writer(str(out / "gba_obstacles"), shapeType=shapefile.POLYGON, encoding="utf-8")
    for fd in (("OBJ_NO", "N", 9, 0), ("ZONE", "C", 30, 0), ("HEIGHT_M", "F", 9, 2), ("H_VAR", "F", 9, 3),
               ("DIST_KM", "F", 9, 3), ("RUNWAY", "C", 60, 0), ("LAT", "F", 12, 7), ("LON", "F", 12, 7),
               ("SOURCE", "C", 20, 0), ("GBA_ID", "C", 40, 0), ("REGION", "C", 20, 0)):
        sw.field(*fd)
    for r in hits:
        parts = [ring for poly in r["polys"] for ring in poly]
        sw.poly([[list(p) for p in ring] for ring in parts])
        sw.record(r["obj_no"], r["zone"], r["height_m"], r["height_var"] or 0, r["dist_km"], r["nearest_runway"][:60],
                  r["lat"], r["lon"], r["source"][:20], r["gba_id"][:40], r["region"][:20])
    sw.close()
    (out / "gba_obstacles.prj").write_text(prj)
    (out / "gba_obstacles.cpg").write_text("UTF-8")

    rw = shapefile.Writer(str(out / "runways"), shapeType=shapefile.POLYLINE, encoding="utf-8")
    rw.field("AIRPORT", "C", 60, 0); rw.field("RUNWAY", "C", 12, 0); rw.field("STATUS", "C", 8, 0)
    for r in runways:
        rw.line([[[r["lon1"], r["lat1"]], [r["lon2"], r["lat2"]]]])
        rw.record(r["airport"][:60], r["runway"], r["status"])
    rw.close()
    (out / "runways.prj").write_text(prj)

    def esc(s):
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    with open(out / "gba_obstacles.kml", "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
                '<name>GBA tall buildings near runways</name>'
                '<Style id="near"><LineStyle><color>ff0000ff</color></LineStyle><PolyStyle><color>b30000ff</color></PolyStyle></Style>'
                '<Style id="far"><LineStyle><color>ff00a5ff</color></LineStyle><PolyStyle><color>b300a5ff</color></PolyStyle></Style>'
                '<Style id="rw"><LineStyle><color>ffffff00</color><width>4</width></LineStyle></Style>\n')
        f.write("<Folder><name>Runways</name>")
        for r in runways:
            f.write(f'<Placemark><name>{esc(r["airport"])} {esc(r["runway"])}</name><styleUrl>#rw</styleUrl><LineString><tessellate>1</tessellate>'
                    f'<coordinates>{r["lon1"]},{r["lat1"]},0 {r["lon2"]},{r["lat2"]},0</coordinates></LineString></Placemark>')
        f.write("</Folder>\n")
        for zone_id, title in (("near", "Within near zone"), ("far", "Outer zone")):
            f.write(f"<Folder><name>{title}</name>\n")
            for r in hits:
                if (r["dist_km"] <= near_km) != (zone_id == "near"):
                    continue
                f.write(f'<Placemark><name>{r["height_m"]:.0f} m</name><description>{esc(r["zone"])}; {r["dist_km"]} km from {esc(r["nearest_runway"])}; '
                        f'GBA {esc(r["source"])} {esc(r["gba_id"])}</description><styleUrl>#{zone_id}</styleUrl><MultiGeometry>')
                for poly in r["polys"]:
                    f.write('<Polygon><extrude>1</extrude><altitudeMode>relativeToGround</altitudeMode>')
                    for j, ring in enumerate(poly):
                        tag = "outerBoundaryIs" if j == 0 else "innerBoundaryIs"
                        coords = " ".join(f"{x:.7f},{y:.7f},{r['height_m']}" for x, y in ring)
                        f.write(f"<{tag}><LinearRing><coordinates>{coords}</coordinates></LinearRing></{tag}>")
                    f.write("</Polygon>")
                f.write("</MultiGeometry></Placemark>\n")
            f.write("</Folder>\n")
        f.write("</Document></kml>\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Tall buildings around runways from GlobalBuildingAtlas (GBA.LoD1).")
    ap.add_argument("--runways", type=Path, default=Path("runways.csv"), help="runway CSV (created on first run)")
    ap.add_argument("--data", type=Path, default=Path("gba_data"), help="download / data folder")
    ap.add_argument("--out", type=Path, default=Path("gba_output"), help="output folder")
    ap.add_argument("--near-km", type=float, default=10.0)
    ap.add_argument("--near-min", type=float, default=15.0, help="near zone: height must be ABOVE this (m)")
    ap.add_argument("--far-km", type=float, default=45.0)
    ap.add_argument("--far-min", type=float, default=46.0, help="outer zone: height must be AT LEAST this (m)")
    ap.add_argument("--include-closed", action="store_true", help="also use runways marked closed")
    ap.add_argument("--no-download", action="store_true", help="use files already in --data")
    a = ap.parse_args(argv)

    runways = load_runways(a.runways, a.include_closed)
    print(f"Runways used: {len(runways)}")
    for r in runways:
        print(f"   {r['airport']} {r['runway']} ({r['status']})")
    bbox = aoi_bbox(runways, a.far_km)
    tiles = tile_names(bbox)
    print(f"Area: lon {bbox[0]:.3f}..{bbox[2]:.3f}, lat {bbox[1]:.3f}..{bbox[3]:.3f}; GBA tiles: {', '.join(tiles)}")
    files = local_tiles(tiles, a.data) if a.no_download else download_tiles(tiles, a.data)
    rd = RunwayDistance(runways)
    hits = []
    for t, js, pg, od in files:
        print(f"Tile {t}: reading heights from {js.name}")
        heights = read_heights(js, min(a.near_min, a.far_min))
        for gj in (od, pg):
            if gj is None:
                continue
            print(f"  scanning footprints {gj}")
            hits += scan_polygons(gj, heights, bbox, rd, a)
    write_outputs(hits, runways, a.out, a.near_km)
    near = sum(1 for r in hits if r["dist_km"] <= a.near_km)
    print(f"\nDone: {len(hits):,} buildings  ({near:,} within {a.near_km:g} km above {a.near_min:g} m, "
          f"{len(hits) - near:,} at {a.near_km:g}-{a.far_km:g} km of {a.far_min:g} m and above)")
    print(f"Output: {a.out.resolve()}  (gba_obstacles.csv / .shp / .kml, runways.shp)")
    return hits


if __name__ == "__main__":
    main()
