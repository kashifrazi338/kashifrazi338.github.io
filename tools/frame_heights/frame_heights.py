#!/usr/bin/env python3
"""Every building inside a small box with its best available height (no height threshold).
Sources (best first): OSM height tag > OSM floors x FLOOR_M > Overture height/floors (non-OSM) > GlobalBuildingAtlas LoD1.
Usage: python frame_heights.py --bbox W S E N --pbf gcc-states-latest.osm.pbf --out frame_out
"""
import argparse, csv, json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "riyadh_obstacles"))
import riyadh_obstacles as ro  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gba_obstacles"))
import gba_obstacles as gba  # noqa: E402


def gba_all(bbox, data_dir):
    """All GBA buildings in bbox (any height), downloading the tile(s) from Hugging Face."""
    from shapely.geometry import Polygon, MultiPolygon
    import ijson
    tiles = gba.tile_names(bbox)
    files = gba.download_tiles(tiles, Path(data_dir))
    w, s, e, n = bbox
    out = []
    for t, js, pg, od in files:
        heights = {}
        with open(js, "rb") as f:
            for k, v in ijson.kvitems(f, ""):
                if isinstance(v, dict) and v.get("height") is not None:
                    heights[k] = float(v["height"])
        ro.log(f"   {t}: {len(heights):,} GBA heights")
        for gj in (od, pg):
            if gj is None:
                continue
            with open(gj, "rb") as f:
                for feat in ijson.items(f, "features.item"):
                    p = feat.get("properties") or {}
                    key = str(p.get("source", "")) + str(p.get("id", "")) + str(p.get("region", ""))
                    polys = gba.polygons_ll(feat.get("geometry") or {})
                    if not polys:
                        continue
                    ring = polys[0][0]
                    cx = sum(q[0] for q in ring) / len(ring); cy = sum(q[1] for q in ring) / len(ring)
                    if not (w <= cx <= e and s <= cy <= n):
                        continue
                    shp = [Polygon(pp[0], pp[1:]) for pp in polys if len(pp[0]) >= 4]
                    if not shp:
                        continue
                    g = shp[0] if len(shp) == 1 else MultiPolygon(shp)
                    h = heights.get(key)
                    out.append(dict(geom=g if g.is_valid else g.buffer(0), height=h, prio=4 if h else 9,
                                    source="GlobalBuildingAtlas LoD1 estimate" if h else "GBA footprint (no height)",
                                    kind="building", name="", ref=f"gba {p.get('source', '')} {p.get('id', '')}"))
    ro.log(f"   {len(out):,} GBA footprints in box")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", nargs=4, type=float, required=True, metavar=("W", "S", "E", "N"))
    ap.add_argument("--pbf", default="gcc-states-latest.osm.pbf")
    ap.add_argument("--floor-m", type=float, default=3.5)
    ap.add_argument("--data", default="gba_data")
    ap.add_argument("--out", type=Path, default=Path("frame_out"))
    a = ap.parse_args()
    ro.FLOOR_M = a.floor_m
    bbox = tuple(a.bbox)
    ro.REGION = (bbox[0] - 0.05, bbox[1] - 0.05, bbox[2] + 0.05, bbox[3] + 0.05)
    proj = ro.Proj((bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2)
    ro.log("OSM"); objs = ro.osm_objects(a.pbf, bbox)
    ro.log("Overture")
    try:
        objs += ro.overture_objects(bbox)
    except Exception as ex:
        ro.log(f"   Overture failed: {ex}")
    ro.log("GBA"); objs += gba_all(bbox, a.data)
    merged = ro.merge(objs, proj)
    a.out.mkdir(parents=True, exist_ok=True)
    feats = []
    with open(a.out / "frame_buildings.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["obj_no", "height_m", "height_source", "kind", "name", "lat", "lon", "footprint_m2", "other_heights_m", "ref"])
        i = 0
        for o in sorted(merged, key=lambda o: -(o["height"] or 0)):
            g = o["geom"]
            if g.geom_type not in ("Polygon", "MultiPolygon", "Point"):
                continue
            c = g.representative_point() if g.geom_type != "Point" else g
            i += 1
            area = proj.geom(g).area if g.geom_type != "Point" else 0
            w.writerow([i, "" if o["height"] is None else round(o["height"], 1), o["source"], o["kind"], o["name"],
                        round(c.y, 7), round(c.x, 7), round(area, 1), o.get("alt_heights", ""), o["ref"]])
            feats.append({"type": "Feature", "properties": {"obj_no": i, "height_m": None if o["height"] is None else round(o["height"], 1),
                          "source": o["source"], "kind": o["kind"], "name": o["name"], "ref": o["ref"]},
                          "geometry": json.loads(json.dumps(ro.mapping(g)))})
    with open(a.out / "frame_buildings.geojson", "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)
    from collections import Counter
    ro.log(f"DONE {len(feats):,} objects; " + ", ".join(f"{k}: {v}" for k, v in Counter(x['properties']['source'].split(' (')[0].split(' x ')[0] for x in feats).most_common(8)))


if __name__ == "__main__":
    main()
