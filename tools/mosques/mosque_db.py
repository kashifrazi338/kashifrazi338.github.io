"""Compact mosque list for the Shadow Height tool: lat, lon, radius_m (half-size of the mosque outline), minaret flag."""
import json, math, sys, gzip, csv, io
fc = json.load(open(sys.argv[1]))
pts = []
for f in fc["features"]:
    g = f["geometry"]; t = g["type"]; props = json.dumps(f["properties"])
    if t == "Point":
        coords = [g["coordinates"]]
    elif t == "LineString":
        coords = g["coordinates"]
    elif t == "Polygon":
        coords = g["coordinates"][0]
    elif t == "MultiPolygon":
        coords = [c for poly in g["coordinates"] for c in poly[0]]
    else:
        continue
    lo = sum(c[0] for c in coords) / len(coords); la = sum(c[1] for c in coords) / len(coords)
    kx = 111320 * math.cos(math.radians(la))
    r = max(math.hypot((c[0] - lo) * kx, (c[1] - la) * 110700) for c in coords)
    pts.append([round(la, 6), round(lo, 6), min(150, round(r)), 1 if "minaret" in props else 0])
# merge duplicates (node + outline of the same mosque): keep the larger outline
pts.sort(key=lambda p: -p[2])
grid, keep = {}, []
for p in pts:
    key = (round(p[0] * 500), round(p[1] * 500)); dup = False
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for q in grid.get((key[0] + dx, key[1] + dy), []):
                if p[3] == 0 and math.hypot((p[0] - q[0]) * 110700, (p[1] - q[1]) * 101000) < max(20, q[2]):
                    dup = True
    if not dup:
        keep.append(p); grid.setdefault(key, []).append(p)
buf = io.StringIO(); w = csv.writer(buf); w.writerow(["lat", "lon", "radius_m", "minaret"]); w.writerows(keep)
with gzip.open(sys.argv[2], "wt") as f:
    f.write(buf.getvalue())
print("mosques", len(pts), "->", len(keep))
