"""Mosques and minarets in a box from the Geofabrik OSM extract (osmium). Writes mosques.geojson."""
import json, subprocess, sys
pbf, w, s, e, n, out = sys.argv[1], *map(float, sys.argv[2:6]), sys.argv[6]
run = lambda c: (print("$", " ".join(c), flush=True), subprocess.run(c, check=True))
run(["osmium", "extract", "-b", f"{w},{s},{e},{n}", pbf, "-o", "box.pbf", "--overwrite", "-s", "smart"])
run(["osmium", "tags-filter", "box.pbf", "nwr/amenity=place_of_worship", "nwr/building=mosque",
     "nwr/tower:type=minaret", "nwr/building:part=minaret", "nwr/man_made=minaret", "-o", "f.pbf", "--overwrite"])
run(["osmium", "export", "f.pbf", "-f", "geojson", "-a", "type,id", "-o", out, "--overwrite"])
fc = json.load(open(out))
keep = [f for f in fc["features"] if f["properties"].get("religion", "muslim") == "muslim" or f["properties"].get("building") == "mosque"
        or "minaret" in json.dumps(f["properties"])]
fc["features"] = keep
json.dump(fc, open(out, "w"))
from collections import Counter
print("DONE", len(keep), Counter(f["geometry"]["type"] for f in keep), sum("minaret" in json.dumps(f["properties"]) for f in keep), "minaret-tagged")
