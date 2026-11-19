"""Fetch Esri World Imagery tiles (z19) around each point, stitch into small GeoTIFFs (EPSG:3857), plus imagery metadata."""
import csv, math, io, json, os, sys, time, urllib.request
import numpy as np, rasterio
from rasterio.transform import from_origin
from PIL import Image
Z = 19; HALF_M = 70.0; R = 6378137.0
URL = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
os.makedirs("out", exist_ok=True)
def merc(lat, lon):
    return R * math.radians(lon), R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
ts = 2 * math.pi * R / 2 ** Z   # tile size in metres
O = math.pi * R
def get(url):
    for k in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research test)"})
            return urllib.request.urlopen(req, timeout=30).read()
        except Exception as e:
            time.sleep(2 + 3 * k); err = e
    raise err
meta = {}
for r in csv.DictReader(open(sys.argv[1])):
    lat, lon = float(r["lat"]), float(r["lon"])
    x, y = merc(lat, lon)
    k = HALF_M / math.cos(math.radians(lat))           # mercator metres for HALF_M ground metres
    tx0, tx1 = int((x - k + O) // ts), int((x + k + O) // ts)
    ty0, ty1 = int((O - (y + k)) // ts), int((O - (y - k)) // ts)
    mos = np.zeros(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256, 3), np.uint8)
    for ty in range(ty0, ty1 + 1):
        for tx in range(tx0, tx1 + 1):
            im = Image.open(io.BytesIO(get(URL.format(z=Z, y=ty, x=tx)))).convert("RGB")
            mos[(ty - ty0) * 256:(ty - ty0 + 1) * 256, (tx - tx0) * 256:(tx - tx0 + 1) * 256] = np.array(im)
    T = from_origin(tx0 * ts - O, O - ty0 * ts, ts / 256, ts / 256)
    with rasterio.open(f"out/{r['name']}.tif", "w", driver="GTiff", width=mos.shape[1], height=mos.shape[0], count=3, dtype="uint8",
                       crs="EPSG:3857", transform=T, compress="jpeg", photometric="ycbcr") as d:
        d.write(np.moveaxis(mos, -1, 0))
    for murl in ("https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/identify?geometry={lon},{lat}&geometryType=esriGeometryPoint&sr=4326&layers=all&tolerance=1&mapExtent={l},{b},{rr},{t}&imageDisplay=400,400,96&returnGeometry=false&f=json",
                 "https://metadata.maptiles.arcgis.com/arcgis/rest/services/World_Imagery_Metadata/MapServer/identify?geometry={lon},{lat}&geometryType=esriGeometryPoint&sr=4326&layers=all&tolerance=1&mapExtent={l},{b},{rr},{t}&imageDisplay=400,400,96&returnGeometry=false&f=json"):
        try:
            u = murl.format(lon=lon, lat=lat, l=lon - .001, b=lat - .001, rr=lon + .001, t=lat + .001)
            meta.setdefault(r["name"], []).append(json.loads(get(u)))
        except Exception as e:
            meta.setdefault(r["name"], []).append({"error": str(e)})
    print(r["name"], mos.shape, flush=True)
json.dump(meta, open("out/metadata.json", "w"))
