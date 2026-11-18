# -*- coding: utf-8 -*-
"""
Shadow Height Batch - heights of minarets, towers, masts and poles from one satellite image.

For each input point the tool searches nearby for the object's thin shadow streak, measures its
length along the sun direction and converts it to height:  height = shadow length x tan(sun elevation).

Sun elevation, in order of preference:
  1. known heights of a few points in the input (column known_height_m)  -> calibrated from the image
  2. a value you type (from the image metadata)
  3. the image date (+ optional time): solved from the shadow direction (NOAA solar position)

Inputs : GeoTIFF / JP2 / IMG image (ECW: export to GeoTIFF or JP2 first), points as CSV, shapefile or KML.
Outputs: CSV, point + shadow-line shapefiles, KML, review sheet JPG.
"""
import csv
import datetime as dt
import math
import os
import re
import sys
import threading
import traceback
import zipfile

import numpy as np
import cv2
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform as warp_transform
import shapefile
from PIL import Image, ImageDraw, ImageFont

APP = "Shadow Height Batch"
VERSION = "1.0"
D2R, R2D = math.pi / 180.0, 180.0 / math.pi
FT = 3.28084


# ======================================================================= solar position (NOAA)
def sun_position(t_utc, lat, lon):
    jd = (t_utc - dt.datetime(1970, 1, 1)).total_seconds() / 86400.0 + 2440587.5
    jc = (jd - 2451545.0) / 36525.0
    L0 = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360.0
    M = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    C = (math.sin(M * D2R) * (1.914602 - jc * (0.004817 + 0.000014 * jc)) + math.sin(2 * M * D2R) * (0.019993 - 0.000101 * jc)
         + math.sin(3 * M * D2R) * 0.000289)
    om = 125.04 - 1934.136 * jc
    app = L0 + C - 0.00569 - 0.00478 * math.sin(om * D2R)
    obl = 23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60 + 0.00256 * math.cos(om * D2R)
    dec = math.asin(math.sin(obl * D2R) * math.sin(app * D2R))
    y = math.tan(obl * D2R / 2) ** 2
    eot = 4 * R2D * (y * math.sin(2 * L0 * D2R) - 2 * e * math.sin(M * D2R) + 4 * e * y * math.sin(M * D2R) * math.cos(2 * L0 * D2R)
                     - 0.5 * y * y * math.sin(4 * L0 * D2R) - 1.25 * e * e * math.sin(2 * M * D2R))
    utc_min = t_utc.hour * 60 + t_utc.minute + t_utc.second / 60.0
    tst = (utc_min + eot + 4 * lon) % 1440.0
    ha = (tst / 4 + 180 if tst / 4 < 0 else tst / 4 - 180) * D2R
    la = lat * D2R
    cz = max(-1.0, min(1.0, math.sin(la) * math.sin(dec) + math.cos(la) * math.cos(dec) * math.cos(ha)))
    el = 90 - math.acos(cz) * R2D
    az = (math.atan2(math.sin(ha), math.cos(ha) * math.sin(la) - math.tan(dec) * math.cos(la)) * R2D + 180 + 360) % 360
    if el <= 85:
        te = math.tan(el * D2R)
        if el > 5:
            r = 58.1 / te - 0.07 / te ** 3 + 0.000086 / te ** 5
        elif el > -0.575:
            r = 1735 + el * (-518.2 + el * (103.4 + el * (-12.79 + el * 0.711)))
        else:
            r = -20.772 / te
        el += r / 3600.0
    return el, az


def wrap180(a):
    a %= 360.0
    return a - 360.0 if a > 180 else a


def elevation_from_date(day, tz, lat, lon, shadow_az, t_local=None):
    """Sun elevation on 'day' when the sun stood opposite the measured shadow direction."""
    sun_az = (shadow_az + 180) % 360
    start = dt.datetime(day.year, day.month, day.day) - dt.timedelta(hours=tz)
    sols, step = [], dt.timedelta(minutes=2)
    tp, fp = start, wrap180(sun_position(start, lat, lon)[1] - sun_az)
    for i in range(1, 721):
        t = start + step * i
        f = wrap180(sun_position(t, lat, lon)[1] - sun_az)
        if (fp == 0 or (fp < 0) != (f < 0)) and abs(f - fp) < 90:
            a, b, fa = tp, t, fp
            for _ in range(30):
                m = a + (b - a) / 2
                fm = wrap180(sun_position(m, lat, lon)[1] - sun_az)
                if (fa < 0) != (fm < 0):
                    b = m
                else:
                    a, fa = m, fm
            tm = a + (b - a) / 2
            el, _ = sun_position(tm, lat, lon)
            if el > 1:
                sols.append((tm, el))
        tp, fp = t, f
    if not sols:
        return None, None
    if t_local is not None:
        t_utc = dt.datetime(day.year, day.month, day.day, t_local.hour, t_local.minute) - dt.timedelta(hours=tz)
        tm, el = min(sols, key=lambda s: abs((s[0] - t_utc).total_seconds()))
    else:
        tm, el = sols[0]
    return el, (tm + dt.timedelta(hours=tz)).strftime("%H:%M")


# ======================================================================= input points
def parse_coord_text(s):
    """'24.57, 46.84' | '46.84 24.57' | DMS '46°50'39"E 24°34'22"N' -> (lat, lon) or None"""
    u = s.upper().replace("′", "'").replace("″", '"').replace("”", '"').replace("’", "'").replace("º", "°")
    dms = re.findall(r"(\d{1,3})\s*[°D\s]\s*(\d{1,2})\s*['M\s]\s*(\d+(?:\.\d+)?)\s*[\"S]?\s*([NSEW])", u)
    if len(dms) >= 2:
        vals = {}
        for d_, m_, s_, h in dms[:2]:
            v = float(d_) + float(m_) / 60 + float(s_) / 3600
            vals[h] = -v if h in "SW" else v
        la = vals.get("N", vals.get("S"))
        lo = vals.get("E", vals.get("W"))
        return (la, lo) if la is not None and lo is not None else None
    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", u)]
    if len(nums) >= 2:
        a, b = nums[0], nums[1]
        if abs(a) > 90 or ("E" in u.split()[0] if u.split() else False):
            a, b = b, a
        return a, b
    return None


def read_points(path):
    """Returns list of dict(name, lat, lon, known_height or None)."""
    ext = os.path.splitext(path)[1].lower()
    pts = []
    if ext in (".csv", ".txt"):
        with open(path, newline="", encoding="utf-8-sig") as f:
            sample = f.read(4096)
            f.seek(0)
            delim = "\t" if "\t" in sample else (";" if sample.count(";") > sample.count(",") else ",")
            rd = csv.DictReader(f, delimiter=delim)
            cols = {c.lower().strip(): c for c in (rd.fieldnames or [])}
            def col(*names):
                for n in names:
                    if n in cols:
                        return cols[n]
                return None
            cn, cla, clo = col("name", "id", "object", "label"), col("lat", "latitude", "y"), col("lon", "long", "longitude", "x")
            cc, ch = col("coord", "coordinates", "dms", "location"), col("known_height_m", "known_height", "height_known", "reference_height")
            for i, r in enumerate(rd, 1):
                ll = None
                if cla and clo and r.get(cla, "").strip() and r.get(clo, "").strip():
                    try:
                        ll = (float(r[cla]), float(r[clo]))
                    except ValueError:
                        ll = parse_coord_text(r[cla] + " " + r[clo])
                elif cc:
                    ll = parse_coord_text(r[cc])
                if not ll:
                    continue
                kh = None
                if ch and str(r.get(ch, "")).strip():
                    try:
                        kh = float(r[ch])
                    except ValueError:
                        kh = None
                pts.append(dict(name=(r.get(cn) if cn else None) or f"P{i}", lat=ll[0], lon=ll[1], known_height=kh))
    elif ext == ".shp":
        prj = os.path.splitext(path)[0] + ".prj"
        if os.path.exists(prj):
            t = open(prj).read().upper()
            if "PROJCS" in t:
                raise ValueError("The points shapefile is projected. Export it in WGS 1984 (geographic) first.")
        r = shapefile.Reader(path)
        fl = [f[0] for f in r.fields[1:]]
        low = {f.lower(): f for f in fl}
        nf = next((low[k] for k in ("name", "id", "object", "label") if k in low), None)
        hf = next((low[k] for k in ("known_h", "known_heig", "known_height_m", "ref_height") if k in low), None)
        for i, sr in enumerate(r.iterShapeRecords(), 1):
            if not sr.shape.points:
                continue
            x, y = sr.shape.points[0]
            rec = dict(zip(fl, sr.record))
            kh = rec.get(hf) if hf else None
            pts.append(dict(name=str(rec.get(nf)) if nf else f"P{i}", lat=y, lon=x, known_height=float(kh) if kh not in (None, "", 0) else None))
    elif ext in (".kml", ".kmz"):
        if ext == ".kmz":
            with zipfile.ZipFile(path) as z:
                name = next(n for n in z.namelist() if n.lower().endswith(".kml"))
                text = z.read(name).decode("utf-8", "replace")
        else:
            text = open(path, encoding="utf-8", errors="replace").read()
        for i, pm in enumerate(re.findall(r"<Placemark[\s\S]*?</Placemark>", text), 1):
            m = re.search(r"<coordinates>\s*([-\d.]+),([-\d.]+)", pm)
            if not m:
                continue
            nm = re.search(r"<name>([\s\S]*?)</name>", pm)
            pts.append(dict(name=(nm.group(1).strip() if nm else f"P{i}"), lat=float(m.group(2)), lon=float(m.group(1)), known_height=None))
    else:
        raise ValueError("Points must be CSV, TXT, SHP, KML or KMZ.")
    if not pts:
        raise ValueError("No points could be read from " + os.path.basename(path))
    return pts


# ======================================================================= image helpers
class Img:
    def __init__(self, path):
        if path.lower().endswith(".ecw"):
            try:
                self.ds = rasterio.open(path)
            except Exception:
                raise ValueError("ECW needs the Hexagon ECW driver. Export the ECW to GeoTIFF or JP2 in ArcGIS Pro "
                                 "(Data > Export Raster) and select that file.")
        else:
            self.ds = rasterio.open(path)
        d = self.ds
        self.crs = d.crs
        self.geographic = bool(d.crs and d.crs.is_geographic)
        latc = (d.bounds.top + d.bounds.bottom) / 2
        if self.geographic:
            self.mx = abs(d.res[0]) * 111320 * math.cos(latc * D2R)
            self.my = abs(d.res[1]) * 110574
        else:
            f = 1.0
            try:
                f = d.crs.linear_units_factor[1]
            except Exception:
                pass
            self.mx, self.my = abs(d.res[0]) * f, abs(d.res[1]) * f
        self.pm = (self.mx + self.my) / 2
        self.bands = (1, 2, 3) if d.count >= 3 else (1,)

    def to_pixel(self, lat, lon):
        if self.geographic:
            x, y = lon, lat
        else:
            xs, ys = warp_transform("EPSG:4326", self.crs, [lon], [lat])
            x, y = xs[0], ys[0]
        r, c = self.ds.index(x, y)
        return c, r

    def to_latlon(self, col, row):
        x, y = self.ds.xy(row, col)
        if self.geographic:
            return y, x
        lons, lats = warp_transform(self.crs, "EPSG:4326", [x], [y])
        return lats[0], lons[0]

    def read(self, col, row, half):
        c0, r0 = int(col - half), int(row - half)
        w = Window(c0, r0, 2 * half, 2 * half)
        a = self.ds.read(self.bands, window=w, boundless=False) if (c0 >= 0 and r0 >= 0 and c0 + 2 * half <= self.ds.width and r0 + 2 * half <= self.ds.height) else None
        if a is None:
            return None, c0, r0
        a = a.astype(np.float32)
        if a.max() > 255:
            a = a / max(1.0, np.percentile(a, 99.5)) * 255.0
        if a.shape[0] == 1:
            a = np.repeat(a, 3, 0)
        return a, c0, r0


def shadow_mask(L, G=None, R=None):
    small = cv2.resize(np.clip(L, 0, 255).astype(np.uint8), None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    med = cv2.resize(cv2.medianBlur(small, 31), (L.shape[1], L.shape[0])).astype(np.float32)
    m = (L < 80) | ((L < 100) & (L < med - 25))
    if G is not None:
        m &= G >= R - 4
    return m


# ======================================================================= shadow direction
def detect_shadow_azimuth(img, pix_points, log):
    """Direction in which shadows lie next to bright roofs (degrees, object -> shadow tip)."""
    R0 = 30
    acc = np.zeros((2 * R0 + 1, 2 * R0 + 1))
    used = 0
    for (c, r) in pix_points[:: max(1, len(pix_points) // 25)]:
        a, _, _ = img.read(c, r, 400)
        if a is None:
            continue
        L = a.mean(0)
        S = (L < np.percentile(L, 10)).astype(np.float32)
        B = (L > np.percentile(L, 80)).astype(np.float32)
        S -= S.mean(); B -= B.mean()
        C = np.fft.irfft2(np.fft.rfft2(B).conj() * np.fft.rfft2(S), s=S.shape)
        C = np.roll(np.roll(C, R0, 0), R0, 1)[:2 * R0 + 1, :2 * R0 + 1]
        acc += C / (np.abs(C).max() + 1e-9)
        used += 1
    if used == 0:
        return None
    best = None
    for az in np.arange(0, 360, 1.0):
        dx, dy = math.sin(az * D2R) / img.mx, -math.cos(az * D2R) / img.my
        n = math.hypot(dx, dy); dx, dy = dx / n, dy / n
        v = 0.0
        for rr in range(2, 16):
            for s in (1, -1):
                v += s * cv2.getRectSubPix(acc.astype(np.float32), (1, 1), (R0 + s * rr * dx, R0 + s * rr * dy))[0, 0]
        if best is None or v > best[0]:
            best = (v, az)
    log(f"Shadow direction detected from {used} image windows: {best[1]:.0f} deg")
    return best[1]


# ======================================================================= thin-shadow streak search
def find_streak(img, col, row, shaz, search_m, max_m, spread=25, step=2, dist_weight=2.0):
    """Longest thin dark streak along ~shaz starting within search_m of the point.
    Returns dict(length_m, width_m, az, start=(col,row), end=(col,row)) or None."""
    half = int((search_m + max_m + 10) / img.pm)
    a, c0, r0 = img.read(col, row, half)
    if a is None:
        return None
    L = a.mean(0)
    best = None
    for az in np.arange(shaz - spread, shaz + spread + 0.1, step):
        ang = math.degrees(math.atan2(-math.cos(az * D2R) / img.my, math.sin(az * D2R) / img.mx))
        M = cv2.getRotationMatrix2D((half, half), ang, 1.0)
        Lr = cv2.warpAffine(L, M, (2 * half, 2 * half), flags=cv2.INTER_LINEAR, borderValue=255)
        k = max(2, int(round(3.0 / img.pm)))
        contrast = np.minimum(np.roll(Lr, k, 0), np.roll(Lr, -k, 0)) - Lr
        dark = ((Lr < 95) & (contrast > 18)).astype(np.uint8)
        n, lab = cv2.connectedComponents(dark, connectivity=8)
        inv = cv2.invertAffineTransform(M)
        for i in range(1, n):
            ys, xs = np.nonzero(lab == i)
            if len(xs) < 6:
                continue
            length = (xs.max() - xs.min()) * img.pm
            width = len(xs) / max(1, xs.max() - xs.min() + 1) * img.pm
            if length < 4 or width > 3.5 or length > max_m:
                continue
            if np.ptp(ys) * img.pm > max(3.5, 0.15 * length):
                continue
            x0 = xs.min(); y0 = int(np.median(ys[xs <= x0 + 2]))
            dist = math.hypot(x0 - half, y0 - half) * img.pm
            if dist > search_m:
                continue
            x1 = xs.max(); y1 = int(np.median(ys[xs >= x1 - 2]))
            p0 = inv @ np.array([x0, y0, 1.0]); p1 = inv @ np.array([x1, y1, 1.0])
            score = length - dist_weight * dist          # prefer the shadow that starts at the point
            if best is None or score > best["score"]:
                best = dict(score=score, length_m=length, width_m=width, az=float(az), start_dist_m=dist,
                            start=(c0 + p0[0], r0 + p0[1]), end=(c0 + p1[0], r0 + p1[1]))
    return best


# ======================================================================= main batch
def run_batch(opt, log=print, progress=lambda f: None):
    img = Img(opt["image"])
    pts = read_points(opt["points"])
    log(f"{len(pts)} points read; image {img.ds.width} x {img.ds.height} px, {img.mx:.2f} x {img.my:.2f} m per pixel")
    dn, de = opt.get("offset_n", 0.0) or 0.0, opt.get("offset_e", 0.0) or 0.0
    if dn or de:
        log(f"Reference offset: your data is {dn:+.1f} m north, {de:+.1f} m east of the image; inputs shifted onto the image, outputs shifted back")
    pix = []
    for p in pts:
        ila = p["lat"] - dn / 110700.0
        ilo = p["lon"] - de / (111320.0 * math.cos(p["lat"] * D2R))
        c, r = img.to_pixel(ila, ilo)
        p["col"], p["row"] = c, r
        p["inside"] = 0 <= c < img.ds.width and 0 <= r < img.ds.height
        if p["inside"]:
            pix.append((c, r))
    if not pix:
        raise ValueError("None of the points fall inside the image. Check that points and image cover the same area.")
    outside = sum(1 for p in pts if not p["inside"])
    if outside:
        log(f"{outside} points are outside the image and will be reported as such")
    shaz = opt.get("shadow_az")
    if shaz is None:
        shaz = detect_shadow_azimuth(img, pix, log)
        if shaz is None:
            raise ValueError("Could not detect the shadow direction; enter it manually (degrees, object -> shadow tip).")
    # measure
    for i, p in enumerate(pts):
        progress(i / max(1, len(pts)))
        p["streak"] = find_streak(img, p["col"], p["row"], shaz, opt["search_m"], opt["max_shadow_m"]) if p["inside"] else None
    progress(1.0)
    # sun elevation
    known = [p for p in pts if p.get("known_height") and p["streak"]]
    el, el_src = None, ""
    if len(known) >= 2 and opt.get("use_known", True):
        ratios = [p["known_height"] / p["streak"]["length_m"] for p in known]
        el = math.degrees(math.atan(float(np.median(ratios))))
        spread = [math.degrees(math.atan(x)) for x in ratios]
        el_src = f"calibrated from {len(known)} known heights (spread {min(spread):.1f}-{max(spread):.1f} deg)"
    elif opt.get("sun_elev") is not None:
        el = float(opt["sun_elev"]); el_src = "entered by user (metadata)"
    elif opt.get("date") is not None:
        lat_c = float(np.mean([p["lat"] for p in pts if p["inside"]])); lon_c = float(np.mean([p["lon"] for p in pts if p["inside"]]))
        el, tloc = elevation_from_date(opt["date"], opt.get("tz", 3.0), lat_c, lon_c, shaz, opt.get("time"))
        if el is None:
            raise ValueError("On that date the sun never stands opposite the measured shadow direction. Check the date.")
        el_src = f"from date {opt['date']} and shadow direction (sun at {tloc} local)"
    else:
        raise ValueError("Give the sun elevation, the image date, or known heights for at least 2 points.")
    log(f"Sun elevation used: {el:.2f} deg  ({el_src})")
    tan = math.tan(el * D2R)
    rows = []
    for p in pts:
        s = p["streak"]
        r = dict(name=p["name"], input_lat=round(p["lat"], 7), input_lon=round(p["lon"], 7), known_height_m=p.get("known_height"))
        if not p["inside"]:
            r.update(status="outside image")
        elif not s:
            r.update(status="no thin shadow found near this point")
        else:
            h = s["length_m"] * tan
            lo_ = max(0.0, (s["length_m"] - 1) * math.tan((el - 2) * D2R)); hi_ = (s["length_m"] + 1) * math.tan((el + 2) * D2R)
            bl, bo = img.to_latlon(*s["start"]); tl, to = img.to_latlon(*s["end"])
            bl += dn / 110700; bo += de / (111320 * math.cos(bl * D2R)); tl += dn / 110700; to += de / (111320 * math.cos(tl * D2R))
            r.update(status="ok" if (opt.get("min_height") is None or h > opt["min_height"]) else "below minimum height",
                     height_m=round(h, 1), height_low_m=round(lo_, 1), height_high_m=round(hi_, 1), height_ft=round(h * FT, 1),
                     shadow_m=round(s["length_m"], 2), shadow_az=round(s["az"], 1), sun_elev=round(el, 2),
                     lat=round(bl, 7), lon=round(bo, 7), dms=f"{dms(bo, 'E', 'W')} {dms(bl, 'N', 'S')}",
                     tip_lat=round(tl, 7), tip_lon=round(to, 7))
            if p.get("known_height"):
                r["error_vs_known_m"] = round(h - p["known_height"], 1)
        r["sun_source"] = el_src
        rows.append((r, p))
    # plausibility: a streak far longer than the rest is usually a kerb, wall or road edge
    hs = [r["height_m"] for r, _ in rows if r.get("height_m") is not None]
    if len(hs) >= 3:
        med = float(np.median(hs))
        for r, p in rows:
            h = r.get("height_m")
            if h is not None and (h > 2 * med or h > 60) and r["status"] == "ok":
                r["status"] = "check: unusually long shadow - verify on review sheet"
            elif h is not None and p["streak"]["start_dist_m"] > 0.6 * opt["search_m"] and r["status"] == "ok":
                r["status"] = "check: shadow starts far from the point - verify on review sheet"
    write_outputs(rows, img, opt, shaz, el, el_src, log)
    return [r for r, _ in rows]


def dms(v, pos, neg):
    h = pos if v >= 0 else neg
    v = abs(v); d = int(v); m = int((v - d) * 60); s = (v - d - m / 60) * 3600
    if s >= 59.995:
        s = 0; m += 1
    return f"{d}°{m:02d}'{s:05.2f}\"{h}"


PRJ = ('GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],'
       'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]')


def write_outputs(rows, img, opt, shaz, el, el_src, log):
    out = opt["out_dir"]; os.makedirs(out, exist_ok=True)
    base = os.path.join(out, opt.get("out_name", "shadow_heights"))
    cols = ["name", "status", "height_m", "height_low_m", "height_high_m", "height_ft", "dms", "lat", "lon", "shadow_m", "shadow_az",
            "sun_elev", "known_height_m", "error_vs_known_m", "tip_lat", "tip_lon", "input_lat", "input_lon", "sun_source"]
    with open(base + ".csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for r, _ in rows:
            w.writerow(r)
    ok = [(r, p) for r, p in rows if r.get("height_m") is not None]
    for kind in ("points", "shadow_lines"):
        sw = shapefile.Writer(f"{base}_{kind}", shapeType=shapefile.POINT if kind == "points" else shapefile.POLYLINE, encoding="utf-8")
        for fd in (("NAME", "C", 40, 0), ("STATUS", "C", 60, 0), ("HEIGHT_M", "F", 8, 1), ("H_LOW_M", "F", 8, 1), ("H_HIGH_M", "F", 8, 1),
                   ("SHADOW_M", "F", 8, 2), ("SHADOW_AZ", "F", 6, 1), ("SUN_ELEV", "F", 6, 2), ("DMS", "C", 40, 0)):
            sw.field(*fd)
        for r, _ in ok:
            if kind == "points":
                sw.point(r["lon"], r["lat"])
            else:
                sw.line([[[r["lon"], r["lat"]], [r["tip_lon"], r["tip_lat"]]]])
            sw.record(str(r["name"])[:40], r["status"][:60], r["height_m"], r["height_low_m"], r["height_high_m"], r["shadow_m"], r["shadow_az"], r["sun_elev"], r["dms"])
        sw.close()
        open(f"{base}_{kind}.prj", "w").write(PRJ); open(f"{base}_{kind}.cpg", "w").write("UTF-8")
    with open(base + ".kml", "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>Shadow heights</name>'
                '<Style id="ok"><IconStyle><color>ff00ffff</color></IconStyle><LineStyle><color>ffffff00</color><width>3</width></LineStyle></Style>'
                '<Style id="low"><IconStyle><color>ff999999</color></IconStyle><LineStyle><color>ff999999</color><width>2</width></LineStyle></Style>')
        for r, _ in ok:
            nm = str(r["name"]).replace("&", "&amp;").replace("<", "&lt;")
            st = "ok" if r["status"] == "ok" else "low"
            f.write(f'<Placemark><name>{nm} {r["height_m"]:.0f} m</name><description>{r["height_low_m"]}-{r["height_high_m"]} m; shadow {r["shadow_m"]} m at {r["shadow_az"]} deg; sun {r["sun_elev"]} deg</description>'
                    f'<styleUrl>#{st}</styleUrl><MultiGeometry><Point><extrude>1</extrude><altitudeMode>relativeToGround</altitudeMode><coordinates>{r["lon"]},{r["lat"]},{r["height_m"]}</coordinates></Point>'
                    f'<LineString><tessellate>1</tessellate><coordinates>{r["lon"]},{r["lat"]},0 {r["tip_lon"]},{r["tip_lat"]},0</coordinates></LineString></MultiGeometry></Placemark>')
        f.write("</Document></kml>")
    # review sheet
    try:
        font = ImageFont.truetype("arialbd.ttf", 13)
    except Exception:
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 13)
        except Exception:
            font = ImageFont.load_default()
    tiles = []
    for r, p in rows:
        if not p["inside"]:
            continue
        half = int(35 / img.pm)
        a, c0, r0 = img.read(p["col"], p["row"], half)
        if a is None:
            continue
        Z = 260 / (2 * half)
        im = Image.fromarray(np.moveaxis(np.clip(a, 0, 255).astype(np.uint8), 0, -1)).resize((260, 260), Image.LANCZOS)
        dr = ImageDraw.Draw(im)
        x, y = (p["col"] - c0) * Z, (p["row"] - r0) * Z
        dr.ellipse([x - 5, y - 5, x + 5, y + 5], outline=(255, 0, 255), width=2)
        s = p["streak"]
        if s:
            xs, ys = (s["start"][0] - c0) * Z, (s["start"][1] - r0) * Z
            xe, ye = (s["end"][0] - c0) * Z, (s["end"][1] - r0) * Z
            dr.line([(xs, ys), (xe, ye)], fill=(0, 255, 255), width=2)
            dr.ellipse([xs - 4, ys - 4, xs + 4, ys + 4], outline=(255, 255, 0), width=2)
        lab = f"{str(r['name'])[:14]}  " + (f"{r['height_m']:.0f} m" if r.get("height_m") is not None else "no shadow")
        dr.rectangle([0, 0, 260, 17], fill=(0, 0, 0)); dr.text((3, 2), lab, fill=(255, 255, 0), font=font)
        tiles.append(im)
    if tiles:
        cols_ = 6
        sheet = Image.new("RGB", (260 * min(cols_, len(tiles)), 260 * ((len(tiles) + cols_ - 1) // cols_)))
        for i, t in enumerate(tiles):
            sheet.paste(t, ((i % cols_) * 260, (i // cols_) * 260))
        sheet.save(base + "_review.jpg", quality=88)
    n_ok = sum(1 for r, _ in rows if r.get("status") == "ok")
    n_chk = sum(1 for r, _ in rows if str(r.get("status", "")).startswith("check"))
    log(f"Done: {n_ok} of {len(rows)} points measured above the minimum, {n_chk} flagged for checking. Outputs in {out}")
    log("   " + ", ".join(os.path.basename(base) + s for s in (".csv", "_points.shp", "_shadow_lines.shp", ".kml", "_review.jpg")))


# ======================================================================= self test
def selftest(folder):
    """Synthetic 0.5 m image with 6 towers and known shadows; checks the full pipeline."""
    from rasterio.transform import from_origin
    os.makedirs(folder, exist_ok=True)
    W = H = 1200
    img = np.full((3, H, W), 175, np.uint8)
    rng = np.random.default_rng(1)
    img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
    el, shaz = 50.0, 300.0
    towers = [(300, 300, 20), (700, 300, 28), (300, 800, 35), (800, 800, 18), (550, 550, 45), (950, 450, 24)]
    # buildings (bright roofs) with their own shadows, towers with thin shadows
    for (c, r, h) in towers:
        L = h / math.tan(el * D2R) / 0.5
        dx, dy = math.sin(shaz * D2R), -math.cos(shaz * D2R)
        for b in range(3):
            cv2.rectangle(img[b], (c - 3, r - 3), (c + 3, r + 3), 235, -1)
            # 5 px line: OpenCV adds 2.5 px round caps at both ends, so trim them to keep the visible length = L
            cv2.line(img[b], (int(round(c + dx * 6.5)), int(round(r + dy * 6.5))), (int(round(c + dx * (1.5 + L))), int(round(r + dy * (1.5 + L)))), 45, 5)
    x0, y0 = 680000.0, 2722000.0                      # UTM 38N, 0.5 m square pixels
    tr = from_origin(x0, y0, 0.5, 0.5)
    tif = os.path.join(folder, "selftest_image.tif")
    with rasterio.open(tif, "w", driver="GTiff", width=W, height=H, count=3, dtype="uint8", crs="EPSG:32638", transform=tr) as d:
        d.write(img)
    pcsv = os.path.join(folder, "selftest_points.csv")
    xs = [x0 + (c + 0.5) * 0.5 for c, r, h in towers]; ys = [y0 - (r + 0.5) * 0.5 for c, r, h in towers]
    lons, lats = warp_transform("EPSG:32638", "EPSG:4326", xs, ys)
    with open(pcsv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["name", "lat", "lon", "true_height"])
        for i, ((c, r, h), la, lo) in enumerate(zip(towers, lats, lons), 1):
            w.writerow([f"T{i}", la, lo, h])
    rows = run_batch(dict(image=tif, points=pcsv, out_dir=os.path.join(folder, "selftest_output"), search_m=8, max_shadow_m=80,
                          sun_elev=el, shadow_az=None, min_height=None), log=print)
    errs = []
    for (c, r, h), row in zip(towers, rows):
        got = row.get("height_m")
        print(f"  tower true {h} m -> measured {got} m")
        errs.append(abs((got or 0) - h))
    ok = max(errs) <= 3.0
    # same image as JPEG 2000: proves the JP2 driver is present in this build
    jp2 = os.path.join(folder, "selftest_image.jp2")
    try:
        with rasterio.open(tif) as src:
            prof = src.profile.copy(); prof.update(driver="JP2OpenJPEG"); prof.pop("tiled", None); prof.pop("interleave", None)
            prof.pop("blockxsize", None); prof.pop("blockysize", None); prof.pop("compress", None)
            with rasterio.open(jp2, "w", **prof) as dst:
                dst.write(src.read())
        rows2 = run_batch(dict(image=jp2, points=pcsv, out_dir=os.path.join(folder, "selftest_output_jp2"), search_m=8, max_shadow_m=80,
                               sun_elev=el, shadow_az=None, min_height=None), log=print)
        e2 = max(abs((r.get("height_m") or 0) - h) for (c, r_, h), r in zip(towers, rows2))
        print("JP2 read test: max error %.1f m" % e2)
        ok = ok and e2 <= 3.0
    except Exception as ex:
        print("JP2 read test FAILED:", ex); ok = False
    print("SELFTEST", "PASSED" if ok else "FAILED", "max error %.1f m" % max(errs))
    return ok


# ======================================================================= GUI
def gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    root = tk.Tk(); root.title(f"{APP} {VERSION}"); root.geometry("760x640")
    v = {k: tk.StringVar() for k in ("image", "points", "out", "elev", "date", "time", "tz", "az", "search", "maxsh", "minh", "dn", "de")}
    v["tz"].set("3"); v["search"].set("25"); v["maxsh"].set("80"); v["minh"].set("15"); v["dn"].set("0"); v["de"].set("0")
    mode = tk.StringVar(value="known")
    frm = ttk.Frame(root, padding=12); frm.pack(fill="both", expand=True)

    def row(r, label, var, browse=None, width=60):
        ttk.Label(frm, text=label).grid(row=r, column=0, sticky="w", pady=3)
        ttk.Entry(frm, textvariable=var, width=width).grid(row=r, column=1, sticky="we", pady=3)
        if browse:
            ttk.Button(frm, text="Browse...", command=browse).grid(row=r, column=2, padx=4)
    row(0, "Image (GeoTIFF / JP2)", v["image"], lambda: v["image"].set(filedialog.askopenfilename(filetypes=[("Images", "*.tif *.tiff *.jp2 *.img *.ecw"), ("All", "*.*")])))
    row(1, "Points (CSV / SHP / KML)", v["points"], lambda: v["points"].set(filedialog.askopenfilename(filetypes=[("Points", "*.csv *.txt *.shp *.kml *.kmz"), ("All", "*.*")])))
    row(2, "Output folder", v["out"], lambda: v["out"].set(filedialog.askdirectory()))
    lf = ttk.LabelFrame(frm, text="Sun elevation", padding=8); lf.grid(row=3, column=0, columnspan=3, sticky="we", pady=8)
    ttk.Radiobutton(lf, text="Calibrate from known heights in the points file (column known_height_m, 2 or more)", variable=mode, value="known").grid(row=0, column=0, columnspan=4, sticky="w")
    ttk.Radiobutton(lf, text="Sun elevation from metadata (deg):", variable=mode, value="elev").grid(row=1, column=0, sticky="w")
    ttk.Entry(lf, textvariable=v["elev"], width=10).grid(row=1, column=1, sticky="w")
    ttk.Radiobutton(lf, text="Image date (YYYY-MM-DD):", variable=mode, value="date").grid(row=2, column=0, sticky="w")
    ttk.Entry(lf, textvariable=v["date"], width=12).grid(row=2, column=1, sticky="w")
    ttk.Label(lf, text="time HH:MM (optional)").grid(row=2, column=2, sticky="e"); ttk.Entry(lf, textvariable=v["time"], width=7).grid(row=2, column=3, sticky="w")
    ttk.Label(lf, text="time zone (h from UTC)").grid(row=3, column=2, sticky="e"); ttk.Entry(lf, textvariable=v["tz"], width=7).grid(row=3, column=3, sticky="w")
    of = ttk.LabelFrame(frm, text="Options", padding=8); of.grid(row=4, column=0, columnspan=3, sticky="we")
    opts = (("Shadow direction deg (blank = detect)", "az"), ("Search radius around each point (m)", "search"), ("Longest shadow (m)", "maxsh"),
            ("Report heights above (m)", "minh"), ("Your data minus image, north (m)", "dn"), ("Your data minus image, east (m)", "de"))
    for i, (lab, k) in enumerate(opts):
        ttk.Label(of, text=lab).grid(row=i // 2, column=(i % 2) * 2, sticky="w", padx=4, pady=2)
        ttk.Entry(of, textvariable=v[k], width=10).grid(row=i // 2, column=(i % 2) * 2 + 1, sticky="w", pady=2)
    pb = ttk.Progressbar(frm, mode="determinate", maximum=100); pb.grid(row=5, column=0, columnspan=3, sticky="we", pady=8)
    txt = tk.Text(frm, height=12, wrap="word"); txt.grid(row=7, column=0, columnspan=3, sticky="nsew")
    frm.columnconfigure(1, weight=1); frm.rowconfigure(7, weight=1)

    def log(m):
        root.after(0, lambda: (txt.insert("end", m + "\n"), txt.see("end")))

    def num(k, default=None):
        s = v[k].get().strip()
        return float(s) if s else default

    def work():
        try:
            o = dict(image=v["image"].get(), points=v["points"].get(), out_dir=v["out"].get() or os.path.dirname(v["points"].get()),
                     search_m=num("search", 25), max_shadow_m=num("maxsh", 80), min_height=num("minh"), shadow_az=num("az"),
                     offset_n=num("dn", 0), offset_e=num("de", 0), tz=num("tz", 3), use_known=mode.get() == "known")
            if mode.get() == "elev":
                o["sun_elev"] = num("elev")
                if o["sun_elev"] is None:
                    raise ValueError("Enter the sun elevation.")
            if mode.get() == "date":
                o["date"] = dt.date.fromisoformat(v["date"].get().strip())
                if v["time"].get().strip():
                    hh, mm = v["time"].get().strip().split(":")[:2]; o["time"] = dt.time(int(hh), int(mm))
            run_batch(o, log=log, progress=lambda f: root.after(0, lambda: pb.configure(value=f * 100)))
            root.after(0, lambda: messagebox.showinfo(APP, "Finished. See the output folder."))
        except Exception as e:
            log("ERROR: " + str(e)); log(traceback.format_exc())
            root.after(0, lambda: messagebox.showerror(APP, str(e)))
        finally:
            root.after(0, lambda: run_btn.configure(state="normal"))

    def start():
        txt.delete("1.0", "end"); run_btn.configure(state="disabled")
        threading.Thread(target=work, daemon=True).start()
    run_btn = ttk.Button(frm, text="Run batch", command=start); run_btn.grid(row=6, column=0, columnspan=3, pady=4)
    log("Points file: CSV with name, lat, lon (or a 'coord' column with DMS); optional known_height_m.\n"
        "Place each point on or next to the minaret / tower. ECW images: export to GeoTIFF or JP2 first.")
    root.mainloop()


# ======================================================================= CLI
def main():
    import argparse
    if len(sys.argv) == 1:
        return gui()
    ap = argparse.ArgumentParser(description=APP)
    ap.add_argument("--selftest", metavar="FOLDER")
    ap.add_argument("--image"); ap.add_argument("--points"); ap.add_argument("--out", default="shadow_height_output")
    ap.add_argument("--sun-elev", type=float); ap.add_argument("--date"); ap.add_argument("--time"); ap.add_argument("--tz", type=float, default=3)
    ap.add_argument("--shadow-az", type=float); ap.add_argument("--search", type=float, default=25); ap.add_argument("--max-shadow", type=float, default=80)
    ap.add_argument("--min-height", type=float, default=15); ap.add_argument("--offset-n", type=float, default=0); ap.add_argument("--offset-e", type=float, default=0)
    ap.add_argument("--no-known", action="store_true", help="ignore known_height_m column")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest(a.selftest) else 1)
    o = dict(image=a.image, points=a.points, out_dir=a.out, search_m=a.search, max_shadow_m=a.max_shadow, min_height=a.min_height,
             shadow_az=a.shadow_az, offset_n=a.offset_n, offset_e=a.offset_e, tz=a.tz, use_known=not a.no_known, sun_elev=a.sun_elev)
    if a.date:
        o["date"] = dt.date.fromisoformat(a.date)
        if a.time:
            hh, mm = a.time.split(":")[:2]; o["time"] = dt.time(int(hh), int(mm))
    run_batch(o)


if __name__ == "__main__":
    main()
