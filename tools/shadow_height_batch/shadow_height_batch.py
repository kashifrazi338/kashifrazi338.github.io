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
VERSION = "2.0"
D2R, R2D = math.pi / 180.0, 180.0 / math.pi
FT = 3.28084
# automatic modes: minaret / mast shadows are 1.4-2.5 m wide; gaps between buildings and kerb lines are thinner
AUTO_MIN_WIDTH, AUTO_OK_WIDTH = 1.3, 1.4


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
        # read the part that lies inside the image; pad the rest bright (never looks like shadow)
        ic0, ir0 = max(0, c0), max(0, r0)
        ic1, ir1 = min(self.ds.width, c0 + 2 * half), min(self.ds.height, r0 + 2 * half)
        if ic1 - ic0 < half or ir1 - ir0 < half:
            return None, c0, r0
        part = self.ds.read(self.bands, window=Window(ic0, ir0, ic1 - ic0, ir1 - ir0)).astype(np.float32)
        if part.max() > 255:
            part = part / max(1.0, np.percentile(part, 99.5)) * 255.0
        part[:, (part <= 0).all(0)] = 255.0                    # no-data pixels
        a = np.full((part.shape[0], 2 * half, 2 * half), 255.0, np.float32)
        a[:, ir0 - r0:ir1 - r0, ic0 - c0:ic1 - c0] = part
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
def find_streak(img, col, row, shaz, search_m, max_m, spread=25, step=2, dist_weight=2.0, collect=False, min_len=4.0):
    """Longest thin dark streak along ~shaz starting within search_m of the point.
    Returns dict(length_m, width_m, az, start=(col,row), end=(col,row)) or None."""
    half = int((search_m + max_m + 10) / img.pm)
    a, c0, r0 = img.read(col, row, half)
    if a is None:
        return [] if collect else None
    L = a.mean(0)
    best, allc = None, []
    for az in np.arange(shaz - spread, shaz + spread + 0.1, step):
        ang = math.degrees(math.atan2(-math.cos(az * D2R) / img.my, math.sin(az * D2R) / img.mx))
        M = cv2.getRotationMatrix2D((half, half), ang, 1.0)
        Lr = cv2.warpAffine(L, M, (2 * half, 2 * half), flags=cv2.INTER_LINEAR, borderValue=255)
        k = max(2, int(round(3.0 / img.pm)))
        contrast = np.minimum(np.roll(Lr, k, 0), np.roll(Lr, -k, 0)) - Lr
        dark = ((Lr < 95) & (contrast > 18)).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
        inv = cv2.invertAffineTransform(M)
        for i in range(1, n):
            bx, by, bw, bh, area = st[i]
            if area < 6 or (bw - 1) * img.pm < min_len or (bh - 1) * img.pm > max(3.5, 0.15 * (bw - 1) * img.pm):
                continue                                  # cheap reject from the bounding box (same tests as below)
            ys, xs = np.nonzero(lab[by:by + bh, bx:bx + bw] == i)
            ys = ys + by; xs = xs + bx
            if len(xs) < 6:
                continue
            length = (xs.max() - xs.min()) * img.pm
            width = len(xs) / max(1, xs.max() - xs.min() + 1) * img.pm
            if length < min_len or width > 3.5 or length > max_m:
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
            cand = dict(score=score, length_m=length, width_m=width, az=float(az), start_dist_m=dist,
                        start=(c0 + p0[0], r0 + p0[1]), end=(c0 + p1[0], r0 + p1[1]),
                        contrast=float(contrast[ys, xs].mean()))
            if collect:
                allc.append(cand)
            elif best is None or score > best["score"]:
                best = cand
    return allc if collect else best


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
                     tip_lat=round(tl, 7), tip_lon=round(to, 7), width_m=round(s["width_m"], 2))
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
            "sun_elev", "known_height_m", "error_vs_known_m", "tip_lat", "tip_lon", "input_lat", "input_lon", "width_m", "mosque_lat", "mosque_lon", "sun_source"]
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
            st = "ok" if (r["status"] == "ok" or r["status"].startswith("likely")) else "low"
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
    from collections import Counter
    cnt = Counter(str(r.get("status", "")).split(" - ")[0].split(":")[0] for r, _ in rows)
    log(f"Done: {len(rows)} rows (" + ", ".join(f"{v} {k}" for k, v in cnt.most_common()) + f"). Outputs in {out}")
    log("   " + ", ".join(os.path.basename(base) + s for s in (".csv", "_points.shp", "_shadow_lines.shp", ".kml", "_review.jpg")))


# ======================================================================= automatic scan (no points needed)
def read_aoi(path, img, buffer_m=0, dn=0.0, de=0.0):
    """Bounding box (col0,row0,col1,row1) in image pixels of the features in a KML/KMZ/SHP/CSV file."""
    lls = []
    ext = os.path.splitext(path)[1].lower()
    if ext in (".kml", ".kmz"):
        if ext == ".kmz":
            with zipfile.ZipFile(path) as z:
                text = z.read(next(n for n in z.namelist() if n.lower().endswith(".kml"))).decode("utf-8", "replace")
        else:
            text = open(path, encoding="utf-8", errors="replace").read()
        for block in re.findall(r"<coordinates>([\s\S]*?)</coordinates>", text):
            for tok in block.split():
                v = tok.split(",")
                if len(v) >= 2:
                    lls.append((float(v[1]), float(v[0])))
    elif ext == ".shp":
        prj = os.path.splitext(path)[0] + ".prj"
        if os.path.exists(prj) and "PROJCS" in open(prj).read().upper():
            raise ValueError("The area shapefile is projected. Export it in WGS 1984 (geographic) first.")
        for sh in shapefile.Reader(path).shapes():
            lls += [(y, x) for x, y in sh.points]
    else:
        lls = [(p["lat"], p["lon"]) for p in read_points(path)]
    if not lls:
        raise ValueError("No coordinates found in the area file.")
    px = [img.to_pixel(la - dn / 110700.0, lo - de / (111320.0 * math.cos(la * D2R))) for la, lo in lls]
    b = buffer_m / img.pm
    c0 = max(0, int(min(c for c, r in px) - b)); c1 = min(img.ds.width, int(max(c for c, r in px) + b) + 1)
    r0 = max(0, int(min(r for c, r in px) - b)); r1 = min(img.ds.height, int(max(r for c, r in px) + b) + 1)
    if c1 <= c0 or r1 <= r0:
        raise ValueError("The area does not overlap the image.")
    return c0, r0, c1, r1


def _image_scale(img):
    """One brightness scale for the whole image (16-bit / 11-bit imagery -> 0..255)."""
    if img.ds.dtypes[0] == "uint8":
        return 1.0
    f = max(1, int(max(img.ds.width, img.ds.height) / 2000))
    a = img.ds.read(img.bands, out_shape=(len(img.bands), max(1, img.ds.height // f), max(1, img.ds.width // f))).astype(np.float32)
    return 255.0 / max(1.0, float(np.percentile(a[a > 0], 99.5)) if (a > 0).any() else 1.0)


def scan_tile(L, img, shaz, spread, step, min_len_m, max_m, core):
    """All thin dark streaks in one tile. L = brightness array; core = (x0,y0,x1,y1) tile-pixel box where a streak
    must start (so that overlapping tiles do not report the same object twice). Returns list of dicts in tile pixels."""
    H, W = L.shape
    out = []
    k = max(2, int(round(3.0 / img.pm)))
    min_px = min_len_m / img.pm
    diag = int(math.ceil(math.hypot(W, H)))
    for az in np.arange(shaz - spread, shaz + spread + 0.1, step):
        ang = math.degrees(math.atan2(-math.cos(az * D2R) / img.my, math.sin(az * D2R) / img.mx))
        M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
        M[0, 2] += (diag - W) / 2; M[1, 2] += (diag - H) / 2      # keep the whole tile inside the rotated canvas
        Lr = cv2.warpAffine(L, M, (diag, diag), flags=cv2.INTER_LINEAR, borderValue=255)
        contrast = np.minimum(np.roll(Lr, k, 0), np.roll(Lr, -k, 0)) - Lr
        dark = ((Lr < 95) & (contrast > 18)).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
        inv = cv2.invertAffineTransform(M)
        for i in range(1, n):
            x, y, w, h, area = st[i]
            if w < min_px or w * img.pm > max_m or area < 6:
                continue
            length = (w - 1) * img.pm
            if h * img.pm > max(3.5, 0.15 * length) + img.pm:
                continue
            width = area / w * img.pm
            if width > 3.5 or width < AUTO_MIN_WIDTH:
                continue
            sub = lab[y:y + h, x:x + w] == i
            ys, xs = np.nonzero(sub)
            xs = xs + x; ys = ys + y
            x0 = xs.min(); y0 = float(np.median(ys[xs <= x0 + 2]))
            x1 = xs.max(); y1 = float(np.median(ys[xs >= x1 - 2]))
            # gaps: a real shadow is continuous; a dashed lane marking is not
            fill = len(np.unique(xs)) / float(w)
            if fill < 0.8:
                continue
            # the sun side (just before the start) must not continue as a dark line (kerbs, lane markings, fences)
            yy = int(round(y0)); a0 = int(max(0, x0 - 6.0 / img.pm)); a1 = int(max(0, x0 - 1.0 / img.pm))
            if a1 > a0:
                before = dark[max(0, yy - 1):yy + 2, a0:a1].any(0).mean()
                if before > 0.35:
                    continue
            p0 = inv @ np.array([x0, y0, 1.0]); p1 = inv @ np.array([x1, y1, 1.0])
            if not (core[0] <= p0[0] < core[2] and core[1] <= p0[1] < core[3]):
                continue
            out.append(dict(length_m=length, width_m=width, az=float(az), start=(p0[0], p0[1]), end=(p1[0], p1[1]),
                            contrast=float(contrast[ys, xs].mean()), start_dist_m=0.0))
    return out


def dedupe_streaks(cands, pm):
    """Same object seen at several angles / in overlapping tiles -> keep the longest."""
    cands.sort(key=lambda s: -s["length_m"])
    keep = []
    for s in cands:
        sx, sy = s["start"]
        dup = False
        for k in keep:
            kx, ky = k["start"]; ex, ey = k["end"]
            if math.hypot(sx - kx, sy - ky) * pm < 5:
                dup = True; break
            # start lies on an already kept (longer) streak -> part of the same shadow
            vx, vy = ex - kx, ey - ky; ll = vx * vx + vy * vy
            if ll > 0:
                t = max(0.0, min(1.0, ((sx - kx) * vx + (sy - ky) * vy) / ll))
                if math.hypot(sx - (kx + t * vx), sy - (ky + t * vy)) * pm < 3:
                    dup = True; break
        if not dup:
            keep.append(s)
    return keep


def sun_for_scan(opt, img, shaz, dn, de, la_c, lo_c, log):
    """Sun elevation for the scan modes: known-height points -> metadata value -> date."""
    # sun elevation
    el, el_src = None, ""
    if opt.get("points") and opt.get("use_known", True):
        kp = [p for p in read_points(opt["points"]) if p.get("known_height")]
        ratios = []
        for p in kp:
            c, r = img.to_pixel(p["lat"] - dn / 110700.0, p["lon"] - de / (111320.0 * math.cos(p["lat"] * D2R)))
            if 0 <= c < img.ds.width and 0 <= r < img.ds.height:
                s = find_streak(img, c, r, shaz, opt.get("search_m", 25), opt["max_shadow_m"])
                if s:
                    ratios.append(p["known_height"] / s["length_m"])
        if len(ratios) >= 2:
            el = math.degrees(math.atan(float(np.median(ratios))))
            el_src = f"calibrated from {len(ratios)} known heights"
        elif kp:
            log(f"Only {len(ratios)} of {len(kp)} known-height points gave a shadow; not enough to calibrate.")
    if el is None and opt.get("sun_elev") is not None:
        el = float(opt["sun_elev"]); el_src = "entered by user (metadata)"
    if el is None and opt.get("date") is not None:
        el, tloc = elevation_from_date(opt["date"], opt.get("tz", 3.0), la_c, lo_c, shaz, opt.get("time"))
        if el is None:
            raise ValueError("On that date the sun never stands opposite the measured shadow direction. Check the date.")
        el_src = f"from date {opt['date']} and shadow direction (sun at {tloc} local)"
    if el is None:
        raise ValueError("Give the sun elevation, the image date, or a points file with 2+ known heights.")
    return el, el_src


def run_scan(opt, log=print, progress=lambda f: None, stop=lambda: False):
    """Scan the whole image (or an area) for thin shadows of minarets, towers, masts, poles."""
    img = Img(opt["image"])
    log(f"Image {img.ds.width} x {img.ds.height} px, {img.mx:.2f} x {img.my:.2f} m per pixel")
    dn, de = opt.get("offset_n", 0.0) or 0.0, opt.get("offset_e", 0.0) or 0.0
    if opt.get("area"):
        C0, R0, C1, R1 = read_aoi(opt["area"], img, 0, dn, de)
        log(f"Scanning only the area in {os.path.basename(opt['area'])}: {C1 - C0} x {R1 - R0} px")
    else:
        C0, R0, C1, R1 = 0, 0, img.ds.width, img.ds.height
    scale = _image_scale(img)
    # shadow direction from a grid of windows inside the scan area
    shaz = opt.get("shadow_az")
    if shaz is None:
        grid = [(C0 + (C1 - C0) * (i + 0.5) / 5, R0 + (R1 - R0) * (j + 0.5) / 5) for i in range(5) for j in range(5)]
        shaz = detect_shadow_azimuth(img, [(int(c), int(r)) for c, r in grid], log)
        if shaz is None:
            raise ValueError("Could not detect the shadow direction (area too small?). Enter it manually.")
    la_c, lo_c = img.to_latlon((C0 + C1) / 2, (R0 + R1) / 2)
    el, el_src = sun_for_scan(opt, img, shaz, dn, de, la_c, lo_c, log)
    log(f"Sun elevation used: {el:.2f} deg  ({el_src})")
    tan = math.tan(el * D2R)
    min_h = opt.get("min_height") or 0.0
    min_len = max(5.0, 0.9 * min_h / tan)                  # 10 % slack: measured length is checked again below
    log(f"Looking for thin shadows {min_len:.1f}-{opt['max_shadow_m']:.0f} m long (objects above about {min_h:.0f} m)")
    # tiles
    T = 1024
    pad = int((opt["max_shadow_m"] + 15) / img.pm)
    tiles = [(c, r) for r in range(R0, R1, T) for c in range(C0, C1, T)]
    cands = []
    for ti, (c, r) in enumerate(tiles):
        if stop():
            raise ValueError("Stopped by user.")
        progress(ti / max(1, len(tiles)))
        cc0, rr0 = max(0, c - pad), max(0, r - pad)
        cc1, rr1 = min(img.ds.width, min(C1, c + T) + pad), min(img.ds.height, min(R1, r + T) + pad)
        a = img.ds.read(img.bands, window=Window(cc0, rr0, cc1 - cc0, rr1 - rr0)).astype(np.float32)
        if scale != 1.0:
            a *= scale
        if a.shape[0] == 1:
            a = np.repeat(a, 3, 0)
        L = a.mean(0)
        if (L > 0).mean() < 0.05:
            continue                                        # empty / no-data tile
        L[L <= 0] = 255                                     # no-data never looks like shadow
        core = (c - cc0, r - rr0, min(C1, c + T) - cc0, min(R1, r + T) - rr0)
        for s in scan_tile(L, img, shaz, opt.get("spread", 20), opt.get("step", 4), min_len, opt["max_shadow_m"], core):
            s["start"] = (s["start"][0] + cc0, s["start"][1] + rr0); s["end"] = (s["end"][0] + cc0, s["end"][1] + rr0)
            cands.append(s)
        if (ti + 1) % 25 == 0:
            log(f"  {ti + 1} of {len(tiles)} tiles scanned, {len(cands)} raw streaks")
    progress(1.0)
    found = dedupe_streaks(cands, img.pm)
    log(f"{len(found)} separate thin shadows found")
    # build result rows (same outputs as point mode)
    rows = []
    azs = [s["az"] for s in found]
    med_az = float(np.median(azs)) if azs else shaz
    for s in found:
        h = s["length_m"] * tan
        if h < min_h:
            continue
        bl, bo = img.to_latlon(*s["start"]); tl, to = img.to_latlon(*s["end"])
        bl += dn / 110700; bo += de / (111320 * math.cos(bl * D2R)); tl += dn / 110700; to += de / (111320 * math.cos(tl * D2R))
        lo_ = max(0.0, (s["length_m"] - 1) * math.tan((el - 2) * D2R)); hi_ = (s["length_m"] + 1) * math.tan((el + 2) * D2R)
        st = "ok"
        if h > 60:
            st = "check: very tall - verify on review sheet"
        elif s["width_m"] < AUTO_OK_WIDTH:
            st = "check: thin shadow - may be a gap between buildings"
        elif abs(wrap180(s["az"] - med_az)) > 10:
            st = "check: direction differs from other shadows"
        elif s["contrast"] < 30:
            st = "check: faint shadow"
        r = dict(status=st, height_m=round(h, 1), height_low_m=round(lo_, 1), height_high_m=round(hi_, 1), height_ft=round(h * FT, 1),
                 shadow_m=round(s["length_m"], 2), shadow_az=round(s["az"], 1), sun_elev=round(el, 2), lat=round(bl, 7), lon=round(bo, 7),
                 dms=f"{dms(bo, 'E', 'W')} {dms(bl, 'N', 'S')}", tip_lat=round(tl, 7), tip_lon=round(to, 7), sun_source=el_src,
                 input_lat="", input_lon="", known_height_m="", width_m=round(s["width_m"], 2))
        p = dict(inside=True, col=s["start"][0], row=s["start"][1], streak=s)
        rows.append((r, p))
    rows.sort(key=lambda rp: -rp[0]["height_m"])
    for i, (r, p) in enumerate(rows, 1):
        r["name"] = f"AUTO-{i:04d}"
    log(f"{len(rows)} objects at or above {min_h:.0f} m "
        f"({sum(1 for r, _ in rows if r['status'] == 'ok')} ok, {sum(1 for r, _ in rows if r['status'] != 'ok')} to check)")
    opt = dict(opt); opt.setdefault("out_name", "auto_scan_heights")
    write_outputs(rows, img, opt, shaz, el, el_src, log)
    return [r for r, _ in rows]


def read_sites(path):
    """Mosques / sites as list of (lat, lon, radius_m). Polygons (KML/SHP) -> centroid + half-size; points -> radius 30 m."""
    ext = os.path.splitext(path)[1].lower()
    feats = []
    if ext == ".gz":
        import gzip
        with gzip.open(path, "rt") as f:
            for r in csv.DictReader(f):
                feats.append((float(r["lat"]), float(r["lon"]), max(15.0, float(r["radius_m"]))))
        return feats
    groups = []
    if ext in (".kml", ".kmz"):
        if ext == ".kmz":
            with zipfile.ZipFile(path) as z:
                text = z.read(next(n for n in z.namelist() if n.lower().endswith(".kml"))).decode("utf-8", "replace")
        else:
            text = open(path, encoding="utf-8", errors="replace").read()
        for pm in re.findall(r"<Placemark[\s\S]*?</Placemark>", text):
            cs = []
            for block in re.findall(r"<coordinates>([\s\S]*?)</coordinates>", pm):
                cs += [(float(t.split(",")[1]), float(t.split(",")[0])) for t in block.split() if len(t.split(",")) >= 2]
            if cs:
                groups.append(cs)
    elif ext == ".shp":
        prj = os.path.splitext(path)[0] + ".prj"
        if os.path.exists(prj) and "PROJCS" in open(prj).read().upper():
            raise ValueError("The mosques shapefile is projected. Export it in WGS 1984 (geographic) first.")
        groups = [[(y, x) for x, y in sh.points] for sh in shapefile.Reader(path).shapes() if sh.points]
    else:
        groups = [[(p["lat"], p["lon"])] for p in read_points(path)]
    for cs in groups:
        la = sum(c[0] for c in cs) / len(cs); lo = sum(c[1] for c in cs) / len(cs)
        r = max([math.hypot((c[0] - la) * 110700, (c[1] - lo) * 111320 * math.cos(la * D2R)) for c in cs] + [0])
        feats.append((la, lo, max(30.0, min(150.0, r))))
    return feats


def builtin_mosques():
    for d in (os.path.dirname(os.path.abspath(__file__)), os.path.dirname(os.path.abspath(sys.argv[0]))):
        f = os.path.join(d, "mosques_ksa.csv.gz")
        if os.path.exists(f):
            return f
    raise ValueError("Built-in mosque list not found. Choose a mosques file (points or outlines).")


def streak_row(s, img, el, el_src, dn, de, status):
    tan = math.tan(el * D2R)
    h = s["length_m"] * tan
    bl, bo = img.to_latlon(*s["start"]); tl, to = img.to_latlon(*s["end"])
    bl += dn / 110700; bo += de / (111320 * math.cos(bl * D2R)); tl += dn / 110700; to += de / (111320 * math.cos(tl * D2R))
    lo_ = max(0.0, (s["length_m"] - 1) * math.tan((el - 2) * D2R)); hi_ = (s["length_m"] + 1) * math.tan((el + 2) * D2R)
    return dict(status=status, height_m=round(h, 1), height_low_m=round(lo_, 1), height_high_m=round(hi_, 1), height_ft=round(h * FT, 1),
                shadow_m=round(s["length_m"], 2), shadow_az=round(s["az"], 1), sun_elev=round(el, 2), lat=round(bl, 7), lon=round(bo, 7),
                dms=f"{dms(bo, 'E', 'W')} {dms(bl, 'N', 'S')}", tip_lat=round(tl, 7), tip_lon=round(to, 7), sun_source=el_src,
                input_lat="", input_lon="", known_height_m="")


def run_mosques(opt, log=print, progress=lambda f: None, stop=lambda: False):
    """Automatic minaret search: look for tall thin shadows around every mosque in the image (built-in list or your own)."""
    img = Img(opt["image"])
    log(f"Image {img.ds.width} x {img.ds.height} px, {img.mx:.2f} x {img.my:.2f} m per pixel")
    dn, de = opt.get("offset_n", 0.0) or 0.0, opt.get("offset_e", 0.0) or 0.0
    src = opt.get("mosques") or builtin_mosques()
    sites = read_sites(src)
    if opt.get("area"):
        C0, R0, C1, R1 = read_aoi(opt["area"], img, 0, dn, de)
    else:
        C0, R0, C1, R1 = 0, 0, img.ds.width, img.ds.height
    inside = []
    for la, lo, rad in sites:
        c, r = img.to_pixel(la - dn / 110700.0, lo - de / (111320.0 * math.cos(la * D2R)))
        if C0 <= c < C1 and R0 <= r < R1:
            inside.append((la, lo, rad, c, r))
    log(f"{len(inside)} mosques inside the {'area' if opt.get('area') else 'image'} "
        f"({'built-in OpenStreetMap list' if not opt.get('mosques') else os.path.basename(src)})")
    if not inside:
        raise ValueError("No mosques fall inside the image / area.")
    shaz = opt.get("shadow_az")
    if shaz is None:
        shaz = detect_shadow_azimuth(img, [(int(c), int(r)) for *_, c, r in inside], log)
        if shaz is None:
            raise ValueError("Could not detect the shadow direction; enter it manually.")
    la_c = float(np.mean([x[0] for x in inside])); lo_c = float(np.mean([x[1] for x in inside]))
    el, el_src = sun_for_scan(opt, img, shaz, dn, de, la_c, lo_c, log)
    log(f"Sun elevation used: {el:.2f} deg  ({el_src})")
    tan = math.tan(el * D2R)
    min_h = opt.get("min_height") or 0.0
    min_len = max(4.0, 0.9 * min_h / tan)
    per = opt.get("per_mosque", 2)
    cands = []
    for i, (la, lo, rad, c, r) in enumerate(inside):
        if stop():
            raise ValueError("Stopped by user.")
        progress(i / len(inside))
        search = min(90.0, rad + 12.0)
        cs = find_streak(img, c, r, shaz, search, opt["max_shadow_m"], spread=opt.get("spread", 25), collect=True, min_len=min_len)
        # thin streaks (< 1.3 m) are mostly gaps between buildings, kerbs or fences, not minarets
        cs = [x for x in cs if x["width_m"] >= AUTO_MIN_WIDTH]
        for x in cs:
            x["score"] = x["length_m"] - 0.3 * x["start_dist_m"]
            x["mosque"] = i; x["site"] = (la, lo)
        cs = dedupe_streaks(cs, img.pm)
        cs.sort(key=lambda x: -x["score"])
        for rank, x in enumerate(cs[:per], 1):
            x["rank"] = rank
        cands += cs[:per]
        if (i + 1) % 50 == 0:
            log(f"  {i + 1} of {len(inside)} mosques searched, {len(cands)} candidate shadows")
    progress(1.0)
    found = dedupe_streaks(cands, img.pm)
    rows = []
    for s in found:
        h = s["length_m"] * tan
        if h < min_h:
            continue
        st = "likely - verify on review sheet"
        if h > 70:
            st = "possible - very tall, verify"
        elif s.get("rank", 1) > 1:
            st = "possible - 2nd shadow at this mosque"
        elif s["width_m"] < AUTO_OK_WIDTH:
            st = "possible - thin, may be a gap between buildings"
        elif s["contrast"] < 28:
            st = "possible - faint shadow"
        rr = streak_row(s, img, el, el_src, dn, de, st)
        rr.update(width_m=round(s["width_m"], 2), mosque_lat=round(s["site"][0], 6), mosque_lon=round(s["site"][1], 6),
                  mosque=s.get("mosque"), dist=round(s["start_dist_m"], 1))
        rows.append((rr, dict(inside=True, col=s["start"][0], row=s["start"][1], streak=s)))
    rows.sort(key=lambda rp: -rp[0]["height_m"])
    for i, (r, p) in enumerate(rows, 1):
        r["name"] = f"MNR-{i:04d}"
    log(f"{len(rows)} candidate minaret shadows at or above {min_h:.0f} m around {len(inside)} mosques "
        f"({sum(1 for r, _ in rows if r['status'].startswith('likely'))} likely). "
        "Open the review sheet and delete the ones that are not minarets (walls and gaps between buildings look similar).")
    opt = dict(opt); opt.setdefault("out_name", "minaret_scan_heights")
    write_outputs(rows, img, opt, shaz, el, el_src, log)
    return [r for r, _ in rows]


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
    # automatic modes on the same image: must find exactly the 6 towers
    try:
        mcsv = os.path.join(folder, "selftest_mosques.csv")
        with open(mcsv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["name", "lat", "lon"])
            for i, ((c, r, h), la, lo) in enumerate(zip(towers, lats, lons), 1):
                w.writerow([f"M{i}", la + 4 / 110700.0, lo])        # site point a few metres off the tower
        for name, fn, extra in (("mosque search", run_mosques, dict(mosques=mcsv)), ("whole-image scan", run_scan, dict())):
            o = dict(image=tif, out_dir=os.path.join(folder, "selftest_" + name.split()[0]), max_shadow_m=80, sun_elev=el,
                     shadow_az=None, min_height=10)
            o.update(extra)
            got = fn(o, log=print)
            hs = sorted(float(r["height_m"]) for r in got)
            errs_a = [abs(a_ - b_) for a_, b_ in zip(hs, sorted(h for c, r, h in towers))]
            good = len(hs) == len(towers) and max(errs_a) <= 3.0
            print(f"{name}: found {len(hs)} of {len(towers)} towers, heights {hs}", "OK" if good else "FAILED")
            ok = ok and good
    except Exception as ex:
        print("Automatic mode test FAILED:", ex); traceback.print_exc(); ok = False
    try:
        n_m = len(read_sites(builtin_mosques()))
        print(f"Built-in mosque list: {n_m} mosques", "OK" if n_m > 10000 else "FAILED")
        ok = ok and n_m > 10000
    except Exception as ex:
        print("Built-in mosque list FAILED:", ex); ok = False
    print("SELFTEST", "PASSED" if ok else "FAILED", "max error %.1f m" % max(errs))
    return ok


# ======================================================================= GUI
def gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    root = tk.Tk(); root.title(f"{APP} {VERSION}"); root.geometry("820x760")
    v = {k: tk.StringVar() for k in ("image", "points", "mosques", "area", "out", "elev", "date", "time", "tz", "az", "search", "maxsh", "minh", "dn", "de")}
    v["tz"].set("3"); v["search"].set("25"); v["maxsh"].set("80"); v["minh"].set("15"); v["dn"].set("0"); v["de"].set("0")
    mode = tk.StringVar(value="known")
    job = tk.StringVar(value="points")
    stop_flag = {"stop": False}
    frm = ttk.Frame(root, padding=12); frm.pack(fill="both", expand=True)

    jf = ttk.LabelFrame(frm, text="What to do", padding=8); jf.grid(row=0, column=0, columnspan=3, sticky="we", pady=(0, 6))
    ttk.Radiobutton(jf, text="Measure heights at my points (points file required)", variable=job, value="points").grid(row=0, column=0, sticky="w")
    ttk.Radiobutton(jf, text="AUTOMATIC: find minarets around every mosque (built-in Saudi/Gulf mosque list, or your own mosques file)",
                    variable=job, value="mosques").grid(row=1, column=0, sticky="w")
    ttk.Radiobutton(jf, text="AUTOMATIC: scan the whole image / area for poles, masts, pylons, towers (best in open ground)",
                    variable=job, value="scan").grid(row=2, column=0, sticky="w")

    def row(r, label, var, browse=None, width=60):
        ttk.Label(frm, text=label).grid(row=r, column=0, sticky="w", pady=3)
        ttk.Entry(frm, textvariable=var, width=width).grid(row=r, column=1, sticky="we", pady=3)
        if browse:
            ttk.Button(frm, text="Browse...", command=browse).grid(row=r, column=2, padx=4)
    vec = [("Vector", "*.csv *.txt *.shp *.kml *.kmz"), ("All", "*.*")]
    row(1, "Image (GeoTIFF / JP2)", v["image"], lambda: v["image"].set(filedialog.askopenfilename(filetypes=[("Images", "*.tif *.tiff *.jp2 *.img *.ecw"), ("All", "*.*")])))
    row(2, "Points (CSV / SHP / KML)", v["points"], lambda: v["points"].set(filedialog.askopenfilename(filetypes=vec)))
    row(3, "Mosques file (optional)", v["mosques"], lambda: v["mosques"].set(filedialog.askopenfilename(filetypes=vec)))
    row(4, "Limit to area (optional)", v["area"], lambda: v["area"].set(filedialog.askopenfilename(filetypes=vec)))
    row(5, "Output folder", v["out"], lambda: v["out"].set(filedialog.askdirectory()))
    ttk.Label(frm, foreground="#555", text="Automatic modes: Points file is optional - use it only to give 2+ known heights for sun calibration. "
              "Mosques file blank = built-in list. Area = any KML/SHP/CSV, its extent is scanned.", wraplength=760).grid(row=6, column=0, columnspan=3, sticky="w")
    lf = ttk.LabelFrame(frm, text="Sun elevation", padding=8); lf.grid(row=7, column=0, columnspan=3, sticky="we", pady=8)
    ttk.Radiobutton(lf, text="Calibrate from known heights in the points file (column known_height_m, 2 or more)", variable=mode, value="known").grid(row=0, column=0, columnspan=4, sticky="w")
    ttk.Radiobutton(lf, text="Sun elevation from metadata (deg):", variable=mode, value="elev").grid(row=1, column=0, sticky="w")
    ttk.Entry(lf, textvariable=v["elev"], width=10).grid(row=1, column=1, sticky="w")
    ttk.Radiobutton(lf, text="Image date (YYYY-MM-DD):", variable=mode, value="date").grid(row=2, column=0, sticky="w")
    ttk.Entry(lf, textvariable=v["date"], width=12).grid(row=2, column=1, sticky="w")
    ttk.Label(lf, text="time HH:MM (optional)").grid(row=2, column=2, sticky="e"); ttk.Entry(lf, textvariable=v["time"], width=7).grid(row=2, column=3, sticky="w")
    ttk.Label(lf, text="time zone (h from UTC)").grid(row=3, column=2, sticky="e"); ttk.Entry(lf, textvariable=v["tz"], width=7).grid(row=3, column=3, sticky="w")
    of = ttk.LabelFrame(frm, text="Options", padding=8); of.grid(row=8, column=0, columnspan=3, sticky="we")
    opts = (("Shadow direction deg (blank = detect)", "az"), ("Search radius around each point (m)", "search"), ("Longest shadow (m)", "maxsh"),
            ("Report heights above (m)", "minh"), ("Your data minus image, north (m)", "dn"), ("Your data minus image, east (m)", "de"))
    for i, (lab, k) in enumerate(opts):
        ttk.Label(of, text=lab).grid(row=i // 2, column=(i % 2) * 2, sticky="w", padx=4, pady=2)
        ttk.Entry(of, textvariable=v[k], width=10).grid(row=i // 2, column=(i % 2) * 2 + 1, sticky="w", pady=2)
    pb = ttk.Progressbar(frm, mode="determinate", maximum=100); pb.grid(row=9, column=0, columnspan=3, sticky="we", pady=8)
    txt = tk.Text(frm, height=12, wrap="word"); txt.grid(row=11, column=0, columnspan=3, sticky="nsew")
    frm.columnconfigure(1, weight=1); frm.rowconfigure(11, weight=1)

    def log(m):
        root.after(0, lambda: (txt.insert("end", m + "\n"), txt.see("end")))

    def num(k, default=None):
        s = v[k].get().strip()
        return float(s) if s else default

    def work():
        try:
            j = job.get()
            out = v["out"].get() or os.path.join(os.path.dirname(v["image"].get()), "shadow_height_output")
            o = dict(image=v["image"].get(), points=v["points"].get().strip() or None, out_dir=out,
                     search_m=num("search", 25), max_shadow_m=num("maxsh", 80), min_height=num("minh"), shadow_az=num("az"),
                     offset_n=num("dn", 0), offset_e=num("de", 0), tz=num("tz", 3), use_known=mode.get() == "known",
                     mosques=v["mosques"].get().strip() or None, area=v["area"].get().strip() or None)
            if not o["image"]:
                raise ValueError("Choose the image.")
            if j == "points" and not o["points"]:
                raise ValueError("Choose the points file (or pick an automatic mode).")
            if mode.get() == "elev":
                o["sun_elev"] = num("elev")
                if o["sun_elev"] is None:
                    raise ValueError("Enter the sun elevation.")
            if mode.get() == "date":
                o["date"] = dt.date.fromisoformat(v["date"].get().strip())
                if v["time"].get().strip():
                    hh, mm = v["time"].get().strip().split(":")[:2]; o["time"] = dt.time(int(hh), int(mm))
            if mode.get() == "known" and not o["points"]:
                raise ValueError("Known-height calibration needs a points file with known_height_m. Or choose metadata elevation / date.")
            prog = lambda f: root.after(0, lambda: pb.configure(value=f * 100))
            stop = lambda: stop_flag["stop"]
            if j == "points":
                run_batch(o, log=log, progress=prog)
            elif j == "mosques":
                run_mosques(o, log=log, progress=prog, stop=stop)
            else:
                run_scan(o, log=log, progress=prog, stop=stop)
            root.after(0, lambda: messagebox.showinfo(APP, "Finished. Check the review sheet (JPG) in the output folder:\n" + out))
        except Exception as e:
            log("ERROR: " + str(e)); log(traceback.format_exc())
            root.after(0, lambda: messagebox.showerror(APP, str(e)))
        finally:
            root.after(0, lambda: (run_btn.configure(state="normal"), stop_btn.configure(state="disabled")))

    def start():
        txt.delete("1.0", "end"); run_btn.configure(state="disabled"); stop_btn.configure(state="normal"); stop_flag["stop"] = False
        threading.Thread(target=work, daemon=True).start()
    bf = ttk.Frame(frm); bf.grid(row=10, column=0, columnspan=3, pady=4)
    run_btn = ttk.Button(bf, text="Run", command=start); run_btn.pack(side="left", padx=6)
    stop_btn = ttk.Button(bf, text="Stop", state="disabled", command=lambda: stop_flag.update(stop=True)); stop_btn.pack(side="left", padx=6)
    log("Measure at points: CSV with name, lat, lon (or a 'coord' column with DMS); optional known_height_m.\n"
        "Automatic minarets: no points needed - mosques come from the built-in list (or your file).\n"
        "Mosaics with several acquisition dates: run each acquisition strip separately (Limit to area), each has its own sun angle.\n"
        "ECW images: export to GeoTIFF or JP2 first.")
    if os.environ.get("SHB_GUI_SMOKE"):                     # build check: open the window, then close it
        def _close():
            open(os.environ["SHB_GUI_SMOKE"], "w").write("GUI OK"); root.destroy()
        root.after(2000, _close)
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
    ap.add_argument("--scan", action="store_true", help="scan the whole image (or --area) for tall thin objects; no points needed")
    ap.add_argument("--area", help="KML/KMZ/SHP/CSV whose extent limits the scan")
    ap.add_argument("--mosques-auto", action="store_true", help="find minarets around mosques of the built-in OpenStreetMap list")
    ap.add_argument("--mosques", help="your own mosques file (points or outlines: CSV/SHP/KML) for the minaret search")
    ap.add_argument("--spread", type=float, default=20); ap.add_argument("--step", type=float, default=4)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest(a.selftest) else 1)
    o = dict(image=a.image, points=a.points, out_dir=a.out, search_m=a.search, max_shadow_m=a.max_shadow, min_height=a.min_height,
             shadow_az=a.shadow_az, offset_n=a.offset_n, offset_e=a.offset_e, tz=a.tz, use_known=not a.no_known, sun_elev=a.sun_elev)
    if a.date:
        o["date"] = dt.date.fromisoformat(a.date)
        if a.time:
            hh, mm = a.time.split(":")[:2]; o["time"] = dt.time(int(hh), int(mm))
    if a.mosques_auto or a.mosques:
        o["area"] = a.area; o["mosques"] = a.mosques
        run_mosques(o)
    elif a.scan:
        o["area"] = a.area; o["spread"] = a.spread; o["step"] = a.step
        run_scan(o)
    else:
        run_batch(o)


if __name__ == "__main__":
    main()
