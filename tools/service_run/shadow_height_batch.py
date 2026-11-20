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
VERSION = "3.5"
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
    if opt.get("second"):
        second_view_check(rows, img, opt, el, log=log, progress=progress)
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
            "sun_elev", "known_height_m", "error_vs_known_m", "tip_lat", "tip_lon", "input_lat", "input_lon", "width_m", "mosque_lat", "mosque_lon", "model_score", "h2_m", "second_check", "second_date", "confirm", "sun_source"]
    with open(base + ".csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for r, _ in rows:
            w.writerow(r)
    ok = [(r, p) for r, p in rows if r.get("height_m") is not None]
    for kind in ("points", "shadow_lines"):
        sw = shapefile.Writer(f"{base}_{kind}", shapeType=shapefile.POINT if kind == "points" else shapefile.POLYLINE, encoding="utf-8")
        for fd in (("NAME", "C", 40, 0), ("STATUS", "C", 60, 0), ("HEIGHT_M", "F", 8, 1), ("H_LOW_M", "F", 8, 1), ("H_HIGH_M", "F", 8, 1),
                   ("SHADOW_M", "F", 8, 2), ("SHADOW_AZ", "F", 6, 1), ("SUN_ELEV", "F", 6, 2), ("DMS", "C", 40, 0),
                   ("MODEL", "F", 8, 2), ("CONFIRM", "C", 3, 0)):
            sw.field(*fd)
        for r, _ in ok:
            if kind == "points":
                sw.point(r["lon"], r["lat"])
            else:
                sw.line([[[r["lon"], r["lat"]], [r["tip_lon"], r["tip_lat"]]]])
            sw.record(str(r["name"])[:40], r["status"][:60], r["height_m"], r["height_low_m"], r["height_high_m"], r["shadow_m"], r["shadow_az"], r["sun_elev"], r["dms"],
                      r.get("model_score") if r.get("model_score") not in ("", None) else None, "")
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
        cols_, per = 6, 240                                  # JPEG height limit: several sheets of 40 rows
        for k in range(0, len(tiles), per):
            part = tiles[k:k + per]
            sheet = Image.new("RGB", (260 * min(cols_, len(part)), 260 * ((len(part) + cols_ - 1) // cols_)))
            for i, t in enumerate(part):
                sheet.paste(t, ((i % cols_) * 260, (i // cols_) * 260))
            sheet.save(base + ("_review.jpg" if k == 0 else f"_review_{k // per + 1}.jpg"), quality=88)
    # candidate chips (for training) and the clickable review page
    chips = [(str(r["name"]), p["chip"], p["vec"]) for r, p in rows if p.get("chip") is not None and p.get("vec") is not None]
    if chips:
        np.savez_compressed(base + "_chips.npz", names=np.array([c[0] for c in chips]), chips=np.array([c[1] for c in chips], np.uint8),
                            vecs=np.array([c[2] for c in chips], np.float32))
    if any(p.get("streak") for r, p in rows):
        try:
            n_html = write_review_html(rows, img, base, os.path.basename(base))
            log(f"Review page: {os.path.basename(base)}_review.html ({n_html} cards) - open it in a browser, mark Yes/No, press 'Save labels'"
                + (" and put the downloaded CSV in this folder; then use 'Train' to teach the model." if chips else "."))
        except Exception as ex:
            log("Review page could not be written: " + str(ex))
    from collections import Counter
    cnt = Counter(str(r.get("status", "")).split(" - ")[0].split(":")[0] for r, _ in rows)
    log(f"Done: {len(rows)} rows (" + ", ".join(f"{v} {k}" for k, v in cnt.most_common()) + f"). Outputs in {out}")
    log("   " + ", ".join(os.path.basename(base) + s for s in (".csv", "_points.shp", "_shadow_lines.shp", ".kml", "_review.jpg")))


# ======================================================================= learned shadow classifier
CHIP_U0, CHIP_U1, CHIP_V, CHIP_RES = -12.0, 36.0, 8.0, 0.5         # chip: 12 m before the shadow start to 36 m after, +-8 m across
CHIP_W, CHIP_H = int((CHIP_U1 - CHIP_U0) / CHIP_RES), int(2 * CHIP_V / CHIP_RES)


def make_chip(img, start, end, width):
    """Shadow-aligned image chip (shadow points right), 0.5 m pixels, RGB + streak mask. uint8 (32, 96, 4)."""
    sc, sr = start; ec, er = end
    vx, vy = ec - sc, er - sr; n = math.hypot(vx, vy) or 1.0
    ux, uy = vx / n, vy / n
    s = CHIP_RES / img.pm
    half = int(45 / img.pm)
    a, x0, y0 = img.read(sc, sr, half)
    if a is None:
        return None
    rgb = np.moveaxis(np.clip(a, 0, 255), 0, -1).astype(np.float32)
    M = np.array([[ux * s, -uy * s, (sc - x0) + ux * CHIP_U0 / img.pm + uy * CHIP_V / img.pm],
                  [uy * s, ux * s, (sr - y0) + uy * CHIP_U0 / img.pm - ux * CHIP_V / img.pm]], np.float32)
    c = cv2.warpAffine(rgb, M, (CHIP_W, CHIP_H), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    m = np.zeros((CHIP_H, CHIP_W), np.uint8)
    L = n * img.pm
    i0, i1 = int(-CHIP_U0 / CHIP_RES), int(min(CHIP_W - 1, (L - CHIP_U0) / CHIP_RES))
    hw = max(1, int(round(width / 2 / CHIP_RES)))
    m[CHIP_H // 2 - hw:CHIP_H // 2 + hw + 1, i0:i1 + 1] = 255
    return np.dstack([c.astype(np.uint8), m])


def chip_vector(ch, length, width, contrast, dist):
    """Feature vector: normalised grey thumbnail + gradient-orientation histograms + shape numbers."""
    g = ch[:, :, :3].astype(np.float32).mean(2)
    g = (g - g.mean()) / (g.std() + 8.0)
    thumb = cv2.resize(g, (CHIP_W // 2, CHIP_H // 2), interpolation=cv2.INTER_AREA).ravel()
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy); ang = (np.degrees(np.arctan2(gy, gx)) % 180.0)
    b = np.minimum((ang / 20.0).astype(int), 8)
    hog = []
    for cy in range(0, CHIP_H, 8):
        for cx in range(0, CHIP_W, 8):
            h = np.bincount(b[cy:cy + 8, cx:cx + 8].ravel(), weights=mag[cy:cy + 8, cx:cx + 8].ravel(), minlength=9)
            hog.append(h / (np.linalg.norm(h) + 1e-3))
    raw = ch[:, :, :3].astype(np.float32).mean(2)
    yc = CHIP_H // 2; hw = max(1, int(round(width / 2 / CHIP_RES)))
    cols = np.nonzero(ch[:, :, 3].max(0))[0]
    i0, i1 = (cols.min(), cols.max()) if len(cols) else (24, 30)
    up = raw[max(0, yc - hw - 5):yc - hw - 1, i0:i1 + 1].mean() if yc - hw - 1 > 0 else 128
    dn = raw[yc + hw + 2:yc + hw + 6, i0:i1 + 1].mean()
    beyond = (raw[yc - 1:yc + 2, min(CHIP_W - 1, i1 + 3):min(CHIP_W, i1 + 15)] < 95).mean() if i1 + 4 < CHIP_W else 0.0
    before = (raw[yc - 1:yc + 2, max(0, i0 - 10):max(1, i0 - 2)] < 95).mean()
    sc = np.array([length / 30.0, width / 3.0, contrast / 60.0, dist / 30.0, up / 255.0, dn / 255.0, abs(up - dn) / 255.0,
                   beyond, before, min(length, 60) ** 0.5 / 8.0], np.float32)
    return np.concatenate([thumb, np.concatenate(hog), sc]).astype(np.float32)


def train_classifier(X, y, l2=None, iters=600, log=print):
    """L2-regularised logistic regression (numpy, Adam). Returns model dict. Positives are up-weighted to balance."""
    X = np.asarray(X, np.float32); y = np.asarray(y, np.float32)
    mu = X.mean(0); sd = X.std(0) + 1e-3
    Z = (X - mu) / sd
    wpos = (len(y) - y.sum()) / max(1.0, y.sum())
    sw = np.where(y > 0, wpos, 1.0).astype(np.float32); sw /= sw.mean()

    def fit(Zt, yt, swt, lam):
        w = np.zeros(Zt.shape[1], np.float32); b = 0.0
        m1 = np.zeros_like(w); v1 = np.zeros_like(w); mb = vb = 0.0
        for t in range(1, iters + 1):
            p = 1 / (1 + np.exp(-np.clip(Zt @ w + b, -30, 30)))
            g = (p - yt) * swt
            gw = Zt.T @ g / len(yt) + lam * w; gb = g.mean()
            m1 = 0.9 * m1 + 0.1 * gw; v1 = 0.999 * v1 + 0.001 * gw * gw
            mb = 0.9 * mb + 0.1 * gb; vb = 0.999 * vb + 0.001 * gb * gb
            lr = 0.01
            w -= lr * (m1 / (1 - 0.9 ** t)) / (np.sqrt(v1 / (1 - 0.999 ** t)) + 1e-8)
            b -= lr * (mb / (1 - 0.9 ** t)) / (np.sqrt(vb / (1 - 0.999 ** t)) + 1e-8)
        return w, b

    def auc(sc, yt):
        P = sc[yt > 0]; N = sc[yt == 0]
        if len(P) == 0 or len(N) == 0:
            return float("nan")
        r = np.argsort(np.argsort(np.concatenate([P, N])))
        return float((r[:len(P)].sum() - len(P) * (len(P) - 1) / 2) / (len(P) * len(N)))

    rng = np.random.default_rng(0)
    folds = rng.permutation(len(y)) % 5
    cv = {}
    for lam in ([l2] if l2 else [0.3, 0.1, 0.03, 0.01]):
        sc = np.zeros(len(y), np.float32)
        for f in range(5):
            tr, te = folds != f, folds == f
            w, b = fit(Z[tr], y[tr], sw[tr], lam)
            sc[te] = Z[te] @ w + b
        cv[lam] = auc(sc, y)
    lam = max(cv, key=lambda k: (cv[k] if cv[k] == cv[k] else -1))
    sc = np.zeros(len(y), np.float32)
    for f in range(5):
        tr, te = folds != f, folds == f
        w, b = fit(Z[tr], y[tr], sw[tr], lam)
        sc[te] = Z[te] @ w + b
    ps = sc[y > 0]
    t50 = float(np.percentile(ps, 50)) if len(ps) else 0.0      # half of the real objects score above this
    t85 = float(np.percentile(ps, 15)) if len(ps) else -1.0     # 85 % score above this
    share = lambda t: float((sc >= t).mean())
    log(f"Review load: top {100 * share(t50):.0f}% of candidates hold half of the real objects, top {100 * share(t85):.0f}% hold 85%.")
    w, b = fit(Z, y, sw, lam)
    log(f"Trained on {int(y.sum())} confirmed / {int(len(y) - y.sum())} rejected examples; cross-validated AUC "
        + ", ".join(f"{cv[k]:.2f} (l2={k})" for k in cv) + f"; using l2={lam}")
    return dict(w=w, b=np.float32(b), mu=mu, sd=sd, n_pos=int(y.sum()), n_neg=int(len(y) - y.sum()), cv_auc=np.float32(cv[lam]),
                t_likely=np.float32(t50), t_possible=np.float32(t85))


def _model_dirs():
    """Where trained models live: next to the program (shared), the user's folder, then the built-in starter model."""
    return [os.path.dirname(os.path.abspath(sys.argv[0])), os.path.join(os.path.expanduser("~"), "ShadowHeightBatch"),
            os.path.dirname(os.path.abspath(__file__))]


def model_path_default():
    for d in _model_dirs():
        f = os.path.join(d, "shadow_model.npz")
        if os.path.exists(f):
            return f
    return None


def model_save_dir():
    for d in _model_dirs()[:2]:
        try:
            os.makedirs(d, exist_ok=True)
            t = os.path.join(d, ".write_test"); open(t, "w").write("x"); os.remove(t)
            return d
        except Exception:
            continue
    return os.getcwd()


def load_model(path=None):
    path = path or model_path_default()
    if not path or not os.path.exists(path):
        return None
    z = np.load(path)
    return {k: z[k] for k in z.files}


def model_score(model, vec):
    """Raw model score (log-odds). Compare with model['t_likely'] / model['t_possible']."""
    zz = (vec - model["mu"]) / model["sd"]
    return float(zz @ model["w"] + model["b"])


def model_prob(model, vec):
    return float(1 / (1 + math.exp(-max(-30.0, min(30.0, model_score(model, vec))))))


def model_class(model, score):
    if score >= float(model["t_likely"]):
        return "likely"
    if score >= float(model["t_possible"]):
        return "possible"
    return "unlikely"


def read_labels(folder):
    """Labels Y/N per candidate name from a review-page export (*_labels.csv) or the 'confirm' column of the results CSV/DBF."""
    lab = {}
    def norm(v):
        v = str(v or "").strip().lower()
        if v in ("y", "yes", "1", "true", "ok", "correct", "minaret", "tower", "pylon", "pole"):
            return 1
        if v in ("n", "no", "0", "false", "wrong", "x", "not"):
            return 0
        return None
    for fn in sorted(os.listdir(folder)):
        f = os.path.join(folder, fn)
        if fn.lower().endswith(".csv"):
            try:
                with open(f, encoding="utf-8-sig") as fh:
                    rd = csv.DictReader(fh)
                    flds = {k.lower(): k for k in (rd.fieldnames or [])}
                    key = flds.get("label") or flds.get("confirm")
                    if not key or "name" not in flds:
                        continue
                    for r in rd:
                        v = norm(r.get(key))
                        if v is not None:
                            lab[r[flds["name"]]] = v
            except Exception:
                pass
        elif fn.lower().endswith("_points.dbf"):
            try:
                rd = shapefile.Reader(dbf=open(f, "rb"))
                names = [fd[0].upper() for fd in rd.fields[1:]]
                if "CONFIRM" in names and "NAME" in names:
                    for rec in rd.records():
                        d = dict(zip(names, rec))
                        v = norm(d.get("CONFIRM"))
                        if v is not None:
                            lab[str(d["NAME"]).strip()] = v
            except Exception:
                pass
    return lab


def run_train(folders, log=print, model_out=None):
    """Learn from confirmed (Y) / rejected (N) candidates in one or more result folders; adds to earlier training data."""
    md = os.path.dirname(model_out) if model_out else model_save_dir()
    data_f = os.path.join(md, "shadow_training_data.npz")
    keys, X, y = [], [], []
    src = data_f if os.path.exists(data_f) else None
    if src is None:                                   # start from the built-in starter examples
        for d in _model_dirs()[2:]:
            f = os.path.join(d, "shadow_training_data.npz")
            if os.path.exists(f):
                src = f
    if src:
        z = np.load(src, allow_pickle=False)
        keys, X, y = list(z["keys"]), list(z["X"]), list(z["y"])
        log(f"Earlier training data: {int(np.sum(y))} confirmed / {len(y) - int(np.sum(y))} rejected ({os.path.basename(src)})")
    index = {k: i for i, k in enumerate(keys)}
    added = 0
    for folder in folders:
        npzs = [f for f in os.listdir(folder) if f.endswith("_chips.npz")]
        if not npzs:
            log(f"  {folder}: no *_chips.npz (run an automatic mode first) - skipped"); continue
        lab = read_labels(folder)
        if not lab:
            log(f"  {folder}: no labels found (review page export *_labels.csv, or fill the 'confirm' column with Y/N) - skipped"); continue
        for f in npzs:
            z = np.load(os.path.join(folder, f), allow_pickle=False)
            for nm, vec in zip(z["names"], z["vecs"]):
                if nm in lab:
                    k = f"{os.path.abspath(folder)}|{f}|{nm}"
                    if k in index:
                        X[index[k]] = vec; y[index[k]] = lab[nm]
                    else:
                        index[k] = len(keys); keys.append(k); X.append(vec); y.append(lab[nm]); added += 1
        log(f"  {folder}: {sum(lab.values())} confirmed, {len(lab) - sum(lab.values())} rejected")
    y = np.array(y, np.float32)
    if y.sum() < 5 or (len(y) - y.sum()) < 5:
        raise ValueError("Need at least 5 confirmed and 5 rejected examples to train.")
    model = train_classifier(np.array(X, np.float32), y, log=log)
    out = model_out or os.path.join(md, "shadow_model.npz")
    np.savez_compressed(out, **model)
    np.savez_compressed(data_f, keys=np.array(keys), X=np.array(X, np.float32), y=y)
    log(f"Model saved: {out}  ({added} new examples). Automatic modes use it from now on. "
        f"Share it with colleagues by copying shadow_model.npz next to their ShadowHeightBatch.exe.")
    return out


def write_review_html(rows, img, base, title):
    """Self-contained review page: one card per candidate, Yes/No buttons, saves <name>_labels.csv for training."""
    import base64, io, html as _html
    cards = []
    for r, p in rows:
        s = p.get("streak")
        if not s:
            continue
        half = int(max(18.0, s["length_m"] * 0.8 + 8) / img.pm)
        a, c0, r0 = img.read((s["start"][0] + s["end"][0]) / 2, (s["start"][1] + s["end"][1]) / 2, half)
        if a is None:
            continue
        T = 230; Z = T / (2 * half)
        im = Image.fromarray(np.moveaxis(np.clip(a, 0, 255).astype(np.uint8), 0, -1)).resize((T, T), Image.LANCZOS)
        dr = ImageDraw.Draw(im)
        vx, vy = s["end"][0] - s["start"][0], s["end"][1] - s["start"][1]; n = math.hypot(vx, vy) or 1
        px_, py_ = -vy / n, vx / n; o = 4.0 / img.pm
        for (cx, cy), col in ((s["start"], (255, 0, 255)), (s["end"], (0, 255, 255))):
            dr.line([((cx - c0 + px_ * o) * Z, (cy - r0 + py_ * o) * Z), ((cx - c0 + px_ * o * 2) * Z, (cy - r0 + py_ * o * 2) * Z)], fill=col, width=3)
        b1 = io.BytesIO(); im.save(b1, "JPEG", quality=80)
        imgs = [base64.b64encode(b1.getvalue()).decode()]
        ch = p.get("chip")
        if ch is not None:
            ci = Image.fromarray(ch[:, :, :3]).resize((CHIP_W * 3, CHIP_H * 3), Image.LANCZOS)
            d2 = ImageDraw.Draw(ci); cols_ = np.nonzero(ch[:, :, 3].max(0))[0]; rws = np.nonzero(ch[:, :, 3].max(1))[0]
            if len(cols_):
                x0_, x1_ = cols_.min() * 3, (cols_.max() + 1) * 3; y0_, y1_ = rws.min() * 3 - 8, (rws.max() + 1) * 3 + 8
                d2.line([x0_, y0_, x1_, y0_], fill=(255, 0, 255), width=2); d2.line([x0_, y1_, x1_, y1_], fill=(255, 0, 255), width=2)
            b2 = io.BytesIO(); ci.save(b2, "JPEG", quality=80); imgs.append(base64.b64encode(b2.getvalue()).decode())
        cards.append(dict(name=str(r["name"]), h=r.get("height_m"), st=r.get("status", ""), sc=r.get("model_score", ""),
                          lat=r.get("lat"), lon=r.get("lon"), imgs=imgs))
    import json as _json
    page = """<!doctype html><html><head><meta charset="utf-8"><title>Review - %(title)s</title><style>
body{font-family:Segoe UI,Arial,sans-serif;background:#1d1f22;color:#eee;margin:0}
header{position:sticky;top:0;background:#111;padding:10px 14px;display:flex;gap:14px;align-items:center;flex-wrap:wrap;z-index:2}
header b{color:#ffd54a} button{font:inherit;border:0;border-radius:6px;padding:6px 12px;cursor:pointer}
.grid{display:flex;flex-wrap:wrap;gap:10px;padding:12px}
.card{background:#2a2d31;border:3px solid #2a2d31;border-radius:8px;padding:6px;width:530px}
.card.sel{border-color:#ffd54a}.card.y{background:#1f3a24}.card.n{background:#3a1f1f}
.card img{display:block;border-radius:4px}.row{display:flex;gap:6px;align-items:flex-start}
.meta{font-size:13px;margin:4px 0}.yes{background:#2e7d32;color:#fff}.no{background:#c62828;color:#fff}.un{background:#555;color:#fff}
.help{font-size:12px;color:#bbb}</style></head><body>
<header><b>%(title)s</b><span id="cnt"></span>
<button class="yes" onclick="save()">Save labels (CSV for training)</button>
<label>Show <select id="flt" onchange="draw()"><option value="all">all</option><option value="u">not labelled</option><option value="y">yes</option><option value="n">no</option></select></label>
<span class="help">Click a card, then press <b>Y</b> = real tall thin object (minaret / tower / pylon / pole), <b>N</b> = not (wall, gap, tree, car), <b>U</b> = unsure. Arrow keys move.
Left picture: north up, magenta tick = shadow start (object base), cyan tick = shadow tip. Right: same shadow turned to point right.</span></header>
<div class="grid" id="g"></div>
<script>
const C=%(cards)s; const KEY="shb_labels_%(key)s"; let L={}; try{L=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
let sel=0;
function store(){try{localStorage.setItem(KEY,JSON.stringify(L))}catch(e){}}
function draw(){const f=document.getElementById("flt").value;const g=document.getElementById("g");g.innerHTML="";
C.forEach((c,i)=>{const v=L[c.name]||"";if(f=="u"&&v)return;if(f=="y"&&v!="Y")return;if(f=="n"&&v!="N")return;
const d=document.createElement("div");d.className="card"+(i==sel?" sel":"")+(v=="Y"?" y":v=="N"?" n":"");d.id="c"+i;
d.innerHTML=`<div class="meta"><b>${c.name}</b> &nbsp; ${c.h==null?"":c.h+" m"} &nbsp; <span style="color:#aaa">${c.st}</span></div>
<div class="row"><img src="data:image/jpeg;base64,${c.imgs[0]}" width="230" height="230">${c.imgs[1]?`<img src="data:image/jpeg;base64,${c.imgs[1]}" width="288" height="96">`:""}</div>
<div class="meta"><button class="yes" onclick="lab(${i},'Y')">Yes (Y)</button> <button class="no" onclick="lab(${i},'N')">No (N)</button> <button class="un" onclick="lab(${i},'')">Unsure (U)</button>
&nbsp; ${v?("<b>"+(v=="Y"?"YES":"NO")+"</b>"):""} <span style="color:#888;font-size:12px">${c.lat}, ${c.lon}</span></div>`;
d.onclick=()=>{sel=i;mark()};g.appendChild(d)});count()}
function mark(){document.querySelectorAll(".card").forEach(e=>e.classList.remove("sel"));const e=document.getElementById("c"+sel);if(e){e.classList.add("sel");e.scrollIntoView({block:"nearest"})}}
function lab(i,v){if(v)L[C[i].name]=v;else delete L[C[i].name];store();sel=Math.min(C.length-1,i+1);draw();mark()}
function count(){const y=Object.values(L).filter(v=>v=="Y").length,n=Object.values(L).filter(v=>v=="N").length;
document.getElementById("cnt").textContent=`${C.length} candidates | yes ${y} | no ${n} | open ${C.length-y-n}`}
function save(){let t="name,label\\n";for(const k in L)t+=k+","+L[k]+"\\n";const a=document.createElement("a");
a.href=URL.createObjectURL(new Blob([t],{type:"text/csv"}));a.download="%(key)s_labels.csv";a.click()}
document.addEventListener("keydown",e=>{const k=e.key.toLowerCase();if(k=="y")lab(sel,"Y");else if(k=="n")lab(sel,"N");else if(k=="u")lab(sel,"");
else if(k=="arrowright"||k=="arrowdown"){sel=Math.min(C.length-1,sel+1);mark();e.preventDefault()}else if(k=="arrowleft"||k=="arrowup"){sel=Math.max(0,sel-1);mark();e.preventDefault()}});
draw();
</script></body></html>"""
    key = os.path.basename(base)
    with open(base + "_review.html", "w", encoding="utf-8") as f:
        f.write(page % dict(title=_html.escape(title), cards=_json.dumps(cards), key=key))
    return len(cards)


# ======================================================================= second image (basemap / WMS / WMTS / file) check
def normalize_service_url(u):
    """Accept what people copy: a Map Viewer / Scene Viewer link (...?url=<service>), a REST page URL with ?f=..., spaces."""
    import urllib.parse
    u = (u or "").strip().strip('"')
    m_ = re.search(r"[?&]url=(https?(?:%3A|:)[^&\s]+)", u, re.I)
    if m_:
        return normalize_service_url(urllib.parse.unquote(m_.group(1)))
    if not u.lower().startswith("http"):
        return u
    q = urllib.parse.urlparse(u)
    qs = urllib.parse.parse_qs(q.query)
    for k in ("url", "URL", "layers_url", "service"):
        if k in qs and qs[k] and qs[k][0].lower().startswith("http"):
            return normalize_service_url(urllib.parse.unquote(qs[k][0]))
    if "/mapserver" in u.lower() or "/imageserver" in u.lower():
        base = u.split("?")[0].rstrip("/")
        keep = {k: v for k, v in qs.items() if k.lower() == "token"}
        return base + ("?" + urllib.parse.urlencode({k: v[0] for k, v in keep.items()}) if keep else "")
    return u


class SecondImage:
    """A second, independent view: an image file, an ArcGIS MapServer (cached tiles), an XYZ/WMTS tile URL with {z}/{x}/{y},
    or a WMS GetMap URL. Small windows are fetched around each candidate and saved as temporary GeoTIFFs (EPSG:4326)."""
    R = 6378137.0

    def __init__(self, spec, workdir, zoom=19, token=None, log=print):
        import tempfile
        self.spec = normalize_service_url(spec); self.zoom = int(zoom); self.token = token; self.log = log
        self.dir = tempfile.mkdtemp(prefix="shb2_", dir=workdir if workdir and os.path.isdir(workdir) else None)
        self.kind = "file"
        u = self.spec.lower()
        if u.startswith("http"):
            if "/mapserver" in u and "{z}" not in u:
                self.kind = "arcgis"; self.base = self.spec.split("?")[0].rstrip("/")
                if self.base.lower().endswith("/tile"):
                    self.base = self.base[:-5]
            elif "{z}" in u or "{tilematrix}" in u:
                self.kind = "xyz"
            elif "wms" in u:
                self.kind = "wms"
            else:
                raise ValueError("Second image URL not recognised. Use an ArcGIS .../MapServer URL, an XYZ/WMTS URL with {z}/{x}/{y}, or a WMS URL.")
        elif not os.path.exists(self.spec):
            raise ValueError("Second image file not found: " + self.spec)
        self.n = 0; self._meta = {}

    def _get(self, url):
        import urllib.request
        if self.token:
            url += ("&" if "?" in url else "?") + "token=" + self.token
        last = None
        for k in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": f"{APP} {VERSION}"})
                return urllib.request.urlopen(req, timeout=30).read()
            except Exception as e:
                last = e
        raise last

    def date(self, lat, lon):
        """Acquisition date / source of the second image at this place (ArcGIS World Imagery style identify), else ''."""
        if self.kind != "arcgis":
            return ""
        key = (round(lat, 3), round(lon, 3))
        if key in self._meta:
            return self._meta[key]
        val = ""
        try:
            import json as _j
            u = (f"{self.base}/identify?geometry={lon},{lat}&geometryType=esriGeometryPoint&sr=4326&layers=all&tolerance=1"
                 f"&mapExtent={lon - .001},{lat - .001},{lon + .001},{lat + .001}&imageDisplay=400,400,96&returnGeometry=false&f=json")
            res = _j.loads(self._get(u)).get("results", [])
            for r in res:
                a = r.get("attributes", {})
                d = a.get("DATE (YYYYMMDD)") or a.get("SRC_DATE") or a.get("SRC_DATE2")
                if d and d != "Null":
                    val = f"{d} {a.get('DESCRIPTION', '')}".strip(); break
        except Exception:
            pass
        self._meta[key] = val
        return val

    def fetch_grid(self, T, w, h):
        """Imagery on a geographic grid (affine T in EPSG:4326, w x h pixels) as float32 (3, h, w); zeros where missing."""
        from rasterio.warp import reproject, Resampling
        from rasterio.transform import from_origin
        import io
        out = np.zeros((3, h, w), np.float32)
        l, t = T.c, T.f; r_, b_ = l + w * T.a, t + h * T.e
        if self.kind == "file":
            with rasterio.open(self.spec) as s:
                bands = [1, 2, 3] if s.count >= 3 else [1, 1, 1]
                for i, b in enumerate(bands):
                    reproject(rasterio.band(s, b), out[i], dst_transform=T, dst_crs="EPSG:4326", resampling=Resampling.bilinear)
            return out
        if self.kind == "wms":
            base = self.spec; u = base
            if "request=getmap" not in base.lower():
                u = base + ("&" if "?" in base else "?") + "SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&STYLES=&FORMAT=image/jpeg&SRS=EPSG:4326"
            u += f"&BBOX={l},{b_},{r_},{t}&WIDTH={w}&HEIGHT={h}"
            data = self._get(u)
            try:
                im = Image.open(io.BytesIO(data)).convert("RGB")
            except Exception:
                raise ValueError("WMS did not return an image: " + data[:300].decode("utf-8", "replace"))
            return np.moveaxis(np.array(im.resize((w, h))).astype(np.float32), -1, 0)
        Z = self.zoom; O = math.pi * self.R; ts = 2 * O / 2 ** Z
        mx = lambda lo: self.R * math.radians(lo)
        my = lambda la: self.R * math.log(math.tan(math.pi / 4 + math.radians(la) / 2))
        tx0, tx1 = int((mx(l) + O) // ts), int((mx(r_) + O) // ts)
        ty0, ty1 = int((O - my(t)) // ts), int((O - my(b_)) // ts)
        mos = np.zeros(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256, 3), np.uint8)
        if not hasattr(self, "_cache"):
            self._cache = {}
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                key = (Z, tx, ty)
                if key not in self._cache:
                    if self.kind == "arcgis":
                        u = f"{self.base}/tile/{Z}/{ty}/{tx}"
                    else:
                        u = (self.spec.replace("{z}", str(Z)).replace("{x}", str(tx)).replace("{y}", str(ty))
                             .replace("{TileMatrix}", str(Z)).replace("{TileCol}", str(tx)).replace("{TileRow}", str(ty)))
                    try:
                        self._cache[key] = np.array(Image.open(io.BytesIO(self._get(u))).convert("RGB"))[:256, :256]
                    except Exception:
                        self._cache[key] = None
                    if len(self._cache) > 3000:
                        self._cache.pop(next(iter(self._cache)))
                tile = self._cache[key]
                if tile is not None:
                    mos[(ty - ty0) * 256:(ty - ty0) * 256 + tile.shape[0], (tx - tx0) * 256:(tx - tx0) * 256 + tile.shape[1]] = tile
        if mos.max() == 0:
            return out
        src_T = from_origin(tx0 * ts - O, O - ty0 * ts, ts / 256, ts / 256)
        for i in range(3):
            reproject(mos[:, :, i].astype(np.float32), out[i], src_transform=src_T, src_crs="EPSG:3857", dst_transform=T,
                      dst_crs="EPSG:4326", resampling=Resampling.bilinear)
        return out

    def window(self, lat, lon, half_m=60.0):
        """Path of a temporary GeoTIFF of the second view around (lat, lon) at 0.3 m, or None."""
        from rasterio.transform import from_origin
        res_m = 0.3
        n = int(2 * half_m / res_m)
        dlat = res_m / 110700.0; dlon = res_m / (111320.0 * math.cos(math.radians(lat)))
        T = from_origin(lon - half_m / (111320.0 * math.cos(math.radians(lat))), lat + half_m / 110700.0, dlon, dlat)
        out = self.fetch_grid(T, n, n)
        if out.max() <= 0:
            return None
        self.n += 1
        p = os.path.join(self.dir, f"w{self.n}.tif")
        with rasterio.open(p, "w", driver="GTiff", width=n, height=n, count=3, dtype="uint8", crs="EPSG:4326", transform=T) as d:
            d.write(np.clip(out, 0, 255).astype(np.uint8))
        return p

    def close(self):
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)


def is_service(x):
    return isinstance(x, str) and normalize_service_url(x).lower().startswith("http")


def extent_for_service(opt):
    """Lat/lon box to download from a service: the 'Limit to area' file, else the points file (+250 m)."""
    src = opt.get("area") or opt.get("points") or opt.get("mosques")
    if not src:
        raise ValueError("With an imagery service as the image, choose 'Limit to area' (a KML/KMZ/SHP/CSV of the area to process, "
                         "or an image file covering it) or a points file - the program downloads only that area.")
    lls = area_latlons(src) if src == opt.get("area") else [(p["lat"], p["lon"]) for p in read_points(src)]
    dn, de = opt.get("offset_n", 0.0) or 0.0, opt.get("offset_e", 0.0) or 0.0
    la = [a - dn / 110700.0 for a, b in lls]; lo = [b - de / (111320.0 * math.cos(math.radians(a))) for a, b in lls]
    buf = 0.0 if opt.get("area") else 250.0
    la0, la1 = min(la) - buf / 110700.0, max(la) + buf / 110700.0
    k = 111320.0 * math.cos(math.radians((la0 + la1) / 2))
    lo0, lo1 = min(lo) - buf / k, max(lo) + buf / k
    w_m, h_m = (lo1 - lo0) * k, (la1 - la0) * 110700.0
    if w_m < 100 or h_m < 100:
        # a point (or a line) was given as the area: use a square of at least 600 m around it
        cla, clo = (la0 + la1) / 2, (lo0 + lo1) / 2
        hw, hh = max(300.0, w_m / 2), max(300.0, h_m / 2)
        la0, la1, lo0, lo1 = cla - hh / 110700.0, cla + hh / 110700.0, clo - hw / k, clo + hw / k
        opt.get("_log", print)(f"The area file {os.path.basename(src)} holds a point / very small shape ({len(lls)} coordinate(s)) - "
                               f"using a {2 * hw:.0f} x {2 * hh:.0f} m square around it. Use a polygon or 2 corner points for a larger area.")
    return la0, lo0, la1, lo1


def download_service_image(opt, log=print, progress=lambda f: None, stop=lambda: False):
    """Download the processing area from a WMS / WMTS / ArcGIS MapServer into a GeoTIFF (EPSG:4326, ~0.3 m) in the output folder."""
    spec = opt["image"].strip()
    la0, lo0, la1, lo1 = extent_for_service(dict(opt, _log=log))
    log(f"Area: lat {la0:.5f} to {la1:.5f}, lon {lo0:.5f} to {lo1:.5f}")
    k = 111320.0 * math.cos(math.radians((la0 + la1) / 2))
    wm, hm = (lo1 - lo0) * k, (la1 - la0) * 110700.0
    km2 = wm * hm / 1e6
    lim = opt.get("max_service_km2", 25.0)
    if km2 > lim:
        raise ValueError(f"The area is {km2:.1f} km2 - larger than {lim:.0f} km2 per run. Split it into smaller areas "
                         "(several 'Limit to area' files) or use an exported image.")
    res_m = opt.get("service_res_m", 0.3)
    W, H = int(wm / res_m) + 1, int(hm / res_m) + 1
    from rasterio.transform import from_origin
    T = from_origin(lo0, la1, (lo1 - lo0) / W, (la1 - la0) / H)
    os.makedirs(opt["out_dir"], exist_ok=True)
    out = os.path.join(opt["out_dir"], "service_image.tif")
    src = SecondImage(spec, opt["out_dir"], zoom=opt.get("service_zoom", 19), token=opt.get("service_token") or opt.get("second_token"), log=log)
    log(f"Downloading {km2:.2f} km2 from the service ({src.kind}) at {res_m} m: {W} x {H} px ...")
    dates = set()
    try:
        for fy in (0.2, 0.5, 0.8):
            for fx in (0.2, 0.5, 0.8):
                d_ = src.date(la1 - fy * (la1 - la0), lo0 + fx * (lo1 - lo0))
                if d_:
                    dates.add(d_)
        B = 1024
        prof = dict(driver="GTiff", width=W, height=H, count=3, dtype="uint8", crs="EPSG:4326", transform=T, tiled=True,
                    blockxsize=256, blockysize=256, compress="deflate", BIGTIFF="IF_SAFER")
        blocks = [(r, c) for r in range(0, H, B) for c in range(0, W, B)]
        filled = 0
        with rasterio.open(out, "w", **prof) as dst:
            for i, (r, c) in enumerate(blocks):
                if stop():
                    raise ValueError("Stopped by user.")
                progress(i / len(blocks))
                h_, w_ = min(B, H - r), min(B, W - c)
                Tb = rasterio.transform.Affine(T.a, 0, T.c + c * T.a, 0, T.e, T.f + r * T.e)
                a = src.fetch_grid(Tb, w_, h_)
                filled += int((a.max(0) > 0).sum())
                dst.write(np.clip(a, 0, 255).astype(np.uint8), window=Window(c, r, w_, h_))
        progress(1.0)
    finally:
        src.close()
    if filled < 0.05 * W * H:
        raise ValueError("The service returned (almost) no imagery for this area - check the URL, layer name, token and zoom level.")
    log(f"Saved {os.path.basename(out)} ({100.0 * filled / (W * H):.0f}% covered).")
    if dates:
        log("Acquisition(s) in this area: " + ", ".join(sorted(dates)))
        if len(dates) > 1:
            log("WARNING: the area mixes several acquisitions with different sun angles - heights are only right for the "
                "acquisition matching the sun you set. Use smaller areas, one acquisition each.")
    return out, sorted(dates)


def prepare_image(opt, log=print, progress=lambda f: None, stop=lambda: False):
    """If the image is a service URL, download the area and fill the date from the service metadata when the sun is to come from the date."""
    if not is_service(opt.get("image")):
        return opt
    opt = dict(opt)
    path, dates = download_service_image(opt, log, progress, stop)
    opt["image"] = path
    opt["area"] = None                      # the downloaded image is exactly the area
    if opt.get("sun_elev") is None and opt.get("date") is None and not opt.get("known_ok") and dates:
        m_ = re.match(r"(\d{4})(\d{2})(\d{2})", dates[0])
        if m_:
            opt["date"] = dt.date(int(m_.group(1)), int(m_.group(2)), int(m_.group(3)))
            log(f"Sun from the service's acquisition date {opt['date']} (time solved from the shadow direction).")
    return opt


def _register(img, w_path, lat, lon, half_m=60.0):
    """Shift (metres east, north) that moves the second view onto the primary image, by phase correlation of edges."""
    w = Img(w_path)
    res_m = 0.5; n = int(2 * half_m / res_m)
    def grab(I):
        c, r = I.to_pixel(lat, lon); half = int(half_m / I.pm)
        a, x0, y0 = I.read(c, r, half)
        if a is None:
            return None
        g = cv2.resize(a.mean(0), (n, n), interpolation=cv2.INTER_AREA)
        g = cv2.GaussianBlur(g, (0, 0), 1.0)
        return np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)).astype(np.float32)
    g1, g2 = grab(img), grab(w)
    if g1 is None or g2 is None:
        return 0.0, 0.0, 0.0
    (dx, dy), resp = cv2.phaseCorrelate(g1, g2, cv2.createHanningWindow((n, n), cv2.CV_32F))
    if math.hypot(dx, dy) * res_m > 15:                    # implausible jump: do not shift
        return 0.0, 0.0, float(resp)
    return dx * res_m, -dy * res_m, float(resp)


def rank_key(rp):
    """Order of automatic results: model class first; inside a class, candidates the second image agrees with come first.
    (Riyadh test: 'likely + 2nd image agrees' ~7 in 10 real, 'possible + agrees' ~1-3 in 10, 'unlikely + agrees' ~0.)"""
    r, p = rp
    st = r.get("status", "")
    cls = {"likely": 0, "possible": 1, "unlikely": 2}.get(st.split()[0], 1) if st else 1
    agree = 0 if "2nd image agrees" in st else 1
    return (cls, agree, -(p.get("streak") or {}).get("mscore", 0.0), -(r.get("height_m") or 0))


def second_view_check(rows, img, opt, el1, log=print, progress=lambda f: None, stop=lambda: False):
    """Measure every candidate again in the second image. A real tall object casts a shadow from the same base in both
    images; its height from the second image must agree. Adds h2_m / second_check columns and adjusts the status."""
    src = SecondImage(opt["second"], opt.get("out_dir"), zoom=opt.get("second_zoom", 19), token=opt.get("second_token"), log=log)
    log(f"Second image: {opt['second']} ({src.kind})")
    meas = []
    try:
        todo = [(r, p) for r, p in rows if p.get("streak")]
        for i, (r, p) in enumerate(todo):
            if stop():
                raise ValueError("Stopped by user.")
            progress(i / max(1, len(todo)))
            s = p["streak"]
            lat, lon = img.to_latlon(*s["start"])
            try:
                wp = src.window(lat, lon)
            except Exception as ex:
                wp = None
                if i == 0:
                    log("  second image fetch failed: " + str(ex))
            if not wp:
                meas.append(None); continue
            de_, dn_, resp = _register(img, wp, lat, lon)
            w = Img(wp)
            la2 = lat - dn_ / 110700.0; lo2 = lon - de_ / (111320.0 * math.cos(math.radians(lat)))
            c2, r2 = w.to_pixel(la2, lo2)
            cs = find_streak(w, c2, r2, s["az"], 7.0, opt["max_shadow_m"], spread=45, step=3, collect=True, min_len=3.0)
            cs = [x for x in cs if 0.8 <= x["width_m"] <= 4.0]
            meas.append(dict(date=src.date(lat, lon), cands=[(x["length_m"], x["az"], x["start_dist_m"]) for x in cs], shift=(de_, dn_)))
            if (i + 1) % 50 == 0:
                log(f"  {i + 1} of {len(todo)} candidates checked in the second image")
        progress(1.0)
    finally:
        src.close()
    # calibrate the second image's sun (direction + elevation) per acquisition from the best-ranked candidates
    todo = [(r, p) for r, p in rows if p.get("streak")]
    groups = {}
    for (r, p), m in zip(todo, meas):
        if m:
            groups.setdefault(m["date"], []).append((r, p, m))
    t1 = math.tan(el1 * D2R)
    for g, items in groups.items():
        az2, tan2 = opt.get("second_az"), (math.tan(opt["second_elev"] * D2R) if opt.get("second_elev") else None)
        good = sorted(items, key=lambda it: -(it[1]["streak"].get("mscore", 0.0)))[:max(8, len(items) // 3)]
        mm = re.match(r"(\d{4})(\d{2})(\d{2})", g or "")
        if mm and az2 is None and tan2 is None:
            # acquisition date known: the sun moves along a known path that day - find the time of day at which the most
            # candidates have a matching shadow (direction within 10 deg, height within 25 %)
            day = dt.datetime(int(mm.group(1)), int(mm.group(2)), int(mm.group(3))) - dt.timedelta(hours=opt.get("tz", 3.0))
            la_ = float(np.mean([img.to_latlon(*it[1]["streak"]["start"])[0] for it in items]))
            lo_ = float(np.mean([img.to_latlon(*it[1]["streak"]["start"])[1] for it in items]))
            bestt = None
            for minute in range(6 * 60 + 30, 15 * 60 + 31, 4):
                e_, a_ = sun_position(day + dt.timedelta(minutes=minute), la_, lo_)[:2]
                if e_ < 15:
                    continue
                sa, tn = (a_ + 180) % 360, math.tan(e_ * D2R)
                sc_ = sum(1 for r, p, m in items
                          if any(abs(wrap180(c[1] - sa)) <= 10 and abs(c[0] * tn - r["height_m"]) <= 0.25 * max(r["height_m"], 1) for c in m["cands"]))
                if bestt is None or sc_ > bestt[0]:
                    bestt = (sc_, sa, tn, minute, e_)
            if bestt and bestt[0] >= 3:
                az2, tan2 = bestt[1], bestt[2]
                log(f"  second image {g}: acquired about {bestt[3] // 60 + int(opt.get('tz', 3.0)) - int(opt.get('tz', 3.0)):02d}:{bestt[3] % 60:02d} local "
                    f"(sun {bestt[4]:.1f} deg) - fits {bestt[0]} candidates")
        if az2 is None:
            a_all = [c[1] for it in good for c in it[2]["cands"][:2] if c[0] >= 4]
            if len(a_all) < 6:
                a_all = [c[1] for it in items for c in it[2]["cands"][:2] if c[0] >= 4]
            if len(a_all) >= 3:
                az2 = max(np.arange(0, 360, 1.0), key=lambda a0: sum(abs(wrap180(a - a0)) <= 8 for a in a_all))
        if tan2 is None and az2 is not None:
            rat = []
            for r, p, m in good:
                c = [c for c in m["cands"] if abs(wrap180(c[1] - az2)) <= 12 and c[0] >= 4]
                if c:
                    rat.append(p["streak"]["length_m"] * t1 / max(c, key=lambda c: c[0])[0])
            tan2 = float(np.median(rat)) if len(rat) >= 4 else None
        if tan2 is None and az2 is not None:                  # fall back to the sun position for the acquisition date
            mm = re.match(r"(\d{4})(\d{2})(\d{2})", g or "")
            if mm:
                la_ = float(np.mean([img.to_latlon(*it[1]["streak"]["start"])[0] for it in items]))
                lo_ = float(np.mean([img.to_latlon(*it[1]["streak"]["start"])[1] for it in items]))
                e2, _ = elevation_from_date(dt.date(int(mm.group(1)), int(mm.group(2)), int(mm.group(3))), opt.get("tz", 3.0), la_, lo_, az2, None)
                if e2:
                    tan2 = math.tan(e2 * D2R)
        if az2 is None or tan2 is None:
            log(f"  second image {g or ''}: could not calibrate its sun - enter 2nd-image sun elevation/azimuth"); continue
        log(f"  second image {g or '(one acquisition)'}: {len(items)} candidates, its shadow direction {az2 % 360:.0f} deg, sun elevation {math.degrees(math.atan(tan2)):.1f} deg")
        for r, p, m in items:
            p["_m2"] = dict(m, az2=az2, tan2=tan2)
            h1 = r["height_m"]
            c = [c for c in m["cands"] if abs(wrap180(c[1] - az2)) <= 15]
            if not c:
                r["second_check"] = "no matching shadow"; r["h2_m"] = ""
            else:
                best = min(c, key=lambda c: abs(c[0] * tan2 - h1))
                h2 = best[0] * tan2
                r["h2_m"] = round(h2, 1)
                r["second_check"] = "agrees" if abs(h2 - h1) <= 0.35 * max(h1, 1.0) else "different height"
            r["second_date"] = g
    n_ag = 0
    for r, p in rows:
        chk = r.get("second_check")
        if not chk:
            continue
        if chk == "agrees":
            n_ag += 1
            r["status"] = r["status"] + " - 2nd image agrees"
        else:
            r["status"] = r["status"] + f" - 2nd image: {chk}"
    log(f"Second image check: {n_ag} of {len(todo)} candidates confirmed (same base, matching shadow and height).")
    return n_ag


# ======================================================================= automatic scan (no points needed)
RASTER_EXT = (".tif", ".tiff", ".jp2", ".img", ".ecw", ".sid", ".png", ".jpg")


def area_latlons(path):
    """Coordinates (lat, lon) of an area file: KML/KMZ, SHP (WGS84), CSV/TXT points, or an image file (its four corners)."""
    lls = []
    ext = os.path.splitext(path)[1].lower()
    if ext in RASTER_EXT:
        from rasterio.warp import transform as _tr
        with rasterio.open(path) as d:
            b = d.bounds
            xs, ys = [b.left, b.right, b.right, b.left], [b.top, b.top, b.bottom, b.bottom]
            if d.crs and not d.crs.is_geographic:
                xs, ys = _tr(d.crs, "EPSG:4326", xs, ys)
        return [(y, x) for x, y in zip(xs, ys)]
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
        try:
            lls = [(p["lat"], p["lon"]) for p in read_points(path)]
        except Exception:
            lls = []
    if not lls:
        raise ValueError(f"'Limit to area' file {os.path.basename(path)} has no coordinates I can read. Use a KML/KMZ, a WGS84 shapefile, "
                         "a CSV with lat/lon columns (corner points are enough), or an image file whose extent is the area.")
    return lls


def read_aoi(path, img, buffer_m=0, dn=0.0, de=0.0):
    """Bounding box (col0,row0,col1,row1) in image pixels of the features in an area file."""
    lls = area_latlons(path)
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
    model = load_model(opt.get("model")) if opt.get("use_model", True) else None
    if model is not None:
        log(f"Ranking with the learned model ({int(model['n_pos'])} confirmed / {int(model['n_neg'])} rejected examples)")
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
        ch = make_chip(img, s["start"], s["end"], s["width_m"])
        vec = chip_vector(ch, s["length_m"], s["width_m"], s["contrast"], 0.0) if ch is not None else None
        ms = model_score(model, vec) if (model is not None and vec is not None) else None
        if ms is not None:
            s["mscore"] = ms
            st = model_class(model, ms) + " (model)" + ("" if st == "ok" else " / " + st.replace("check: ", ""))
        r = dict(status=st, height_m=round(h, 1), height_low_m=round(lo_, 1), height_high_m=round(hi_, 1), height_ft=round(h * FT, 1),
                 shadow_m=round(s["length_m"], 2), shadow_az=round(s["az"], 1), sun_elev=round(el, 2), lat=round(bl, 7), lon=round(bo, 7),
                 dms=f"{dms(bo, 'E', 'W')} {dms(bl, 'N', 'S')}", tip_lat=round(tl, 7), tip_lon=round(to, 7), sun_source=el_src,
                 input_lat="", input_lon="", known_height_m="", width_m=round(s["width_m"], 2),
                 model_score=round(ms, 2) if ms is not None else "")
        p = dict(inside=True, col=s["start"][0], row=s["start"][1], streak=s, chip=ch, vec=vec)
        rows.append((r, p))
    if opt.get("second"):
        second_view_check(rows, img, opt, el, log=log, progress=progress, stop=stop)
    rows.sort(key=rank_key)
    cap = int(opt.get("max_results", 1000))
    if len(rows) > cap:
        log(f"Keeping the best {cap} of {len(rows)} candidates (ranked) - in dense city blocks most of the rest are walls and gaps.")
        rows = rows[:cap]
    for i, (r, p) in enumerate(rows, 1):
        r["name"] = f"AUTO-{i:04d}"
    from collections import Counter as _C
    _c = _C(r["status"].split()[0] for r, _ in rows)
    log(f"{len(rows)} objects at or above {min_h:.0f} m (" + ", ".join(f"{v} {k}" for k, v in _c.most_common()) + ")")
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
    # the built-in list is OpenStreetMap (true WGS84, like the image); a file of yours is in your reference frame
    sdn, sde = (dn, de) if opt.get("mosques") else (0.0, 0.0)
    for la, lo, rad in sites:
        c, r = img.to_pixel(la - sdn / 110700.0, lo - sde / (111320.0 * math.cos(la * D2R)))
        if C0 <= c < C1 and R0 <= r < R1:
            inside.append((la, lo, rad, c, r))
    log(f"{len(inside)} mosques inside the {'area' if opt.get('area') else 'image'} "
        f"({'built-in OpenStreetMap list' if not opt.get('mosques') else os.path.basename(src)})")
    if not inside:
        raise ValueError("No mosques fall inside the image / area.")
    shaz = opt.get("shadow_az")
    spread = opt.get("spread", 20)
    la_c = float(np.mean([x[0] for x in inside])); lo_c = float(np.mean([x[1] for x in inside]))
    if shaz is None:
        shaz = detect_shadow_azimuth(img, [(int(c), int(r)) for *_, c, r in inside], log)
        if shaz is None:
            raise ValueError("Could not detect the shadow direction; enter it manually.")
        # the image-based estimate can be 20 deg off in dense blocks: search a wide fan around it
        spread = opt.get("spread", 35)
        log("Tip: enter the shadow direction (sun azimuth from the metadata + 180) for faster, cleaner results.")
    el, el_src = sun_for_scan(opt, img, shaz, dn, de, la_c, lo_c, log)
    log(f"Sun elevation used: {el:.2f} deg  ({el_src})")
    tan = math.tan(el * D2R)
    min_h = opt.get("min_height") or 0.0
    min_len = max(4.0, 0.9 * min_h / tan)
    model = load_model(opt.get("model")) if opt.get("use_model", True) else None
    if model is not None:
        log(f"Using learned model ({int(model['n_pos'])} confirmed / {int(model['n_neg'])} rejected examples, AUC {float(model['cv_auc']):.2f})")
    else:
        log("No learned model - candidates are ranked by shadow length. Label the review page and use 'Train' to create one.")
    per = opt.get("per_mosque", 3 if model is not None else 2)
    max_h = opt.get("max_height", 60.0)
    cands = []
    for i, (la, lo, rad, c, r) in enumerate(inside):
        if stop():
            raise ValueError("Stopped by user.")
        progress(i / len(inside))
        search = min(90.0, rad + 12.0)
        cs = find_streak(img, c, r, shaz, search, opt["max_shadow_m"], spread=spread, collect=True, min_len=min_len)
        # thin streaks (< 1.3 m) are mostly gaps between buildings, kerbs or fences, not minarets
        cs = [x for x in cs if x["width_m"] >= AUTO_MIN_WIDTH]
        for x in cs:
            x["score"] = x["length_m"] - 0.3 * x["start_dist_m"]
            x["mosque"] = i; x["site"] = (la, lo)
        cs = dedupe_streaks(cs, img.pm)
        cs.sort(key=lambda x: -x["score"])
        if model is not None:                       # learned ranking: look at the 6 longest, keep the best by model score
            cs = cs[:6]
            for x in cs:
                x["chip"] = make_chip(img, x["start"], x["end"], x["width_m"])
                if x["chip"] is not None:
                    x["vec"] = chip_vector(x["chip"], x["length_m"], x["width_m"], x["contrast"], x["start_dist_m"])
                    x["mscore"] = model_score(model, x["vec"])
                else:
                    x["mscore"] = -99.0
            cs.sort(key=lambda x: -x["mscore"])
        for rank, x in enumerate(cs[:per], 1):
            x["rank"] = rank
        cands += cs[:per]
        if (i + 1) % 50 == 0:
            log(f"  {i + 1} of {len(inside)} mosques searched, {len(cands)} candidate shadows")
    progress(1.0)
    if model is None:                                # no model: chips are still saved so the results can be labelled
        for s in cands:
            s["chip"] = make_chip(img, s["start"], s["end"], s["width_m"])
            if s["chip"] is not None:
                s["vec"] = chip_vector(s["chip"], s["length_m"], s["width_m"], s["contrast"], s["start_dist_m"])
    found = dedupe_streaks(cands, img.pm)
    rows = []
    for s in found:
        h = s["length_m"] * tan
        if h < min_h:
            continue
        if model is not None:
            st = model_class(model, s.get("mscore", -99.0)) + " (model)"
        else:
            st = "likely - verify" if s.get("rank", 1) == 1 else "possible - 2nd shadow at this mosque"
            if s["width_m"] < AUTO_OK_WIDTH:
                st = "possible - thin, may be a gap between buildings"
        if h > max_h:
            st = "unlikely - longer than a minaret: wall or kerb shadow?"
        rr = streak_row(s, img, el, el_src, dn, de, st)
        rr.update(width_m=round(s["width_m"], 2), mosque_lat=round(s["site"][0], 6), mosque_lon=round(s["site"][1], 6),
                  mosque=s.get("mosque"), dist=round(s["start_dist_m"], 1),
                  model_score=round(s["mscore"], 2) if "mscore" in s else "")
        rows.append((rr, dict(inside=True, col=s["start"][0], row=s["start"][1], streak=s, chip=s.get("chip"), vec=s.get("vec"))))
    if opt.get("second"):
        second_view_check(rows, img, opt, el, log=log, progress=progress, stop=stop)
    rows.sort(key=rank_key)
    for i, (r, p) in enumerate(rows, 1):
        r["name"] = f"MNR-{i:04d}"
    n_l = sum(1 for r, _ in rows if r["status"].startswith("likely"))
    log(f"{len(rows)} candidate minaret shadows at or above {min_h:.0f} m around {len(inside)} mosques ({n_l} likely). "
        "Open the review page, mark Yes/No and train: the model learns your imagery.")
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
    # second-image check (here the same image again, so every tower must be confirmed)
    try:
        o = dict(image=tif, out_dir=os.path.join(folder, "selftest_second"), max_shadow_m=80, sun_elev=el, shadow_az=None, min_height=10,
                 mosques=mcsv, second=tif)
        got = run_mosques(o, log=print)
        n_c = sum(1 for r in got if "2nd image agrees" in r["status"])
        print(f"Second image check: {n_c} of {len(got)} confirmed", "OK" if n_c == len(towers) else "FAILED")
        ok = ok and n_c == len(towers)
    except Exception as ex:
        print("Second image test FAILED:", ex); traceback.print_exc(); ok = False
    # training: label the 6 towers found by the mosque search as confirmed, learn together with the starter examples
    try:
        fo = os.path.join(folder, "selftest_mosque")
        names = [r["name"] for r in csv.DictReader(open(os.path.join(fo, "minaret_scan_heights.csv"), encoding="utf-8-sig"))]
        with open(os.path.join(fo, "minaret_scan_heights_labels.csv"), "w", newline="") as f:
            f.write("name,label\n" + "".join(f"{n},Y\n" for n in names))
        mp = run_train([fo], log=print, model_out=os.path.join(folder, "selftest_model.npz"))
        mm = load_model(mp)
        good = mm is not None and int(mm["n_pos"]) >= 6 and os.path.exists(os.path.join(folder, "selftest_mosque", "minaret_scan_heights_review.html"))
        print("Training + review page:", "OK" if good else "FAILED")
        ok = ok and good
    except Exception as ex:
        print("Training test FAILED:", ex); traceback.print_exc(); ok = False
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
    root = tk.Tk(); root.title(f"{APP} {VERSION}"); root.geometry("860x900")
    v = {k: tk.StringVar() for k in ("image", "points", "mosques", "area", "out", "elev", "date", "time", "tz", "az", "search", "maxsh", "minh", "dn", "de", "train",
                                     "second", "token", "elev2", "az2")}
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
    ttk.Radiobutton(jf, text="TRAIN: learn from reviewed results (folders where you saved labels from the review page)",
                    variable=job, value="train").grid(row=3, column=0, sticky="w")
    tf = ttk.Frame(jf); tf.grid(row=4, column=0, sticky="we")
    ttk.Label(tf, text="Results folder(s) to learn from:").pack(side="left")
    ttk.Entry(tf, textvariable=v["train"], width=58).pack(side="left", padx=4)
    def add_train():
        d = filedialog.askdirectory()
        if d:
            v["train"].set((v["train"].get() + ";" if v["train"].get().strip() else "") + d)
    ttk.Button(tf, text="Add folder...", command=add_train).pack(side="left")

    def row(r, label, var, browse=None, width=60):
        ttk.Label(frm, text=label).grid(row=r, column=0, sticky="w", pady=3)
        ttk.Entry(frm, textvariable=var, width=width).grid(row=r, column=1, sticky="we", pady=3)
        if browse:
            ttk.Button(frm, text="Browse...", command=browse).grid(row=r, column=2, padx=4)
    vec = [("Vector", "*.csv *.txt *.shp *.kml *.kmz"), ("All", "*.*")]
    row(1, "Image (file or service URL)", v["image"], lambda: v["image"].set(filedialog.askopenfilename(filetypes=[("Images", "*.tif *.tiff *.jp2 *.img *.ecw"), ("All", "*.*")])))
    row(2, "Points (CSV / SHP / KML)", v["points"], lambda: v["points"].set(filedialog.askopenfilename(filetypes=vec)))
    row(3, "Mosques file (optional)", v["mosques"], lambda: v["mosques"].set(filedialog.askopenfilename(filetypes=vec)))
    row(4, "Limit to area (optional)", v["area"], lambda: v["area"].set(filedialog.askopenfilename(filetypes=vec)))
    row(5, "Output folder", v["out"], lambda: v["out"].set(filedialog.askdirectory()))
    sf = ttk.LabelFrame(frm, text="Second image check (optional) - basemap / WMS / WMTS / ArcGIS MapServer URL, or an image file of another date", padding=6)
    sf.grid(row=9, column=0, columnspan=3, sticky="we", pady=4)
    ttk.Entry(sf, textvariable=v["second"], width=78).grid(row=0, column=0, columnspan=5, sticky="we")
    ttk.Button(sf, text="File...", command=lambda: v["second"].set(filedialog.askopenfilename(filetypes=[("Images", "*.tif *.tiff *.jp2"), ("All", "*.*")]))).grid(row=0, column=5, padx=4)
    ttk.Label(sf, text="token (if the service needs one)").grid(row=1, column=0, sticky="w"); ttk.Entry(sf, textvariable=v["token"], width=22).grid(row=1, column=1, sticky="w")
    ttk.Label(sf, text="its sun elevation / shadow dir. (blank = calibrate)").grid(row=1, column=2, sticky="e")
    ttk.Entry(sf, textvariable=v["elev2"], width=6).grid(row=1, column=3, sticky="w"); ttk.Entry(sf, textvariable=v["az2"], width=6).grid(row=1, column=4, sticky="w")
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
    opts = (("Shadow dir. deg (sun azimuth+180; blank=detect)", "az"), ("Search radius around each point (m)", "search"), ("Longest shadow (m)", "maxsh"),
            ("Report heights above (m)", "minh"), ("Your data minus image, north (m)", "dn"), ("Your data minus image, east (m)", "de"))
    for i, (lab, k) in enumerate(opts):
        ttk.Label(of, text=lab).grid(row=i // 2, column=(i % 2) * 2, sticky="w", padx=4, pady=2)
        ttk.Entry(of, textvariable=v[k], width=10).grid(row=i // 2, column=(i % 2) * 2 + 1, sticky="w", pady=2)
    pb = ttk.Progressbar(frm, mode="determinate", maximum=100); pb.grid(row=14, column=0, columnspan=3, sticky="we", pady=8)
    txt = tk.Text(frm, height=12, wrap="word"); txt.grid(row=16, column=0, columnspan=3, sticky="nsew")
    frm.columnconfigure(1, weight=1); frm.rowconfigure(16, weight=1)

    def log(m):
        root.after(0, lambda: (txt.insert("end", m + "\n"), txt.see("end")))

    def num(k, default=None):
        s = v[k].get().strip()
        return float(s) if s else default

    def work():
        try:
            j = job.get()
            if j == "train":
                folders = [x.strip() for x in v["train"].get().split(";") if x.strip()]
                if not folders:
                    raise ValueError("Add at least one results folder that contains *_labels.csv from the review page.")
                outm = run_train(folders, log=log)
                root.after(0, lambda: messagebox.showinfo(APP, "Model trained and saved:\n" + outm))
                return
            out = v["out"].get() or os.path.join(os.path.dirname(v["image"].get()), "shadow_height_output")
            o = dict(image=v["image"].get(), points=v["points"].get().strip() or None, out_dir=out,
                     search_m=num("search", 25), max_shadow_m=num("maxsh", 80), min_height=num("minh"), shadow_az=num("az"),
                     offset_n=num("dn", 0), offset_e=num("de", 0), tz=num("tz", 3), use_known=mode.get() == "known",
                     mosques=v["mosques"].get().strip() or None, area=v["area"].get().strip() or None,
                     second=v["second"].get().strip() or None, second_token=v["token"].get().strip() or None,
                     second_elev=num("elev2"), second_az=num("az2"))
            if not o["image"]:
                raise ValueError("Choose the image: a GeoTIFF / JP2 file, or paste an imagery service URL (WMS / WMTS / ArcGIS MapServer) "
                                 "into the Image box together with a 'Limit to area' file.")
            if is_service(o["image"]) and not v["out"].get().strip():
                raise ValueError("With an imagery service as the image, choose an output folder.")
            if j == "points" and not o["points"]:
                raise ValueError("Choose the points file (or pick an automatic mode).")
            elev_txt, date_txt = v["elev"].get().strip(), v["date"].get().strip()
            sun = mode.get()
            if sun == "known":
                n_known = 0
                if o["points"]:
                    try:
                        n_known = sum(1 for q in read_points(o["points"]) if q.get("known_height"))
                    except Exception:
                        n_known = 0
                if n_known < 2:
                    if elev_txt:
                        sun = "elev"; log(f"No points with 2+ known heights - using the sun elevation you entered ({elev_txt} deg).")
                    elif date_txt:
                        sun = "date"; log(f"No points with 2+ known heights - using the image date you entered ({date_txt}).")
                    elif is_service(o["image"]):
                        sun = "service"; o["use_known"] = False
                    else:
                        raise ValueError("Sun elevation: there is no points file with 2+ known heights. Select 'Sun elevation from metadata' "
                                         "and type it, or 'Image date' and type the date.")
            if sun == "elev":
                o["sun_elev"] = num("elev"); o["use_known"] = False
                if o["sun_elev"] is None:
                    raise ValueError("Enter the sun elevation.")
            if sun == "date":
                o["use_known"] = False
                try:
                    o["date"] = dt.date.fromisoformat(date_txt)
                except Exception:
                    raise ValueError("Image date must look like 2025-03-30.")
                if v["time"].get().strip():
                    hh, mm = v["time"].get().strip().split(":")[:2]; o["time"] = dt.time(int(hh), int(mm))
            prog = lambda f: root.after(0, lambda: pb.configure(value=f * 100))
            stop = lambda: stop_flag["stop"]
            o = prepare_image(o, log=log, progress=prog, stop=stop)
            if o.get("sun_elev") is None and o.get("date") is None and not o.get("use_known", True):
                raise ValueError("The service gives no acquisition date here - enter the sun elevation or the image date.")
            if j == "points":
                run_batch(o, log=log, progress=prog)
            elif j == "mosques":
                run_mosques(o, log=log, progress=prog, stop=stop)
            else:
                run_scan(o, log=log, progress=prog, stop=stop)
            root.after(0, lambda: messagebox.showinfo(APP, "Finished. Open the review page (HTML) or sheet (JPG) in the output folder:\n" + out))
        except Exception as e:
            log("ERROR: " + str(e)); log(traceback.format_exc())
            root.after(0, lambda: messagebox.showerror(APP, str(e)))
        finally:
            root.after(0, lambda: (run_btn.configure(state="normal"), stop_btn.configure(state="disabled")))

    def start():
        txt.delete("1.0", "end"); run_btn.configure(state="disabled"); stop_btn.configure(state="normal"); stop_flag["stop"] = False
        threading.Thread(target=work, daemon=True).start()
    bf = ttk.Frame(frm); bf.grid(row=15, column=0, columnspan=3, pady=4)
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
    ap.add_argument("--second", help="second view: image file, ArcGIS .../MapServer URL, XYZ/WMTS URL with {z}/{x}/{y}, or WMS URL")
    ap.add_argument("--second-token"); ap.add_argument("--second-zoom", type=int, default=19)
    ap.add_argument("--second-elev", type=float); ap.add_argument("--second-az", type=float)
    ap.add_argument("--test-second", metavar="URL", help="check that a second-image service can be read (fetches one window in Riyadh)")
    ap.add_argument("--train", nargs="+", metavar="FOLDER", help="learn from reviewed result folders (labels from the review page)")
    ap.add_argument("--model", help="model file to use / write (default: shadow_model.npz next to the program)")
    ap.add_argument("--no-model", action="store_true", help="rank automatic results by length only")
    ap.add_argument("--mosques-auto", action="store_true", help="find minarets around mosques of the built-in OpenStreetMap list")
    ap.add_argument("--mosques", help="your own mosques file (points or outlines: CSV/SHP/KML) for the minaret search")
    ap.add_argument("--spread", type=float, default=20); ap.add_argument("--step", type=float, default=4)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest(a.selftest) else 1)
    if a.train:
        run_train(a.train, model_out=a.model)
        return
    if a.test_second:
        src = SecondImage(a.test_second, None, zoom=a.second_zoom, token=a.second_token)
        try:
            pth = src.window(24.7248, 46.8440, 40)
            ok_ = bool(pth) and Img(pth).ds.read(1).max() > 0
            print("second image window:", "OK" if ok_ else "EMPTY", "| acquisition:", src.date(24.7248, 46.8440) or "(no metadata)")
        finally:
            src.close()
        sys.exit(0 if ok_ else 1)
    o = dict(image=a.image, points=a.points, out_dir=a.out, search_m=a.search, max_shadow_m=a.max_shadow, min_height=a.min_height,
             shadow_az=a.shadow_az, offset_n=a.offset_n, offset_e=a.offset_e, tz=a.tz, use_known=not a.no_known, sun_elev=a.sun_elev,
             model=a.model, use_model=not a.no_model, second=a.second, second_token=a.second_token, second_zoom=a.second_zoom,
             second_elev=a.second_elev, second_az=a.second_az)
    if a.date:
        o["date"] = dt.date.fromisoformat(a.date)
        if a.time:
            hh, mm = a.time.split(":")[:2]; o["time"] = dt.time(int(hh), int(mm))
    if is_service(o.get("image")):
        o["area"] = a.area; o["mosques"] = a.mosques
        if not o.get("out_dir"):
            raise SystemExit("--out is required with an imagery service")
        o = prepare_image(o)
    if a.mosques_auto or a.mosques:
        o.setdefault("area", a.area); o["mosques"] = a.mosques
        run_mosques(o)
    elif a.scan:
        o.setdefault("area", a.area); o["spread"] = a.spread; o["step"] = a.step
        run_scan(o)
    else:
        run_batch(o)


if __name__ == "__main__":
    main()
