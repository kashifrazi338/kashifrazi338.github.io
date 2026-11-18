SHADOW HEIGHT BATCH 2.0  -  heights of minarets, towers, poles and pylons from satellite shadows
================================================================================================

WHAT IT DOES
Finds the thin dark shadow of a tall object, measures its length and converts it to height:
height = shadow length x tan(sun elevation).
Outputs: CSV, shapefile (points + shadow lines), KML for Google Earth, and a review sheet (JPG)
with an image chip of every result so you can check each shadow by eye.

START
Double-click ShadowHeightBatch.exe (no installation, no Python needed).
Windows may show "Windows protected your PC" the first time: click More info -> Run anyway.
First start takes 10-20 seconds (the program unpacks itself).

THREE MODES ("What to do" at the top of the window)
1. Measure heights at my points             - most accurate. You give a point list (on/next to each object).
2. AUTOMATIC: find minarets around mosques  - no points needed. Uses the built-in list of ~21,000 mosques
                                              (Saudi Arabia and Gulf, OpenStreetMap), or your own mosques file
                                              (points or building outlines: SHP / KML / CSV).
                                              Up to 2 candidate shadows per mosque.
3. AUTOMATIC: scan whole image / area       - finds every thin tall-object shadow. Use it for pylons, masts and
                                              poles in OPEN ground (desert, highways). In dense city blocks it gives
                                              hundreds of false hits (gaps between buildings look the same).

IMPORTANT FOR THE AUTOMATIC MODES - ALWAYS CHECK THE REVIEW SHEET
In dense Riyadh blocks, narrow gaps between buildings cast thin shadows exactly like minarets.
Test on Riyadh imagery (56 mosques, 25 checked minarets): the minaret search found 21 of the 25.
About 1 in 3 "likely" results and 1 in 12 "possible" results is a real minaret; the rest are walls/gaps.
So: open <name>_review.jpg, look at each chip, and delete the wrong rows (by NAME) in the CSV/shapefile.
For final, accurate heights run mode 1 on the confirmed points.

INPUTS
Image   GeoTIFF (.tif) or JPEG 2000 (.jp2), georeferenced, 0.3-0.6 m pixels work best.
        ECW is not supported - export it first (ArcGIS Pro: Export Raster -> TIFF).
Points  CSV/TXT, shapefile (WGS84) or KML/KMZ. CSV columns: name, lat, lon (decimal or DMS like 24°34'55"N),
        optional known_height_m. See points_template.csv.
        In the automatic modes the points file is optional: only to give known heights for sun calibration.
Mosques file (optional, mode 2)  your own mosque points or outlines. Blank = built-in list.
Limit to area (optional, modes 2-3)  any KML/SHP/CSV; only its extent (rectangle) is processed.

SUN ELEVATION (choose one, in this order of accuracy)
a) Known heights: 2 or more points with known_height_m (from a DSM / survey) -> calibrated from the image.
b) Sun elevation in degrees from the image metadata (SUN_ELEVATION / sunElevation).
c) Image date (+ time if known; time zone 3 for Saudi Arabia). Time is solved from the shadow direction.
MOSAICS: a mosaic made of several acquisitions has a different sun angle in each strip. Run each strip
separately with "Limit to area" (a KML/SHP of that strip) and its own sun elevation.

OPTIONS
Shadow direction       blank = detected from the image. If results look wrong, enter it (degrees, object -> shadow tip).
Search radius (m)      mode 1: how far from your point the shadow may start (25; use 10-15 if points are accurate)
Longest shadow (m)     ignore streaks longer than this (default 80)
Report heights above   default 15 m
Your data minus image  shift between your reference data and the image, metres north / east.
                       Riyadh ECW mosaic vs. our reference: north 23.9, east -13.5.

OUTPUT (output folder)
  shadow_heights.*          mode 1          minaret_scan_heights.*   mode 2          auto_scan_heights.*   mode 3
  .csv                      one row per result: name, status, height_m, low/high, feet, DMS + decimal coords,
                            shadow length/azimuth/width, sun elevation (mode 2 also the mosque location)
  _points.shp / _shadow_lines.shp   WGS84 shapefiles      .kml   Google Earth      _review.jpg   image chips

STATUS VALUES
  ok / likely - verify      measured; automatic results must still be confirmed on the review sheet
  possible - ...            weaker candidate (2nd shadow at a mosque, thin or faint shadow, very tall)
  check: ...                measured but unusual - look at the review sheet
  below minimum height / no thin shadow found / outside image

COMMAND LINE (optional)
  ShadowHeightBatch.exe --image clip.tif --points pts.csv --out results --sun-elev 61.9 --search 15 --offset-n 23.9 --offset-e -13.5
  ShadowHeightBatch.exe --mosques-auto --image clip.tif --out results --sun-elev 61.9 [--area strip.kml] [--mosques my_mosques.shp]
  ShadowHeightBatch.exe --scan --image clip.tif --out results --sun-elev 61.9 [--area area.kml]
  ShadowHeightBatch.exe --selftest test_folder     (checks the installation: must end with SELFTEST PASSED)

SPEED: mode 1 ~1 s per point; mode 2 ~2-3 s per mosque; mode 3 ~3 s per 0.25 km2.
ACCURACY: typically +/- 2-3 m for clear, unobstructed shadows.
