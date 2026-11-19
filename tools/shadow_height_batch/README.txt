SHADOW HEIGHT BATCH 3.3  -  heights of minarets, towers, poles and pylons from satellite shadows
================================================================================================

WHAT IT DOES
Finds the thin dark shadow of a tall object, measures its length and converts it to height:
height = shadow length x tan(sun elevation).
Outputs: CSV, shapefiles (points + shadow lines), KML, a review sheet (JPG) and a clickable REVIEW PAGE (HTML).
NEW in 3.0: a learning model. You mark results Yes/No on the review page, press Train, and the program learns
what real minarets / towers / pylons look like in YOUR imagery. It gets better every time you train.

START
Double-click ShadowHeightBatch.exe (no installation, no Python needed).
Windows may show "Windows protected your PC" the first time: click More info -> Run anyway.

FOUR MODES ("What to do")
1. Measure heights at my points             - most accurate. You give a point list (on/next to each object).
2. AUTOMATIC: find minarets around mosques  - no points needed. Built-in list of ~21,000 mosques (Saudi Arabia + Gulf,
                                              OpenStreetMap) or your own mosques file (points or outlines: SHP/KML/CSV).
                                              Up to 3 candidate shadows per mosque, ranked by the learned model.
3. AUTOMATIC: scan whole image / area       - every thin tall-object shadow; best for pylons, masts, poles in open ground.
4. TRAIN                                    - learn from your Yes/No labels (see below).

THE LEARNING LOOP (do this for the first few images, then the automatic results improve)
 a) Run mode 2 or 3. In the output folder open  <name>_review.html  in Chrome/Edge.
 b) Click the first card, then press  Y  (real minaret / tower / pylon / pole)  or  N  (wall, gap between buildings,
    tree row, car, kerb). U = unsure. The page moves to the next card automatically.
    Left picture: north up, magenta tick = shadow start (object base), cyan tick = shadow tip.
    Right picture: the same shadow turned to point right, magenta brackets = the measured shadow.
 c) Press "Save labels". Put the downloaded  <name>_labels.csv  into the same output folder.
    (Alternative: type Y or N in the 'confirm' column of the CSV, or the CONFIRM field of the points shapefile.)
 d) Mode 4 TRAIN: add one or more such folders, Run. The model (shadow_model.npz) is saved next to the program
    and used automatically from then on. Every training adds to the earlier examples.
 e) Share the improvement: copy shadow_model.npz (and shadow_training_data.npz) next to colleagues' ShadowHeightBatch.exe.
The log after training shows the quality (AUC: 0.5 = guessing, 1.0 = perfect) and how much of the list you need to
look at to find half / 85% of the real objects.

SECOND IMAGE CHECK (NEW in 3.1/3.2) - the strongest filter
A real minaret / tower casts a shadow from the SAME base in a second image of another date, and the height measured there
must agree. Walls, gaps and kerbs fail this test. Give a second view in "Second image check":
   - ArcGIS MapServer (cached tiles), e.g.  https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer
     (acquisition dates are read from the service, each acquisition is calibrated separately)
   - XYZ / WMTS tile URL with {z}/{x}/{y}  (or {TileMatrix}/{TileCol}/{TileRow}), Google-Maps tiling
   - WMS URL (GetMap, EPSG:4326)
   - or an image file (GeoTIFF/JP2) of another date
Small windows (about 120 m) are fetched around each candidate only. The program lines the two images up, measures the shadow
again and works out the second image's sun: from its acquisition date (ArcGIS World Imagery tells the date of every patch)
by finding the time of day that fits the shadows, or from the candidates, or from the values you enter.
Status then ends with "2nd image agrees" (shadow and height agree) or "2nd image: no matching shadow / different height".
Results are ordered: likely + agrees, likely, possible + agrees, possible, unlikely.
Riyadh test (K763X53392 clip, 567 candidates, World Imagery WV03 5 Sep 2025 ~11:00 / GE01 8 Dec 2025 ~10:00 / WV02 23 Sep 2025):
   independent sample checked in World Imagery: "likely + agrees" about 7 in 10 are real minarets,
   "possible + agrees" about 1-3 in 10, "unlikely + agrees" none.
   Not agreeing does NOT mean wrong: in the second image the shadow may fall on a building, or a car may hide the base.
Use services under your organisation's licence (Esri World Imagery: ArcGIS account / terms of use); add a token if required.
Internet access is needed for URLs.

STARTER MODEL - WHAT TO EXPECT
The built-in model learned from your 30 checked results (21 real, 9 false) plus 786 rejected shadows in the Riyadh
K763X53392 clip. Test result (cross-validated): AUC 0.74 (0.5 = guessing). It still needs your labels to get better.
In dense blocks, narrow gaps between buildings still look like minaret shadows, so ALWAYS check the review page.
About 200-300 labelled examples from your own imagery should improve it clearly.
Shadows that fall into a dark courtyard or onto another building's shadow cannot be seen - those minarets are missed.

WORK FROM A SERVICE URL ONLY - NO IMAGE FILE (NEW in 3.3)
Paste the imagery service into the Image box instead of a file:
   ArcGIS MapServer   https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer
   WMS                https://yourserver/geoserver/wms?LAYERS=riyadh_2025   (put LAYERS= in the URL)
   ArcGIS WMS         https://yourserver/arcgis/services/Imagery/Riyadh/MapServer/WMSServer?LAYERS=0
   WMTS / XYZ         https://yourserver/.../{TileMatrix}/{TileRow}/{TileCol}.jpg   or   .../{z}/{x}/{y}.jpg
   (a Map Viewer link "...mapviewer/index.html?url=<service>" also works - the service URL is taken out of it)
and choose "Limit to area" (KML/KMZ/SHP/CSV, or an image file whose extent is the area) and an Output folder.
The program downloads only that area (up to 25 km2 per run) at 0.3 m into service_image.tif in the output folder and
extracts from it. Sun: World Imagery tells the acquisition date of the area - leave the sun options as they are
(or choose "Image date" and leave it empty) and the program uses that date and the shadow direction it measures
(about +-10% on heights). If the service gives no date (plain WMS), enter the sun elevation or the date.
If the area mixes several acquisitions the log warns you: use smaller areas, one acquisition each.
Use services under your organisation's licence; add a token in the second-image token box if required.

COORDINATES
Results are in the coordinates of the image itself: pins sit on the shadows when you overlay the shapefile/KML on the
same image. Leave "Your data minus image" at 0 unless your reference layer is known to be shifted from the image.

INPUTS
Image   GeoTIFF (.tif) or JPEG 2000 (.jp2), georeferenced, 0.3-0.6 m pixels work best.
        ECW is not supported - export it first (ArcGIS Pro: Export Raster -> TIFF).
Points  CSV/TXT, shapefile (WGS84) or KML/KMZ. CSV columns: name, lat, lon (decimal or DMS like 24°34'55"N),
        optional known_height_m. See points_template.csv.
Mosques file (optional, mode 2)  your own mosque points or outlines. Blank = built-in list.
Limit to area (optional, modes 2-3)  any KML/SHP/CSV; only its extent (rectangle) is processed.

SUN ELEVATION (choose one, in this order of accuracy)
a) Known heights: 2 or more points with known_height_m -> calibrated from the image.
b) Sun elevation in degrees from the image metadata (SUN_ELEVATION / sunElevation).
c) Image date (+ time if known; time zone 3 for Saudi Arabia).
SHADOW DIRECTION: enter it if you can (= sun azimuth from the metadata + 180). Finds more, fewer false hits.
MOSAICS: each acquisition strip has its own sun angle - run each strip separately with "Limit to area".

STATUS VALUES
  ok                         measured (mode 1)
  likely / possible / unlikely (model)   learned-model ranking (automatic modes) - confirm on the review page
  unlikely - longer than a minaret       very long streak (> 60 m height): usually a wall or kerb shadow
  check: ...                 measured but unusual - look at the review sheet
  below minimum height / no thin shadow found / outside image

OUTPUT (output folder)
  shadow_heights.* (mode 1), minaret_scan_heights.* (mode 2), auto_scan_heights.* (mode 3):
  .csv (with model_score and an empty 'confirm' column), _points.shp, _shadow_lines.shp, .kml, _review.jpg,
  _review.html (labelling page), _chips.npz (image chips the training uses - keep them with the labels)

COMMAND LINE (optional)
  ShadowHeightBatch.exe --image clip.tif --points pts.csv --out results --sun-elev 61.9 --search 15
  ShadowHeightBatch.exe --mosques-auto --image clip.tif --out results --sun-elev 61.9 --shadow-az 312 [--area strip.kml]
  ShadowHeightBatch.exe --scan --image clip.tif --out results --sun-elev 61.9 [--area area.kml]
  ShadowHeightBatch.exe --mosques-auto --image clip.tif --out results --sun-elev 61.9 --second https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer
  ShadowHeightBatch.exe --train results1 results2            (learn from labelled folders)
  ShadowHeightBatch.exe --selftest test_folder               (installation check: must end with SELFTEST PASSED)

SPEED: mode 1 ~1 s per point; mode 2 ~3-6 s per mosque; mode 3 ~3 s per 0.25 km2; training a few seconds.
ACCURACY of heights: typically +/- 2-3 m for clear, unobstructed shadows.
