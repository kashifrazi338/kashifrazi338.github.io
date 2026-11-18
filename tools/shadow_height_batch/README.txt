SHADOW HEIGHT BATCH  -  height of minarets, towers, poles and pylons from satellite shadows
=========================================================================================

WHAT IT DOES
For every point in your list it finds the thin dark shadow next to the point in the image,
measures its length and converts it to a height:  height = shadow length x tan(sun elevation).
Results: CSV, shapefile (points + shadow lines), KML for Google Earth, and a review sheet (JPG)
with a small image chip of every point so you can check each shadow by eye.

START
Double-click ShadowHeightBatch.exe (no installation, no Python needed).
Windows may show "Windows protected your PC" the first time: click More info -> Run anyway.
First start takes 10-20 seconds (the program unpacks itself).

INPUTS
1. Image: GeoTIFF (.tif) or JPEG 2000 (.jp2), georeferenced, 0.3-0.6 m pixels work best.
   ECW is not supported - export it first (ArcGIS Pro: Export Raster -> TIFF, or Global Mapper / QGIS).
2. Points: CSV/TXT, shapefile (WGS84) or KML/KMZ. Put each point ON or right next to the object base.
   CSV columns: name, lat, lon  (decimal degrees or DMS like 24°34'55"N)   optional: known_height_m
   See points_template.csv.
3. Sun (choose one, in this order of accuracy):
   a) Known heights - give 2 or more points a known_height_m (e.g. checked in a DSM / survey).
      The tool calibrates the sun elevation from them. Best option.
   b) Sun elevation in degrees, from the image metadata (SUN_ELEVATION / sunElevation).
   c) Image date (+ time if known, time zone 3 for Saudi Arabia). Time is solved from the shadow direction.
4. Options
   Search radius (m)      how far from your point the shadow may start (default 25; use 10-15 if points are accurate)
   Longest shadow (m)     ignore streaks longer than this (default 80)
   Report heights above   heights below this get status "below minimum height" (default 15 m)
   Your data minus image  if your reference data is shifted from the image, enter the shift in metres
                          (north, east). For the Riyadh ECW mosaic vs. our reference: north 23.9, east -13.5.
                          The tool shifts your points onto the image and the results back to your frame.

OUTPUT (in the output folder)
   shadow_heights.csv            one row per point: height_m, low/high, feet, DMS + decimal coords,
                                 shadow length/azimuth, sun elevation, status
   shadow_heights_points.shp     points with heights (WGS84)
   shadow_heights_shadow_lines.shp measured shadow (base -> tip) for checking in ArcGIS
   shadow_heights.kml            open in Google Earth
   shadow_heights_review.jpg     image chip per point with the measured shadow drawn

STATUS VALUES
   ok                            measured
   check: ...                    measured but unusual - look at the review sheet before using
   below minimum height          measured, lower than "Report heights above"
   no shadow found               nothing thin and dark in the search radius - move the point or increase radius

COMMAND LINE (optional, for very large batches)
   ShadowHeightBatch.exe --image clip.tif --points pts.csv --out results --sun-elev 61.9 --search 15 --offset-n 23.9 --offset-e -13.5
   ShadowHeightBatch.exe --selftest test_folder      (checks the installation: must print SELFTEST PASSED)

ACCURACY
   Typically +/- 2-3 m for clear, unobstructed shadows. Shadows falling on walls/roofs, merging with
   building shadows, or cut by trees are flagged or missed - always look at the review sheet.
