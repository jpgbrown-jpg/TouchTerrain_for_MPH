# Multicolor export (TouchTerrain_for_MPH)

Exports the terrain plus **roads, water and towns as separate, aligned STL parts**
so a multi-toolhead printer (built for the Flashforge Creator 5 + Flash Studio)
can print each layer in its own color.

## How it works
1. Elevation comes from Earth Engine (same sources as TouchTerrain) or a local GeoTIFF (`importedDEM`).
2. Roads, water and town areas come from OpenStreetMap (Overpass API), or a local GeoJSON (`features_file`).
3. Every road and thin river is widened to a **minimum printed width** in mm, whatever the map scale.
4. Each layer sits in a pocket cut into the terrain (`inlay_depth_mm`), so parts touch but never overlap.
5. One closed STL per layer is written, named `<name>_<layer>_T<toolhead>.stl`.

## Run
```
docker compose run --rm touchterrain python TouchTerrain_multicolor.py /jobs/<job>.json
```
Offline check (no Earth Engine or internet): `jobs/offline-sample.json`.
Real test area: `jobs/stansbury-test.json`.

## Settings (job file)
| Setting | Default | Meaning |
|---|---|---|
| layers | roads, water, towns | layers to build |
| road_min_width_mm | 0.8 | narrowest a road prints |
| water_min_width_mm | 1.0 | narrowest a river/stream prints |
| min_feature_size_mm | 2.0 | lakes/towns smaller than this across are dropped |
| road_classes | motorway..tertiary | OSM road types included |
| inlay_depth_mm | 0.6 | depth of each layer's pocket |
| road_height_mm / water_height_mm / town_height_mm | 0.4 / 0 / 0 | surface above terrain |
| feature_res_mm | 0.25 | grid size on the print (capped at 1/3 of the smallest min width) |
| feature_priority | roads, water, towns | first wins where layers overlap |
| toolheads | terrain 1, roads 2, water 3, towns 4 | used in file names (3MF later) |
| features_file | none | local GeoJSON instead of OpenStreetMap |

Note: to guarantee the minimum, narrow features come out about one grid cell wider
than the minimum on average (0.8 mm setting -> about 0.9-1.2 mm printed at 0.25 mm grid).

## In Flash Studio
Import all layer STLs together and load them as one object with multiple parts,
then set each part's filament to the toolhead in its file name.

## Tests
`python test/multicolor/test_multicolor.py` - closed meshes, alignment, minimum widths, OSM parsing.
