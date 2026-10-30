"""
TouchTerrain multicolor runner (TouchTerrain_for_MPH).

Usage:
    python TouchTerrain_multicolor.py job.json

The job file uses the normal TouchTerrain settings (DEM_name, trlat/trlon/
bllat/bllon or importedDEM, tilewidth, zscale, basethick, zip_file_name, ...)
plus the multicolor settings listed in touchterrain/common/multicolor.py.

Output: <output_folder>/<zip_file_name>.zip holding one STL per layer, named
<name>_<layer>_T<toolhead>.stl, plus a summary json and a log.
"""

import glob
import json
import os
import sys
import zipfile

from touchterrain.common import multicolor


def get_dem_from_earth_engine(job, work_dir, log):
    """Ask TouchTerrain for the projected elevation GeoTIFF of the area."""
    from touchterrain.common import TouchTerrainEarthEngine as TouchTerrain
    args = {k: job[k] for k in ("DEM_name", "trlat", "trlon", "bllat", "bllon",
                                "polygon", "poly_file", "polyURL", "projection")
            if job.get(k) is not None}
    args.update(fileformat="GeoTiff", temp_folder=work_dir,
                zip_file_name=job.get("zip_file_name", "terrain") + "_dem",
                printres=multicolor.settings_with_defaults(job)["feature_res_mm"],
                tilewidth=job.get("tilewidth", 120), ntilesx=1, ntilesy=1)
    _, zip_path = TouchTerrain.get_zipped_tiles(**args)
    with zipfile.ZipFile(zip_path) as z:
        tifs = [n for n in z.namelist() if n.lower().endswith(".tif")]
        if not tifs:
            raise RuntimeError("Earth Engine export returned no GeoTIFF")
        z.extract(tifs[0], work_dir)
    log("elevation from Earth Engine:", job["DEM_name"])
    return os.path.join(work_dir, tifs[0])


def main(job_path):
    with open(job_path) as f:
        job = json.load(f)
    out_root = job.get("output_folder", "/output" if os.path.isdir("/output") else "output")
    name = job.get("zip_file_name", "terrain")
    out_dir = os.path.join(out_root, name)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, name + "_multicolor.log")
    logf = open(log_path, "w")

    def log(*a):
        line = " ".join(str(x) for x in a)
        print(line)
        logf.write(line + "\n")
        logf.flush()

    if job.get("importedDEM"):
        dem = job["importedDEM"]
        log("elevation from local file:", dem)
    else:
        dem = get_dem_from_earth_engine(job, out_dir, log)

    summary = multicolor.make_multicolor(dem, job, out_dir, log)
    logf.close()

    zip_path = os.path.join(out_root, name + ".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for fn in list(summary["files"].values()) + [name + "_multicolor_summary.json",
                                                    name + "_multicolor.log"]:
            z.write(os.path.join(out_dir, fn), fn)
    print("\nCreated", zip_path)
    print("Scale", summary["scale"], "| model", summary["model_mm"], "mm")
    for layer, fn in summary["files"].items():
        print(f"  {layer:8s} -> {fn}")
    return zip_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
