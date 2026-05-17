"""Shared helpers + JSON loaders for KG_pre.

Replaces CUKG/preprocess/utils.py: the inline ``osm2euluc`` dict and the
embedded class-mapping dicts (seed scripts 6 / 7+1, only_use_cell.py) now
live in ``map_data/*.json`` and are loaded here, mirroring how
``KG_construction`` loads its ``map_data``.
"""

import json
import os

from osgeo import gdal

_MAP_DIR = os.path.join(os.path.dirname(__file__), "map_data")


def load_map(name: str) -> dict:
    """Load a JSON mapping file from ``KG_pre/map_data/``."""
    with open(os.path.join(_MAP_DIR, name), encoding="utf-8") as f:
        return json.load(f)


# OSM code -> EULUC code. Keys are str in JSON; the OSM `code` column is
# numeric, so expose an int-keyed view as well (parity with the old dict).
_osm2euluc_str = load_map("OSM_Class_Similar_to_EULUC_Class.json")
osm2euluc = {int(k): int(v) for k, v in _osm2euluc_str.items()}

cell_class2euluc = load_map("Cell_Class_Similar_to_EULUC_Class.json")
amap_class2euluc = load_map("Amap_Class_Similar_to_EULUC_Class.json")
euluc_code2chinese = load_map("EULUC_Code_to_Chinese.json")
chinese_label2index = load_map("Chinese_Label_to_Index.json")


def set_gdal_utf8() -> None:
    """Force UTF-8 shapefile encoding (was duplicated at module top of
    main_2023.py lines 11-12)."""
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
    gdal.SetConfigOption("SHAPE_ENCODING", "UTF-8")


def get_ratio(coords) -> float:
    """Aspect ratio of the minimum rotated rectangle (main_2023.py:258)."""
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]
    width = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    height = ((x2 - x3) ** 2 + (y2 - y3) ** 2) ** 0.5
    return width / height if width > height else height / width


def ensure_dir(path: str) -> str:
    """``os.makedirs(path, exist_ok=True)`` and return ``path``."""
    os.makedirs(path, exist_ok=True)
    return path


def list_shp(directory: str):
    """Sorted .shp filenames in *directory*, skipping macOS dotfiles."""
    if not os.path.isdir(directory):
        return []
    return sorted(
        f for f in os.listdir(directory)
        if f.endswith(".shp") and not f.startswith(".")
    )


def list_csv(directory: str):
    """Sorted .csv filenames in *directory*."""
    if not os.path.isdir(directory):
        return []
    return sorted(f for f in os.listdir(directory) if f.endswith(".csv"))


def city_stems(directory: str):
    """City names (filename without .shp) found in *directory*."""
    return [os.path.splitext(f)[0] for f in list_shp(directory)]
