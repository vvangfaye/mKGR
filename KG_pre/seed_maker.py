"""Seed-label generation stage.

Collapses CUKG/preprocess/generate_seed/{3..11} (Chinese-named numbered
scripts) into ordered methods, preserving the
chip -> sjoin -> merge -> map -> dedup -> group -> ratio sequence.
``only_use_cell.py`` becomes the alternative ``seed_from_cell_only``,
selected by ``cfg.seed_mode``.

Final output: ``cfg.seed_dir/{city}.shp`` (the seed shapefiles that
KG_construction's ``build_seed_triples`` consumes).

Per-step pool workers are module-level functions (mp.Pool can't pickle
bound methods); each script's ``__main__`` becomes a public ``stepN``.
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from config import PreConfig
from utils import (amap_class2euluc, cell_class2euluc, ensure_dir,
                   euluc_code2chinese, list_csv, list_shp, set_gdal_utf8)

set_gdal_utf8()


# --------------------------------------------------------------------------- #
# step 3: unit/area/cell/osm overlay -> chip polygons
# --------------------------------------------------------------------------- #
def _overlay_polygonize(pns, out_path, crs):
    lines = []
    for pn in pns:
        lines.extend(list(pn.boundary.explode(index_parts=True)))
    merged_lines = shapely.ops.linemerge(lines)
    border_lines = shapely.ops.unary_union(merged_lines)
    decomposition = shapely.ops.polygonize(border_lines)
    area = gpd.GeoDataFrame({"geometry": list(decomposition)}, crs=crs)
    area.to_file(out_path)


def _overlay_worker(args):
    cfg, city = args
    out_path = os.path.join(cfg.chip_dir, city)
    if os.path.exists(out_path):
        return
    pns = []
    for item in ("unit", "area", "cell", "osm"):
        path = os.path.join(cfg.data_dir, item, city)
        gdf = gpd.read_file(path, engine=cfg.io_engine, encoding="utf-8")
        if item == "cell":
            gdf = gdf.dissolve(by="cls").explode(index_parts=True)
        if gdf.crs != cfg.CRS_GEOG:
            gdf = gdf.to_crs(cfg.CRS_GEOG)
        pns.append(gdf)
    _overlay_polygonize(pns, out_path, cfg.CRS_GEOG)


# --------------------------------------------------------------------------- #
# step 4: spatial-join chip representative point with each entity layer
# --------------------------------------------------------------------------- #
def _sjoin_worker(args):
    cfg, leftdir, rightdir, outdir, file = args
    out_shp = os.path.join(outdir, file)
    if os.path.exists(out_shp.replace(".shp", ".csv")):
        return
    rightshp = os.path.join(rightdir, file)
    leftshp = os.path.join(leftdir, file)
    if not os.path.exists(rightshp):
        return
    right = gpd.read_file(rightshp, engine=cfg.io_engine, encoding="utf-8")
    right["right_area"] = right.to_crs(cfg.CRS_AREA).area
    if "FID" not in right.columns:
        right.reset_index(inplace=True)
        right.rename(columns={"index": "FID"}, inplace=True)
    if right.crs != cfg.CRS_GEOG:
        right = right.to_crs(cfg.CRS_GEOG)
    left = gpd.read_file(leftshp, engine=cfg.io_engine, encoding="utf-8")
    left["left_area"] = left.to_crs(cfg.CRS_AREA).area
    left["rp"] = left.representative_point()
    left = left.set_geometry("rp")
    sjoin = left.sjoin(right)
    sjoin = sjoin.drop(columns=["rp", "index_right"])
    sjoin = sjoin.set_geometry("geometry")
    len_fid_right = len(sjoin["FID_right"].unique())
    if len(right) != len_fid_right:
        print("{} {} != {}".format(file, len(right), len_fid_right))
    sjoin.to_file(out_shp, engine=cfg.io_engine, encoding="utf-8")
    sjoin.drop("geometry", axis=1).to_csv(out_shp.replace(".shp", ".csv"))


# --------------------------------------------------------------------------- #
# step 5: merge the four chip-{item} csvs, add `from` column
# --------------------------------------------------------------------------- #
def _merge_worker(args):
    cfg, city = args
    out_path = os.path.join(cfg.chip4_dir, city)
    if os.path.exists(out_path):
        return
    dfs = []
    for item in ("unit", "area", "cell", "osm"):
        path = os.path.join(cfg.data_dir, "chip-{}".format(item), city)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df = df.drop(df.columns[0], axis=1)
        dfs.append(df)
    df = pd.concat(dfs)
    df.loc[df["block"].notnull(), "from"] = "area"
    df.loc[df["cls"].notnull(), "from"] = "cell"
    df.loc[df["code"].notnull(), "from"] = "osm"
    df.loc[df["from"].isnull(), "from"] = "unit"
    df.to_csv(out_path)


# --------------------------------------------------------------------------- #
# step 6: map cell class -> euluc_cell + consensus euluc (mode)
# --------------------------------------------------------------------------- #
def _map_cell_worker(args):
    cfg, city = args
    out_path = os.path.join(cfg.chip4_cell_dir, city)
    if os.path.exists(out_path):
        return
    df = pd.read_csv(os.path.join(cfg.chip4_dir, city))
    df["euluc_cell"] = df["cls"].map(cell_class2euluc)
    df["euluc"] = df[["euluc_osm", "euluc_area", "euluc_cell"]].mode(axis=1)[0]
    df.to_csv(out_path)


# --------------------------------------------------------------------------- #
# step 7: dedup keeping the largest-area row per (FID_left, from)
# --------------------------------------------------------------------------- #
def _dedup_worker(args):
    cfg, city = args
    out_path = os.path.join(cfg.chip4_dedup_dir, city)
    if os.path.exists(out_path):
        return
    df = pd.read_csv(os.path.join(cfg.chip4_dir, city))
    df = df.drop(df.columns[0], axis=1)
    df = df.sort_values("right_area", ascending=False)
    df.drop_duplicates(subset=["FID_left", "from"], keep="first", inplace=True)
    df.to_csv(out_path)


# --------------------------------------------------------------------------- #
# step 7+1: map amap (area) class -> euluc_area
# --------------------------------------------------------------------------- #
def _map_area_worker(args):
    cfg, city = args
    out_path = os.path.join(cfg.chip4_dedup_area_dir, city)
    if os.path.exists(out_path):
        return
    df = pd.read_csv(os.path.join(cfg.chip4_dedup_dir, city))
    df["euluc_area"] = df["type"].map(amap_class2euluc)
    df.to_csv(out_path)


# --------------------------------------------------------------------------- #
# step 8: group by FID_left, resolve euluc (osm > area, 201 override), to shp
# --------------------------------------------------------------------------- #
def _agg_func2(x):
    return pd.Series({
        "euluc_area": "".join(x["euluc_area"]),
        "euluc_osm": "".join(x["euluc_osm"]),
        "cls_new": "".join(x["cls"]),
    })


def _dedup_group_worker(args):
    cfg, city = args
    out_csv = os.path.join(cfg.chip4_dedup_group_dir, city)
    if os.path.exists(out_csv):
        return
    df = pd.read_csv(os.path.join(cfg.chip4_dedup_area_dir, city))
    df = df.drop(df.columns[0], axis=1)
    df["euluc_area"] = df["euluc_area"].astype("Int64").astype(str).replace("<NA>", "")
    df["euluc_osm"] = df["euluc_osm"].astype("Int64").astype(str).replace("<NA>", "")
    df["cls"] = df["cls"].astype(str).replace("nan", "")
    df = df.groupby("FID_left").apply(_agg_func2)
    df["euluc_osm"] = df["euluc_osm"].replace("", np.nan)
    df["euluc_area"] = df["euluc_area"].replace("", np.nan)
    df["euluc"] = df["euluc_osm"]
    df["euluc"] = df["euluc"].fillna(df["euluc_area"])
    df.loc[df["euluc_area"] == "201", "euluc"] = "201"
    df.to_csv(out_csv)
    df = df.reset_index()
    df = df[["FID_left", "euluc"]]
    gdf = gpd.read_file(os.path.join(cfg.chip_dir, city.replace(".csv", ".shp")),
                        engine=cfg.io_engine, encoding="utf-8")
    gdf = gdf.merge(df, left_on="FID", right_on="FID_left")
    gdf.to_file(os.path.join(cfg.chip4_dedup_group_shp_dir, city.replace(".csv", ".shp")),
                engine=cfg.io_engine, encoding="utf-8")


# --------------------------------------------------------------------------- #
# step 9: merge chip-level labels with chip-unit attributes
# --------------------------------------------------------------------------- #
def _unit_dedup_group_worker(args):
    cfg, city = args
    out_path = os.path.join(cfg.chip_unit_dedup_group_dir, city)
    if os.path.exists(out_path):
        return
    df = pd.read_csv(os.path.join(cfg.chip4_dedup_group_dir, city))
    dfunit = pd.read_csv(os.path.join(cfg.chip_unit_dir, city))
    dfunit = dfunit.drop(dfunit.columns[0], axis=1)
    df = dfunit.merge(df, on="FID_left")
    df.to_csv(out_path)


# --------------------------------------------------------------------------- #
# step 10: per-unit occupation ratio by label class
# --------------------------------------------------------------------------- #
def _unit_ratio_worker(args):
    cfg, city = args
    out_csv = os.path.join(cfg.chip_unit_ratio_dir, city)
    if os.path.exists(out_csv):
        return
    df = pd.read_csv(os.path.join(cfg.chip_unit_dedup_group_dir, city))
    df = df.drop(df.columns[0], axis=1)
    df["left_ratio"] = df["left_area"] / df["right_area"]
    df = df.groupby(["FID_right", "euluc"])["left_ratio"].sum().reset_index()
    df = df.sort_values("left_ratio", ascending=False)
    df.drop_duplicates(subset=["FID_right"], keep="first", inplace=True)
    df.to_csv(out_csv)
    gdf = gpd.read_file(os.path.join(cfg.unit_dir, city.replace(".csv", ".shp")),
                        engine=cfg.io_engine, encoding="utf-8")
    gdf = gdf.merge(df, left_on="FID", right_on="FID_right", how="left")
    gdf.to_file(os.path.join(cfg.chip_unit_ratio_shp_dir, city.replace(".csv", ".shp")),
                engine=cfg.io_engine, encoding="utf-8")


# --------------------------------------------------------------------------- #
# step 11: keep top `unit_ratio_keep` fraction -> final seed shapefile
# --------------------------------------------------------------------------- #
def _unit_ratio_pct_worker(args):
    cfg, city = args
    out_shp = os.path.join(cfg.seed_dir, city.replace(".csv", ".shp"))
    if os.path.exists(out_shp):
        return
    df = pd.read_csv(os.path.join(cfg.chip_unit_ratio_dir, city))
    df = df.drop(df.columns[0], axis=1)
    end = int(len(df) * cfg.unit_ratio_keep)
    df = df.iloc[:end, :]
    gdf = gpd.read_file(os.path.join(cfg.unit_dir, city.replace(".csv", ".shp")),
                        engine=cfg.io_engine, encoding="utf-8")
    gdf = gdf.merge(df, left_on="FID", right_on="FID_right", how="left")
    gdf.to_file(out_shp, engine=cfg.io_engine, encoding="utf-8")


# --------------------------------------------------------------------------- #
# only_use_cell.py alternative
# --------------------------------------------------------------------------- #
def _cell_only_worker(args):
    cfg, city = args
    out_shp = os.path.join(cfg.seed_dir, "{}.shp".format(city))
    if os.path.exists(out_shp):
        return
    gdf1 = gpd.read_file(os.path.join(cfg.unit_dir, "{}.shp".format(city)), encoding="utf-8")
    gdf2 = gpd.read_file(os.path.join(cfg.cell_dir, "{}.shp".format(city)), encoding="utf-8")
    if "cls" not in gdf2.columns:
        raise ValueError("shp2 必须包含名为 'cls' 的字段。")
    result_gdf = gdf1.copy()
    result_gdf["max_cls"] = None
    result_gdf["euluc_osm"] = None
    result_gdf["euluc_cn"] = None
    result_gdf["max_area"] = 0
    for idx, row in gdf1.iterrows():
        if not row.geometry.is_valid:
            continue
        inter = gdf2[gdf2.intersects(row.geometry)]
        if not inter.empty:
            inter = inter.copy()
            inter["area"] = inter.intersection(row.geometry).area
            top = inter.sort_values("area", ascending=False).iloc[0]
            result_gdf.at[idx, "max_cls"] = top["cls"]
            result_gdf.at[idx, "max_area"] = top["area"]
            if top["cls"] in cell_class2euluc:
                euluc = cell_class2euluc[top["cls"]]
                result_gdf.at[idx, "euluc_osm"] = euluc
                result_gdf.at[idx, "euluc_cn"] = euluc_code2chinese[euluc]
    result_gdf.drop(columns=["max_area"], inplace=True)
    result_gdf.to_file(out_shp, encoding="utf-8")


# --------------------------------------------------------------------------- #
class SeedMaker:
    def __init__(self, cfg: PreConfig):
        self.cfg = cfg

    def _pool(self):
        import multiprocessing as mp
        return mp.Pool(self.cfg.num_processes)

    def _run_cities(self, out_dir, worker, src_dir=None, ext=".shp"):
        cfg = self.cfg
        ensure_dir(out_dir)
        src = src_dir or cfg.unit_dir
        names = list_shp(src) if ext == ".shp" else list_csv(src)
        args_list = [(cfg, n) for n in names]
        with self._pool() as pool:
            pool.map(worker, args_list)

    # ordered steps (run.bat order: 3->4->5->6->7->7+1->8->9->10->11) ----- #
    def overlay_chip(self):
        self._run_cities(self.cfg.chip_dir, _overlay_worker,
                         src_dir=self.cfg.unit_dir, ext=".shp")

    def sjoin_chip(self):
        cfg = self.cfg
        names = list_shp(cfg.chip_dir)
        for item, outdir in (("osm", cfg.chip_osm_dir), ("unit", cfg.chip_unit_dir),
                             ("area", cfg.chip_area_dir), ("cell", cfg.chip_cell_dir)):
            ensure_dir(outdir)
            rightdir = os.path.join(cfg.data_dir, item)
            args_list = [(cfg, cfg.chip_dir, rightdir, outdir, n) for n in names]
            with self._pool() as pool:
                pool.map(_sjoin_worker, args_list)

    def merge_chip4(self):
        self._run_cities(self.cfg.chip4_dir, _merge_worker,
                         src_dir=self.cfg.chip_unit_dir, ext=".csv")

    def map_cell_class(self):
        self._run_cities(self.cfg.chip4_cell_dir, _map_cell_worker,
                         src_dir=self.cfg.chip4_dir, ext=".csv")

    def pick_osm_max_area(self):
        self._run_cities(self.cfg.chip4_dedup_dir, _dedup_worker,
                         src_dir=self.cfg.chip4_dir, ext=".csv")

    def map_area_class(self):
        self._run_cities(self.cfg.chip4_dedup_area_dir, _map_area_worker,
                         src_dir=self.cfg.chip4_dedup_dir, ext=".csv")

    def dedup_group_shp(self):
        ensure_dir(self.cfg.chip4_dedup_group_shp_dir)
        self._run_cities(self.cfg.chip4_dedup_group_dir, _dedup_group_worker,
                         src_dir=self.cfg.chip4_dedup_area_dir, ext=".csv")

    def unit_dedup_group(self):
        self._run_cities(self.cfg.chip_unit_dedup_group_dir, _unit_dedup_group_worker,
                         src_dir=self.cfg.chip4_dedup_group_dir, ext=".csv")

    def unit_ratio(self):
        ensure_dir(self.cfg.chip_unit_ratio_shp_dir)
        self._run_cities(self.cfg.chip_unit_ratio_dir, _unit_ratio_worker,
                         src_dir=self.cfg.chip_unit_dedup_group_dir, ext=".csv")

    def unit_ratio_pct(self):
        self._run_cities(self.cfg.seed_dir, _unit_ratio_pct_worker,
                         src_dir=self.cfg.chip_unit_ratio_dir, ext=".csv")

    def seed_from_cell_only(self):
        cfg = self.cfg
        ensure_dir(cfg.seed_dir)
        from utils import city_stems
        args_list = [(cfg, c) for c in city_stems(cfg.unit_dir)]
        with self._pool() as pool:
            pool.map(_cell_only_worker, args_list)

    def run(self):
        if self.cfg.seed_mode == "cell_only":
            self.seed_from_cell_only()
            return
        self.overlay_chip()
        self.sjoin_chip()
        self.merge_chip4()
        self.map_cell_class()
        self.pick_osm_max_area()
        self.map_area_class()
        self.dedup_group_shp()
        self.unit_dedup_group()
        self.unit_ratio()
        self.unit_ratio_pct()
