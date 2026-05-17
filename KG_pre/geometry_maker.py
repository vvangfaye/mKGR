"""Geometry generation stage.

Port of CUKG/preprocess/main_2023.py (authoritative) + the unique
``combine_shp_list`` of sum_road.py + the province ``sum_poi_by_pr`` of
main_2023_pr.py. ``get_block.py`` is a verbatim subset of main_2023.py and
is intentionally dropped; ``fliter_2018_osm.py`` is 2018-only and dropped.

Produces, under ``cfg.data_dir``: bound/, osm/, poi_sample_0.1/, block/,
unit/, cell/, area/ — exactly the shapefiles KG_construction consumes.

Multiprocessing note: ``mp.Pool`` cannot pickle bound methods, so the
per-feature workers stay module-level functions taking an args tuple; the
``GeometryMaker`` methods only assemble the args list and own the pool.
"""

import os
import time

import geopandas as gpd
import pandas as pd
import shapely
from shapely.strtree import STRtree

from config import PreConfig
from utils import ensure_dir, get_ratio, list_shp, osm2euluc, set_gdal_utf8

set_gdal_utf8()


# --------------------------------------------------------------------------- #
# Module-level pool workers (unchanged logic from main_2023.py)
# --------------------------------------------------------------------------- #
def _clip_poi_worker(args):
    bound_path, poi_dir, save_dir = args
    bound_gdf = gpd.read_file(bound_path, encoding="utf-8")
    polygon = bound_gdf["geometry"].unary_union

    ct_name = os.path.splitext(os.path.basename(bound_path))[0]
    out_path = os.path.join(save_dir, "{}.shp".format(ct_name))
    if os.path.exists(out_path):
        print("exist: {}".format(ct_name))
        return

    poi_path = os.path.join(poi_dir, "{}.shp".format(ct_name))
    if not os.path.exists(poi_path):
        print("not exist: {}".format(ct_name))
        return
    poi_gdf = gpd.read_file(poi_path, encoding="utf-8")
    clipped_poi = gpd.overlay(poi_gdf, gpd.GeoDataFrame(geometry=[polygon]))
    try:
        clipped_poi.to_file(out_path, driver="ESRI Shapefile", encoding="utf-8")
    except Exception:
        print("error: {}".format(ct_name))


def _sample_poi_worker(args):
    poi, source_dir, save_dir, sample_rate, threshold = args
    if not poi.endswith(".shp"):
        return
    out_path = os.path.join(save_dir, poi.split(".")[0] + ".shp")
    if os.path.exists(out_path):
        return
    gdf = gpd.read_file(os.path.join(source_dir, poi), encoding="utf-8")
    # >threshold 个 POI 则随机抽样 sample_rate (CUKG/utils.py random_sample_poi_worker)
    if len(gdf) > threshold:
        gdf = gdf.sample(frac=sample_rate)
    gdf["FID"] = range(len(gdf))
    gdf.reset_index(drop=True, inplace=True)
    gdf.to_file(out_path, encoding="utf-8")


def _block_worker(args):
    bound_dir, block_dir, bound, save_dir1, cfg = args
    dt_name = bound.split(".")[0]
    if os.path.exists(os.path.join(save_dir1, "{}.shp".format(dt_name))):
        return
    bound_path = os.path.join(bound_dir, bound)
    block_road_path = os.path.join(block_dir, bound)
    bound_gdf = gpd.read_file(bound_path, encoding="utf-8")
    if os.path.exists(block_road_path):
        block_road_gdf = gpd.read_file(block_road_path, encoding="utf-8").explode(index_parts=True)
    else:
        block_road_gdf = gpd.GeoDataFrame(columns=["geometry"])
    bound_gdf = bound_gdf.to_crs(cfg.CRS_GEOG)
    bound_lines = bound_gdf.boundary.explode(index_parts=True)
    lines = list(block_road_gdf["geometry"]) + list(bound_lines)
    merged_lines = shapely.ops.linemerge(lines)
    border_lines = shapely.ops.unary_union(merged_lines)
    decomposition = shapely.ops.polygonize_full(border_lines)
    area = gpd.GeoDataFrame(
        {"geometry": decomposition[0]}, crs=cfg.CRS_GEOG, index=[0]
    ).explode(index_parts=True)
    # 道路宽度缓冲
    area1 = area.to_crs(cfg.CRS_AREA).buffer(cfg.block_road_buffer_m)
    area1 = area1[-area1.is_empty]
    area1 = gpd.GeoDataFrame({"geometry": area1})
    if area1.empty:
        area1 = area
    area = area.explode(index_parts=True)
    # 过滤过小地块
    area1 = area1[area1.area > cfg.min_block_area]
    if area1.empty:
        area1 = area
    # 过滤长宽比过大的地块
    area1["ratio"] = area1.apply(
        lambda x: get_ratio(x.geometry.minimum_rotated_rectangle.boundary.coords), axis=1
    )
    area1 = area1[area1.ratio < cfg.max_block_ratio]
    if area1.empty:
        area1 = area
    area1 = area1.to_crs(cfg.CRS_GEOG)
    area1.drop(columns=["ratio"], inplace=True)
    area1 = gpd.overlay(area1, bound_gdf)
    area1["FID"] = range(len(area1))
    area1 = area1.reset_index(drop=True)
    area1.to_file(os.path.join(save_dir1, "{}.shp".format(dt_name)), encoding="utf-8")


def _unit_worker(args):
    bound_dir, block_dir, block, unit_dir, save_dir, cfg = args
    dt_name = block.split(".")[0]
    if os.path.exists(os.path.join(save_dir, "{}.shp".format(dt_name))):
        return
    bound_path = os.path.join(bound_dir, block)
    block_path = os.path.join(block_dir, block)
    unit_road_path = os.path.join(unit_dir, block)
    block_gdf = gpd.read_file(block_path, encoding="utf-8")
    if os.path.exists(unit_road_path):
        unit_road_gdf = gpd.read_file(unit_road_path, encoding="utf-8").explode(index_parts=True)
    else:
        return
    bound_gdf = gpd.read_file(bound_path, encoding="utf-8")
    block_gdf = block_gdf.to_crs(cfg.CRS_GEOG)
    bound_gdf = bound_gdf.to_crs(cfg.CRS_GEOG)
    block_lines = block_gdf.boundary.explode(index_parts=True)
    lines = list(unit_road_gdf["geometry"]) + list(block_lines)
    merged_lines = shapely.ops.linemerge(lines)
    border_lines = shapely.ops.unary_union(merged_lines)
    decomposition = shapely.ops.polygonize_full(border_lines)
    area = gpd.GeoDataFrame(
        {"geometry": decomposition[0]}, crs=cfg.CRS_GEOG, index=[0]
    ).explode(index_parts=True)
    area = gpd.overlay(area, block_gdf)
    area["FID"] = range(len(area))
    area = area.reset_index(drop=True)
    area.to_file(os.path.join(save_dir, "{}.shp".format(dt_name)), encoding="utf-8")


# Full cell layer, set in the parent before mp.Pool() so fork()ed workers
# inherit it copy-on-write (no re-read, no pickling). Used by clip_cell_rtree_mp.
_MP_CELL = None


def _cell_mp_worker(args):
    ct, idx, poly_wkb, save_dir = args
    out = os.path.join(save_dir, "{}.shp".format(ct))
    if os.path.exists(out):
        return (ct, -1)
    from shapely import wkb

    poly = wkb.loads(poly_wkb)
    sub = _MP_CELL.iloc[idx].copy()
    clipped = sub.geometry.intersection(poly)
    keep = ~(clipped.is_empty | clipped.isna())
    sub = sub.loc[keep].copy()
    sub["geometry"] = clipped.loc[keep]
    sub["ct_name"] = ct
    sub["FID"] = range(len(sub))
    sub = sub.reset_index(drop=True)
    sub.to_file(out, driver="ESRI Shapefile", encoding="utf-8")
    return (ct, len(sub))


# --------------------------------------------------------------------------- #
class GeometryMaker:
    """Boundary -> osm/poi/block/unit/cell/area shapefiles for the whole
    boundary file (split per ``ct_name``)."""

    def __init__(self, cfg: PreConfig):
        self.cfg = cfg

    # -- step 1: city/urban boundary from admin division + nightlight ----- #
    def get_city_boundary(self):
        cfg = self.cfg
        ensure_dir(os.path.dirname(cfg.bound_path))
        dis_gdf = gpd.read_file(cfg.admin_path, encoding=cfg.admin_encoding)
        nightlight_gdf = gpd.read_file(cfg.nightlight_path, encoding="utf-8")
        result_gdf = gpd.GeoDataFrame(columns=["ct_name", "geometry"])
        begin_time = time.time()
        for admin_index, (_, admin_row) in enumerate(dis_gdf.iterrows()):
            avg = (time.time() - begin_time) / (admin_index + 1)
            print("get_city_boundary: {}/{} eta {:.0f}s".format(
                admin_index + 1, len(dis_gdf), avg * (len(dis_gdf) - admin_index)))
            admin_polygon = admin_row["geometry"]
            clipped = gpd.overlay(nightlight_gdf, gpd.GeoDataFrame(geometry=[admin_polygon]))
            clipped["ct_name"] = admin_row[cfg.admin_name_field]
            result_gdf = pd.concat([result_gdf, clipped], ignore_index=True)
        result_gdf.to_file(cfg.bound_path, driver="ESRI Shapefile", encoding="utf-8")

    # -- step 2: split unified boundary per city -------------------------- #
    def split_bound(self):
        cfg = self.cfg
        ensure_dir(cfg.split_bound_dir)
        bound_gdf = gpd.read_file(cfg.bound_path, encoding="utf-8")
        for i, ct_name in enumerate(bound_gdf["ct_name"].unique()):
            print("split_bound: {}/{}".format(i, len(bound_gdf["ct_name"].unique())))
            row = bound_gdf[bound_gdf["ct_name"] == ct_name]
            row.to_file(os.path.join(cfg.split_bound_dir, "{}.shp".format(ct_name)),
                        driver="ESRI Shapefile", encoding="utf-8")

    # -- step 3: OSM code -> EULUC --------------------------------------- #
    def map_osm_to_euluc(self):
        cfg = self.cfg
        ensure_dir(os.path.dirname(cfg.osm_euluc_path))
        osm_gdf = gpd.read_file(cfg.osm_plus_path, encoding="utf-8")
        osm_gdf = osm_gdf[osm_gdf["code"].isin(osm2euluc.keys())]
        osm_gdf["euluc"] = osm_gdf["code"].apply(lambda x: osm2euluc[x])
        osm_gdf.drop(columns=["osm_id", "layer", "path"], inplace=True)
        osm_gdf.to_file(cfg.osm_euluc_path, driver="ESRI Shapefile", encoding="utf-8")

    # -- step 4a: clip OSM to boundaries --------------------------------- #
    def clip_osm(self):
        cfg = self.cfg
        ensure_dir(cfg.osm_dir)
        osm_gdf = gpd.read_file(cfg.osm_euluc_path, encoding="utf-8")
        bound_gdf = gpd.read_file(cfg.bound_path, encoding="utf-8")
        clipped_osm = gpd.overlay(osm_gdf, bound_gdf)
        clipped_osm["euluc_osm"] = clipped_osm["euluc"]
        for i, ct_name in enumerate(clipped_osm["ct_name"].unique()):
            out = os.path.join(cfg.osm_dir, "{}.shp".format(ct_name))
            if os.path.exists(out):
                continue
            print("clip_osm: {}".format(ct_name))
            row = clipped_osm[clipped_osm["ct_name"] == ct_name].copy()
            row["FID"] = range(len(row))
            row = row.reset_index(drop=True)
            row.to_file(out, driver="ESRI Shapefile", encoding="utf-8")

    # -- step 4b: clip POI (split per-city src, parallel) ----------------- #
    def clip_poi(self):
        cfg = self.cfg
        ensure_dir(cfg.poi_clip_dir)
        bound_paths = [os.path.join(cfg.split_bound_dir, f)
                       for f in list_shp(cfg.split_bound_dir)]
        args_list = [(bp, cfg.poi_src_dir, cfg.poi_clip_dir) for bp in bound_paths]
        with self._pool() as pool:
            pool.map(_clip_poi_worker, args_list)

    # -- step 4b': random-sample clipped POI -> poi_sample_0.1 ------------ #
    def sample_poi(self):
        """clip 后的 poi/ -> 抽样 (>threshold 则 frac=rate) + FID -> poi_sample_0.1/
        (port of CUKG/utils.py random_sample_poi_parallel)."""
        cfg = self.cfg
        ensure_dir(cfg.poi_dir)
        args_list = [(p, cfg.poi_clip_dir, cfg.poi_dir,
                      cfg.poi_sample_rate, cfg.poi_sample_threshold)
                     for p in list_shp(cfg.poi_clip_dir)]
        with self._pool() as pool:
            pool.map(_sample_poi_worker, args_list)

    # -- step 5.1: merge roads + railways + waterways --------------------- #
    def combine_roads_water(self):
        cfg = self.cfg
        ensure_dir(os.path.dirname(cfg.road_all_path))
        road = gpd.read_file(cfg.road_osm_path, encoding="utf-8")[["fclass", "name", "geometry"]]
        rail = gpd.read_file(cfg.railway_path, encoding="utf-8")[["fclass", "name", "geometry"]]
        rail["fclass"] = "railway"
        water = gpd.read_file(cfg.waterway_path, encoding="utf-8")[["fclass", "name", "geometry"]]
        water["fclass"] = "water"
        out = pd.concat([road, rail, water], ignore_index=True)
        out.to_file(cfg.road_all_path, driver="ESRI Shapefile", encoding="utf-8")

    # sum_road.combine_shp_list: merge the supplemental road blocks
    def combine_extra_roads(self):
        cfg = self.cfg
        ensure_dir(os.path.dirname(cfg.road_extra_merged_path))
        src = cfg.road_extra_src_dir
        shp_list = [os.path.join(src, f) for f in os.listdir(src)
                    if f.endswith(".shp") and not f.startswith("._")]
        gdf = pd.concat([gpd.read_file(s, encoding="utf-8") for s in shp_list],
                        ignore_index=True)
        gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")
        gdf.to_file(cfg.road_extra_merged_path, driver="ESRI Shapefile", encoding="utf-8")

    # -- step 5.2: clip road network into block/unit roads ---------------- #
    def clip_roads(self):
        cfg = self.cfg
        ensure_dir(cfg.block_road_dir)
        ensure_dir(cfg.unit_road_dir)
        bound_gdf = gpd.read_file(cfg.bound_path, encoding="utf-8")
        road_gdf = gpd.read_file(cfg.road_all_path, encoding="utf-8")
        bound_gdf = bound_gdf.to_crs(road_gdf.crs)
        clipped = gpd.overlay(road_gdf, bound_gdf)
        block_fclass = ["motorway", "motorway_link", "trunk", "trunk_link",
                        "primary", "primary_link", "secondary", "secondary_link",
                        "tertiary", "tertiary_link"]
        for ct_name in clipped["ct_name"].unique():
            try:
                row = clipped[clipped["ct_name"] == ct_name]
                row[row["fclass"].isin(block_fclass)].to_file(
                    os.path.join(cfg.block_road_dir, "{}.shp".format(ct_name)),
                    driver="ESRI Shapefile", encoding="utf-8")
                row[row["fclass"].isin(["residential", "unclassified"])].to_file(
                    os.path.join(cfg.unit_road_dir, "{}.shp".format(ct_name)),
                    driver="ESRI Shapefile", encoding="utf-8")
            except Exception:
                print("clip_roads error: {}".format(ct_name))

        # supplemental unit roads + merge into unit_all_road
        if os.path.exists(cfg.road_extra_path):
            self._clip_roads_plus()
            self._combine_all_road()

    def _clip_roads_plus(self):
        cfg = self.cfg
        ensure_dir(cfg.unit_plus_road_dir)
        bound_gdf = gpd.read_file(cfg.bound_path, encoding="utf-8")
        road_gdf = gpd.read_file(cfg.road_extra_path, encoding="utf-8")
        bound_gdf = bound_gdf.to_crs(road_gdf.crs)
        clipped = gpd.overlay(road_gdf, bound_gdf)
        for ct_name in clipped["ct_name"].unique():
            try:
                clipped[clipped["ct_name"] == ct_name].to_file(
                    os.path.join(cfg.unit_plus_road_dir, "{}.shp".format(ct_name)),
                    driver="ESRI Shapefile", encoding="utf-8")
            except Exception:
                print("clip_roads_plus error: {}".format(ct_name))

    def _combine_all_road(self):
        cfg = self.cfg
        ensure_dir(cfg.unit_all_road_dir)
        for f in list_shp(cfg.unit_plus_road_dir):
            if "台湾" in f:
                continue
            unit_road = os.path.join(cfg.unit_road_dir, f)
            if not os.path.exists(unit_road):
                continue
            a = gpd.read_file(unit_road, encoding="utf-8")
            b = gpd.read_file(os.path.join(cfg.unit_plus_road_dir, f), encoding="utf-8")
            pd.concat([a, b], ignore_index=True).to_file(
                os.path.join(cfg.unit_all_road_dir, f),
                driver="ESRI Shapefile", encoding="utf-8")

    # -- step 5.3: polygonize block roads -> block ------------------------ #
    def make_block(self):
        cfg = self.cfg
        ensure_dir(cfg.block_dir)
        args_list = [
            (cfg.split_bound_dir, cfg.block_road_dir, bound, cfg.block_dir, cfg)
            for bound in list_shp(cfg.split_bound_dir)
        ]
        with self._pool() as pool:
            pool.map(_block_worker, args_list)

    # -- step 5.4: cut block by unit roads -> unit ------------------------ #
    def make_unit(self):
        cfg = self.cfg
        ensure_dir(cfg.unit_dir)
        unit_road = (cfg.unit_all_road_dir
                     if os.path.isdir(cfg.unit_all_road_dir)
                     and list_shp(cfg.unit_all_road_dir) else cfg.unit_road_dir)
        args_list = [
            (cfg.split_bound_dir, cfg.block_dir, block, unit_road, cfg.unit_dir, cfg)
            for block in list_shp(cfg.block_dir)
        ]
        with self._pool() as pool:
            pool.map(_unit_worker, args_list)

    # -- step 4c / 5: clip cell + area ----------------------------------- #
    def clip_cell(self):
        self._clip_overlay(self.cfg.cell_src_path, self.cfg.cell_dir, set_crs=True)

    def clip_area(self):
        self._clip_overlay(self.cfg.area_src_path, self.cfg.area_dir, set_crs=False)

    def _clip_overlay(self, src_path, save_dir, set_crs):
        cfg = self.cfg
        ensure_dir(save_dir)
        bound_gdf = gpd.read_file(cfg.bound_path, encoding="utf-8")
        src_gdf = gpd.read_file(src_path, encoding="utf-8")
        if set_crs:
            src_gdf.set_crs(cfg.CRS_GEOG, inplace=True, allow_override=True)
        clipped = gpd.overlay(src_gdf, bound_gdf)
        for ct_name in clipped["ct_name"].unique():
            out = os.path.join(save_dir, "{}.shp".format(ct_name))
            if os.path.exists(out):
                continue
            print("clip -> {}".format(ct_name))
            row = clipped[clipped["ct_name"] == ct_name].copy()
            row["FID"] = range(len(row))
            row = row.reset_index(drop=True)
            row.to_file(out, driver="ESRI Shapefile", encoding="utf-8")

    def clip_cell_rtree_mp(self, num_processes=8, src_path=None, save_dir=None):
        """Multiprocess variant of :meth:`clip_cell_rtree`.

        Reads the (huge) cell layer **once** in the parent and builds the
        STRtree there; ``STRtree.query`` (cheap) is run single-threaded in the
        parent, and only the expensive per-city ``.intersection()`` + write is
        parallelised. Workers inherit the cell layer via Linux ``fork``
        copy-on-write (no re-read, no pickling of the 13M-feature frame).
        Resumable: an already-written city is skipped."""
        import multiprocessing as mp

        from shapely import wkb

        cfg = self.cfg
        src_path = src_path or cfg.cell_src_path
        save_dir = save_dir or cfg.cell_dir
        ensure_dir(save_dir)

        bound = gpd.read_file(cfg.bound_path, encoding="utf-8")
        if bound.crs is None:
            bound.set_crs(cfg.CRS_GEOG, inplace=True)
        bound = bound.to_crs(cfg.CRS_GEOG).dissolve(by="ct_name")

        global _MP_CELL
        _MP_CELL = gpd.read_file(src_path, encoding="utf-8")
        _MP_CELL = _MP_CELL.set_crs(cfg.CRS_GEOG, allow_override=True)
        tree = STRtree(_MP_CELL.geometry.values)

        tasks = []
        for ct in bound.index:
            if os.path.exists(os.path.join(save_dir, "{}.shp".format(ct))):
                continue
            poly = bound.geometry.loc[ct]
            idx = tree.query(poly, predicate="intersects")
            if len(idx) == 0:
                continue
            tasks.append((ct, idx, wkb.dumps(poly), save_dir))

        print("clip_cell_rtree_mp: {} cities to do, {} procs".format(
            len(tasks), num_processes), flush=True)
        with mp.Pool(num_processes) as pool:        # fork → COW-inherit _MP_CELL
            for i, (ct, n) in enumerate(
                    pool.imap_unordered(_cell_mp_worker, tasks)):
                print("clip_cell_rtree_mp {}/{} {} -> {}".format(
                    i + 1, len(tasks), ct, n), flush=True)

    def clip_cell_rtree(self, src_path=None, save_dir=None):
        """Spatial-index per-city cell clip (port of the original
        ``clip_cell_by_boundary_rtree``, fixed for shapely>=2.0).

        Scales to the national 13M-feature cell layer where ``gpd.overlay``
        does not: builds one STRtree, then per ``ct_name`` queries candidates
        and intersects. Writes each city as it goes — incremental progress and
        resumable (skips an already-written city)."""
        cfg = self.cfg
        src_path = src_path or cfg.cell_src_path
        save_dir = save_dir or cfg.cell_dir
        ensure_dir(save_dir)

        bound = gpd.read_file(cfg.bound_path, encoding="utf-8")
        if bound.crs is None:
            bound.set_crs(cfg.CRS_GEOG, inplace=True)
        bound = bound.to_crs(cfg.CRS_GEOG)
        bound_by_ct = bound.dissolve(by="ct_name")          # 366 unioned polys

        cell = gpd.read_file(src_path, encoding="utf-8")
        cell = cell.set_crs(cfg.CRS_GEOG, allow_override=True)
        tree = STRtree(cell.geometry.values)

        ct_names = list(bound_by_ct.index)
        for i, ct in enumerate(ct_names):
            out = os.path.join(save_dir, "{}.shp".format(ct))
            if os.path.exists(out):
                continue
            poly = bound_by_ct.geometry.loc[ct]
            idx = tree.query(poly, predicate="intersects")
            if len(idx) == 0:
                print("clip_cell_rtree {}/{} {} -> 0 (skip)".format(
                    i + 1, len(ct_names), ct), flush=True)
                continue
            sub = cell.iloc[idx].copy()
            clipped = sub.geometry.intersection(poly)
            keep = ~(clipped.is_empty | clipped.isna())
            sub = sub.loc[keep].copy()
            sub["geometry"] = clipped.loc[keep]
            sub["ct_name"] = ct
            sub["FID"] = range(len(sub))
            sub = sub.reset_index(drop=True)
            sub.to_file(out, driver="ESRI Shapefile", encoding="utf-8")
            print("clip_cell_rtree {}/{} {} -> {}".format(
                i + 1, len(ct_names), ct, len(sub)), flush=True)

    # -- province: aggregate city shapefiles into province shapefiles ----- #
    def aggregate_by_province(self):
        """main_2023_pr.sum_poi_by_pr, driven by city_province.json."""
        import json

        cfg = self.cfg
        cp = json.load(open(os.path.join(os.path.dirname(__file__),
                                          "city_province.json"), encoding="utf-8"))
        for province, city_list in cp.items():
            # 7 类 = KG_construction 的逐城输入: 同省城市 shp 纵向合并 + 重排 FID
            for sub in ("unit", "block", "area", "cell",
                        "poi_sample_0.1", "osm", "seed"):
                src_dir = os.path.join(cfg.data_dir, sub)
                dst_dir = ensure_dir(os.path.join(cfg.data_dir + "_pr", sub))
                if os.path.exists(os.path.join(dst_dir, province + ".shp")):
                    continue
                merged = gpd.GeoDataFrame()
                for city in city_list:
                    cp_path = os.path.join(src_dir, city + ".shp")
                    if not os.path.exists(cp_path):
                        continue
                    merged = pd.concat(
                        [merged, gpd.read_file(cp_path, encoding="utf-8")], axis=0)
                if merged.empty:
                    continue
                merged["FID"] = range(len(merged))
                merged.reset_index(drop=True, inplace=True)
                merged.to_file(os.path.join(dst_dir, province + ".shp"),
                               encoding="utf-8")

    # ------------------------------------------------------------------- #
    def _pool(self):
        import multiprocessing as mp
        return mp.Pool(self.cfg.num_processes)

    def run(self):
        """Full geometry stage; sub-steps toggle via cfg.do_* (replicates
        the original comment-out-to-skip workflow)."""
        cfg = self.cfg
        if cfg.do_boundary:
            self.get_city_boundary()
        if cfg.do_split_bound:
            self.split_bound()
        if cfg.do_map_osm:
            self.map_osm_to_euluc()
        if cfg.do_clip_osm:
            self.clip_osm()
        if cfg.do_clip_poi:
            self.clip_poi()
        if cfg.do_sample_poi:
            self.sample_poi()
        if cfg.do_combine_roads:
            self.combine_roads_water()
        if cfg.do_clip_roads:
            self.clip_roads()
        if cfg.do_make_block:
            self.make_block()
        if cfg.do_make_unit:
            self.make_unit()
        if cfg.do_clip_cell:
            self.clip_cell()
        if cfg.do_clip_area:
            self.clip_area()
        if cfg.level == "province":
            self.aggregate_by_province()
