"""Test / label generation stage.

Port of CUKG/preprocess/make_test_dataset.py + get_google_earth_id.py.

Produces the label files KG_construction consumes:
  - ``label/{city}.shp``                (combine_unit_label)
  - ``label/2023_chinese_label.shp``    (combine_shp over final_label/)
  - ``label/new_label/标注/{city}.shp``  (generate_label / get_final_label)

``fishgrid_pred_to_shp`` depends on KG_embedding prediction output and is a
*post*-construction utility — exposed here but NOT chained into the default
geometry->seed flow.
"""

import json
import os
import time

import geopandas as gpd
import numpy as np
import pandas as pd
from osgeo import ogr, osr

from config import PreConfig
from utils import (chinese_label2index, ensure_dir, euluc_code2chinese,
                   set_gdal_utf8)

set_gdal_utf8()


class TestMaker:
    def __init__(self, cfg: PreConfig):
        self.cfg = cfg

    # -- unit x ground-truth label spatial join -------------------------- #
    def combine_unit_label(self, unit_path, label_path, save_path):
        unit_ds = ogr.Open(unit_path)
        label_ds = ogr.Open(label_path)
        unit_lyr = unit_ds.GetLayer()
        label_lyr = label_ds.GetLayer()

        driver = ogr.GetDriverByName("ESRI Shapefile")
        combine_ds = driver.CreateDataSource(save_path)
        combine_lyr = combine_ds.CreateLayer(
            "combine", unit_lyr.GetSpatialRef(),
            geom_type=ogr.wkbPolygon, options=["ENCODING=UTF-8"])
        for fld in unit_lyr.schema:
            combine_lyr.CreateField(fld)
        combine_lyr.CreateField(ogr.FieldDefn("label", ogr.OFTString))

        ori_time = time.time()
        for i, feature in enumerate(unit_lyr):
            if feature.GetField("euluc") is not None:
                continue
            if i % 100 == 0:
                avg = (time.time() - ori_time) / (i + 1)
                print("combine_unit_label {}/{} eta {:.0f}s".format(
                    i + 1, unit_lyr.GetFeatureCount(),
                    (unit_lyr.GetFeatureCount() - i - 1) * avg))
            label_lyr.SetSpatialFilter(None)
            unit_geom = feature.GetGeometryRef()
            label_lyr.SetSpatialFilter(unit_geom)
            precise_matches = [
                f for f in label_lyr
                if f.GetGeometryRef().Intersection(unit_geom) is not None
                and f.GetGeometryRef().Intersection(unit_geom).Area() > 1e-15
            ]
            if not precise_matches:
                continue
            max_area = 0
            max_match = None
            for pm in precise_matches:
                if pm.GetGeometryRef().Area() > max_area:
                    max_area = pm.GetGeometryRef().Intersection(unit_geom).Area()
                    max_match = pm
            unit_area = unit_geom.Area()
            label = max_match.GetField("euluc_cn")
            if label in ("公园绿地", "教育用地", "居民地"):
                if max_area / unit_area < 0.3 or unit_area < 1e-6:
                    continue
            combine_feature = ogr.Feature(combine_lyr.GetLayerDefn())
            combine_feature.SetGeometry(unit_geom)
            for fld in unit_lyr.schema:
                combine_feature.SetField(fld.name, feature.GetField(fld.name))
            combine_feature.SetField("label", label)
            combine_lyr.CreateFeature(combine_feature)
            combine_feature = None

    def statistic_label(self, label_path):
        label_ds = ogr.Open(label_path)
        label_lyr = label_ds.GetLayer()
        label_array = np.zeros(len(chinese_label2index))
        label_area_array = np.zeros(len(chinese_label2index))
        source_srs = osr.SpatialReference()
        source_srs.ImportFromEPSG(self.cfg.CRS_GEOG)
        target_srs = osr.SpatialReference()
        target_srs.ImportFromEPSG(self.cfg.CRS_AREA)
        transform = osr.CoordinateTransformation(source_srs, target_srs)
        for feature in label_lyr:
            label = feature.GetField("label")
            if label not in chinese_label2index:
                continue
            label_array[chinese_label2index[label]] += 1
            geom = feature.GetGeometryRef()
            geom.Transform(transform)
            label_area_array[chinese_label2index[label]] += geom.Area() * 1000000
        return label_array, label_area_array

    def fliter_label(self, seed_path, label_path):
        save_label_path = label_path.split(".")[0] + "_fliter.shp"
        seed_ds = ogr.Open(seed_path)
        label_ds = ogr.Open(label_path)
        seed_lyr = seed_ds.GetLayer()
        label_lyr = label_ds.GetLayer()
        driver = ogr.GetDriverByName("ESRI Shapefile")
        fliter_ds = driver.CreateDataSource(save_label_path, options=["ENCODING=UTF-8"])
        fliter_lyr = fliter_ds.CreateLayer(
            "fliter", label_lyr.GetSpatialRef(), geom_type=ogr.wkbPolygon)
        for fld in label_lyr.schema:
            fliter_lyr.CreateField(fld)
        for feature in label_lyr:
            fid = feature.GetField("FID")
            seed_lyr.SetAttributeFilter("FID = {}".format(fid))
            seed_feature = seed_lyr.GetNextFeature()
            if seed_feature.GetField("euluc") is None:
                fliter_lyr.CreateFeature(feature)
        return save_label_path

    # -- KG_embedding predictions -> label shapefiles -------------------- #
    def generate_label(self, predict_path, save_path):
        gdf_predict = gpd.read_file(predict_path, encoding="utf-8")
        gdf_filtered = gdf_predict[
            (gdf_predict["source"] == "predict") & (gdf_predict["correct"] == "1")]
        gdf_filtered = gdf_filtered[["FID", "euluc_osm", "euluc_cn", "geometry"]]
        gdf_filtered.to_file(save_path, driver="ESRI Shapefile", encoding="utf-8")

    def get_final_label(self, data_path, ori_unit_path, save_shp_path):
        unit_gpd = gpd.read_file(ori_unit_path, encoding="utf-8")
        label_gpd = unit_gpd[["FID", "geometry"]].copy()
        label_gpd["euluc_osm"] = None
        label_gpd["euluc_cn"] = None
        entity2idx = {}
        with open(os.path.join(data_path, "entity2id.txt"), "r") as lines:
            for line in lines:
                entity, idx = line.strip().split("\t")
                entity2idx[entity] = int(idx)
        idx2entity = {v: k for k, v in entity2idx.items()}
        with open(os.path.join(data_path, "test"), "r") as lines:
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                entity, _, label = line.split("\t")
                fid = idx2entity[int(entity)]
                label = idx2entity[int(label)]
                fid = int(fid.split("/")[2])
                label = label.split("/")[1]
                label_gpd.loc[label_gpd["FID"] == fid, "euluc_osm"] = label
                label_gpd.loc[label_gpd["FID"] == fid, "euluc_cn"] = euluc_code2chinese[label]
        label_gpd = label_gpd[label_gpd["euluc_cn"].notna()]
        label_gpd.to_file(save_shp_path, driver="ESRI Shapefile", encoding="utf-8")

    def combine_shp(self, shp_path, save_path):
        shp_list = []
        for root, _dirs, files in os.walk(shp_path):
            for file in files:
                if file.endswith(".shp") and file[0] != ".":
                    shp_list.append(os.path.join(root, file))
        df = gpd.GeoDataFrame()
        for shp in shp_list:
            gdf = gpd.read_file(shp, encoding="utf-8")
            df = gdf if df.empty else pd.concat([df, gdf], axis=0)
        ensure_dir(os.path.dirname(save_path))
        df.to_file(save_path, encoding="utf-8")

    def build_city_province_json(self, save_json=None):
        cfg = self.cfg
        save_json = save_json or os.path.join(
            os.path.dirname(__file__), "city_province.json")
        shp_gdf = gpd.read_file(cfg.admin_div_path, encoding="GBK")
        city_province_map = {}
        for city in shp_gdf["行政区划_c"].unique():
            province = shp_gdf.loc[shp_gdf["行政区划_c"] == city, "FIRST_行政"].unique()[0]
            city_province_map.setdefault(province, []).append(city)
        json.dump(city_province_map, open(save_json, "w"), indent=4)

    # -- top-level: build the test labels consumed by KG_construction ---- #
    def build_labels(self, city_list=None):
        """combine_unit_label for each city against 2023_chinese_label.shp,
        producing ``label/{city}.shp``."""
        cfg = self.cfg
        ensure_dir(cfg.label_dir)
        if not os.path.exists(cfg.chinese_label_path):
            print("skip build_labels: {} not found".format(cfg.chinese_label_path))
            return
        from utils import city_stems
        cities = city_list or city_stems(cfg.unit_dir)
        for city in cities:
            unit_path = os.path.join(cfg.unit_dir, city + ".shp")
            save_path = os.path.join(cfg.label_dir, city + ".shp")
            if not os.path.exists(unit_path) or os.path.exists(save_path):
                continue
            print("combine_unit_label {}...".format(city))
            self.combine_unit_label(unit_path, cfg.chinese_label_path, save_path)

    # -- post-embedding: prediction txt + json -> scored shapefile ------- #
    @staticmethod
    def xyz2lnglatrange(xyz):
        x, y, z = xyz
        size = 360 / 2 ** z
        return (-180 + x * size, -180 + (x + 1) * size,
                180 - (y + 1) * size, 180 - y * size)

    def fishgrid_pred_to_shp(self, outfile, infile, cls_name, score):
        outdriver = ogr.GetDriverByName("ESRI Shapefile")
        if os.path.exists(outfile):
            outdriver.DeleteDataSource(outfile)
        outds = outdriver.CreateDataSource(outfile)
        outlayer = outds.CreateLayer(outfile, geom_type=ogr.wkbPolygon)
        f1 = ogr.FieldDefn("cls", ogr.OFTString)
        f1.SetWidth(12)
        outlayer.CreateField(f1)
        f2 = ogr.FieldDefn("score", ogr.OFTReal)
        f2.SetPrecision(3)
        outlayer.CreateField(f2)
        outfielddefn = outlayer.GetLayerDefn()
        lines = open(infile).readlines()
        for i, line in enumerate(lines):
            valid_str = line.split(" ")[0]
            xmin = int(valid_str.split("_")[2]) / 10000.0
            ymin = int(valid_str.split("_")[3]) / 10000.0
            xmax = int(valid_str.split("_")[4]) / 10000.0
            ymax = int(valid_str.split("_")[5][:-4]) / 10000.0
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(xmin, ymax)
            ring.AddPoint(xmax, ymax)
            ring.AddPoint(xmax, ymin)
            ring.AddPoint(xmin, ymin)
            ring.CloseRings()
            poly = ogr.Geometry(ogr.wkbPolygon)
            poly.AddGeometry(ring)
            outfeat = ogr.Feature(outfielddefn)
            outfeat.SetGeometry(poly)
            outfeat.SetField("cls", cls_name[i])
            outfeat.SetField("score", score[i])
            outlayer.CreateFeature(outfeat)
        outds = None
