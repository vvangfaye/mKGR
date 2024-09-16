import shutil
from shapely.geometry import Polygon, LineString, Point, MultiPoint, MultiPolygon
import geopandas as gpd
import pandas as pd
import os
import math
import numpy as np
import time
from osgeo import gdal, ogr, osr

gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
gdal.SetConfigOption("SHAPE_ENCODING", "")
import random
label_chinese_map = {
    '101': "居民地",
    '201': "商业办公区",
    '202': "商业服务区",
    '301': "工业区",
    '402': "交通用地",
    '403': "机场",
    '501': "行政用地",
    '502': "教育用地",
    '503': "医院",
    '504': "体育文化",
    '505': "公园绿地",
    '800': "水体",
    '401': "道路",
}
chinese_euluc_map = {
    "居民地": '101',
    "商业办公区": '201',
    "商业服务区": '202',
    "工业区": '301',
    "交通用地": '402',
    "机场": '403',
    "行政用地": '501',
    "教育用地": '502',
    "医院": '503',
    "体育文化": '504',
    "公园绿地": '505',
    "水体": '800',
    "道路": '401'
}
chinese_label_map = {
    "居民地": 0,
    "商业办公区": 1,
    "商业服务区": 2,
    "工业区": 3,
    "交通用地": 4,
    "机场": 5,
    "行政用地": 6,
    "教育用地": 7,
    "医院": 8,
    "体育文化": 9,
    "公园绿地": 10,
}
cell2euluc = {
    "residential": "101",
    "Forest": "505",
    "Industrial": "301",
    "football": "504",
    "Park": "505",
    "college": "502",
    "government": "501",
    "Airport": "403",
    "meadow": "505",
    "golf": "504",
    "Stadium": "504",
    "resi1": "101",
    "Commercial": "202",
    "baseball": "504",
    "orchard": "505",
    "Port": "402",
    "shrub": "505",
    "Hospital": "503",
    "resi3": "101",
    "power_plant": "301",
    "StorageTanks": "301",
    "RailwayStation": "402",
    "School": "502",
    "tennis": "504",
    "substation": "301",
    "office": "201",
    "basketball": "504",
    "center": "504",
    "charge_station": "301"
}
def combine_shp(shp_path, save_path):
    shp_list = []
    for root, dirs, files in os.walk(shp_path):
        for file in files:
            if file.endswith('.shp') and file[0] != '.':
                shp_list.append(os.path.join(root, file))

    df = gpd.GeoDataFrame()  # 初始化一个空的GeoDataFrame
    for shp in shp_list:
        gdf = gpd.read_file(shp)
        if df.empty:
            df = gdf
        else:
            df = pd.concat([df, gdf], axis=0)
    dir = os.path.dirname(save_path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    # 使用map函数和label_chinese_map字典来创建新的euluc_cn字段
    # df['euluc_cn'] = df['euluc_osm'].map(label_chinese_map)
    # 保存合并后的shapefile文件
    df.to_file(save_path, encoding='utf-8')


def get_f1_acc(mid_data_path):
    log_list = []
    for root, dirs, files in os.walk(mid_data_path):
        for file in files:
            if 'log.txt' == file and 'new_normal' in root:
                log_list.append(os.path.join(root, file))
    F1_array = np.zeros((len(log_list), 1))
    ACC_array = np.zeros((len(log_list), 1))
    for log_path in log_list:
        with open(log_path, 'r') as f:
            lines = f.readlines()
            last_line = lines[-1]
            F1 = float(last_line.split(',')[-2].strip().split(':')[-1].strip())
            ACC = float(last_line.split(',')[-1].strip().split(':')[-1].strip())
            F1_array[log_list.index(log_path)] = F1
            ACC_array[log_list.index(log_path)] = ACC
    mean_F1 = np.mean(F1_array)
    mean_ACC = np.mean(ACC_array)
    print("file nember:{}, mean_F1:{}, mean_ACC:{}".format(len(log_list), mean_F1, mean_ACC))


def calculate_accuracies(confusion_matrix):
    # Calculate Producer's Accuracy
    diagonal_values = np.diag(confusion_matrix)
    row_sums = np.sum(confusion_matrix, axis=1)
    producers_accuracy = diagonal_values / row_sums
    mean_producers_accuracy = np.mean(producers_accuracy)

    # Calculate User's Accuracy
    col_sums = np.sum(confusion_matrix, axis=0)
    users_accuracy = diagonal_values / col_sums
    mean_users_accuracy = np.mean(users_accuracy)

    # Calculate Overall Accuracy
    overall_accuracy = np.sum(diagonal_values) / np.sum(confusion_matrix, axis=0).sum()

    return mean_producers_accuracy, mean_users_accuracy, overall_accuracy
    # get the precision and recall

def overlapEvaluate_geo(label_shp, predict_shp):
    # Load the data
    label_df = gpd.read_file(label_shp)
    predict_df = gpd.read_file(predict_shp)

    # trans the crs
    label_df = label_df.to_crs('EPSG:6933')
    predict_df = predict_df.to_crs('EPSG:6933')

    count = len(label_df)

    confusion_matrix = np.ones((11, 11)) * 1e-10

    ori_time = time.time()
    num = len(label_df)
    for i, row in label_df.iterrows():
        now_time = time.time()
        avg_time = (now_time - ori_time) / (i + 1)
        remain_time = avg_time * (count - i)
        if str(row["llama3-70b"]) == "0":
            continue

        geom = row.geometry
        # 检测geom是否valid
        if not geom.is_valid:
            continue
        try:
            precise_matches = predict_df[predict_df.geometry.intersects(geom)]
        except:
            continue

        for idx, match_row in precise_matches.iterrows():
            try:
                intersection = match_row.geometry.intersection(geom)
                if intersection.is_empty or isinstance(intersection, Polygon) and intersection.area <= 0.000000000000001:
                    continue

                chinese_label = label_chinese_map[str(row["landuse"])]
                if chinese_label == 'None' or chinese_label is None:
                    continue
                if chinese_label == '水体':
                    continue
                label = chinese_label_map[chinese_label]
                if match_row["euluc_osm"] not in label_chinese_map.keys():
                    continue
                code = label_chinese_map[match_row["euluc_osm"]]
                # if match_row["euluc"] not in label_chinese_map.keys():
                #     continue
                # code = label_chinese_map[match_row["euluc"]]
                
                if code == '水体' or code == '道路':
                    continue
                predict = chinese_label_map[code]

                area = intersection.area
                confusion_matrix[label, predict] += area
            except:
                continue

        if i % 500 == 0:
            print(f"caculate fid:{i}/{count}, remain_time:{remain_time}")
        if i % 10000 == 0:
            pa, ua, oa = calculate_accuracies(confusion_matrix)
            print(oa)

    return confusion_matrix


if __name__ == '__main__':

    matrix = overlapEvaluate_geo(
        '/media/dell/DATA/wy/data/nature_data/ori_data/labeled_result/labeled_result.shp',
        '/media/dell/DATA/wy/code/CUKG/Hubei/logs/VecS_4/predict_shp/all_result.shp')
    df = pd.DataFrame(matrix)
    pa, ua, oa = calculate_accuracies(matrix)
    print("All mean producers_accuracy:{}, mean users_accuracy:{}, overall_accuracy:{}".format(pa, ua, oa))