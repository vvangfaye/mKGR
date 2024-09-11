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
DN_chinese_map = {
    '1': "居民地",
    '2': "商业办公区",
    '3': "商业服务区",
    '4': "工业区",
    '6': "交通用地",
    '7': "机场",
    '8': "行政用地",
    '9': "教育用地",
    '10': "医院",
    '11': "体育文化",
    '12': "公园绿地",
    '13': "水体",
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
cell2euluc_v2 = {
    "Adminstrative": 501,
    "Airport": 403,
    "Bussiness_Office": 201,
    "Commercial": 202,
    "Education": 502,
    "Industrial": 301,
    "Medical": 503,
    "Park_Green": 505,
    "Residential": 101,
    "Sport_Cultural": 504,
    "Transportation": 402,
}
def combine_shp(shp_path, save_path):
    shp_list = []
    for root, dirs, files in os.walk(shp_path):
        for file in files:
            if file.endswith('.shp'):
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
    df['euluc_cn'] = df['euluc_osm'].map(label_chinese_map)
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

def overlapEvaluate(label_shp, predict_shp):
    # Open the data sources
    label_ds = ogr.Open(label_shp, 1)
    predict_ds = ogr.Open(predict_shp, 1)

    # Get the first layer
    label_layer = label_ds.GetLayer()
    predict_layer = predict_ds.GetLayer()

    count = label_layer.GetFeatureCount()
    
    # 创建空间索引
    label_sql = 'CREATE SPATIAL INDEX ON ' + label_layer.GetName()
    label_ds.ExecuteSQL(label_sql)
    predict_sql = 'CREATE SPATIAL INDEX ON ' + predict_layer.GetName()
    predict_ds.ExecuteSQL(predict_sql)

    ori_time = time.time()
    # confusion_matrix = np.ones((11, 11)) * 1e-10
    confusion_matrix = np.zeros((11, 11))
    # 保存预测对的地块到shp文件
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(os.path.join(os.path.dirname(predict_shp), 'predict_valid.shp')):
        driver.DeleteDataSource(os.path.join(os.path.dirname(predict_shp), 'predict_valid.shp'))
    predict_right_ds = driver.CreateDataSource(os.path.join(os.path.dirname(predict_shp), 'predict_valid.shp'))
    # 设置坐标系
    srs = predict_layer.GetSpatialRef()
    predict_right_layer = predict_right_ds.CreateLayer("predict_valid", srs, ogr.wkbPolygon)
    
    field_count = predict_layer.GetLayerDefn().GetFieldCount()
    
    for i in range(field_count):
        field_defn = predict_layer.GetLayerDefn().GetFieldDefn(i)
        predict_right_layer.CreateField(field_defn)
    
    
    for i, feature in enumerate(label_layer):
        if feature.GetField("DN") is None:
            continue
        now_time = time.time()
        avg_time = (now_time - ori_time) / (i+1)
        remain_time = avg_time * (count - i)
        
        predict_layer.SetSpatialFilter(None)
        
        geom = feature.GetGeometryRef()
        # Find intersecting features using spatial index
        predict_layer.SetSpatialFilter(geom)
        fid = feature.GetField("FID")
        precise_matches = [f for f in predict_layer if f.GetGeometryRef().Intersection(geom) is not None and f.GetGeometryRef().Intersection(geom).Area() > 0.000000000000001]
        
        # caculate the confusion matrix
        for precise_match in precise_matches:
            chinese_label = feature.GetField("euluc_cn")
            if chinese_label == 'None' or chinese_label is None:
                continue
            if chinese_label == '水体':
                continue
            label = chinese_label_map[chinese_label]
            
            code = label_chinese_map[precise_match.GetField("euluc_osm")]
            if code == '水体' or code == '道路':
                continue
            predict = chinese_label_map[code]
            
            if label == predict:
                predict_right_feature = ogr.Feature(predict_right_layer.GetLayerDefn())
                predict_right_feature.SetGeometry(precise_match.GetGeometryRef().Intersection(geom))
                for j in range(field_count):
                    predict_right_feature.SetField(predict_right_layer.GetLayerDefn().GetFieldDefn(j).GetNameRef(), precise_match.GetField(j))
                    if predict_right_layer.GetLayerDefn().GetFieldDefn(j).GetNameRef() == 'euluc_osm':
                        predict_right_feature.SetField(j, chinese_euluc_map[chinese_label])
                predict_right_layer.CreateFeature(predict_right_feature)
                predict_right_feature.Destroy()
            
            area = precise_match.GetGeometryRef().Intersection(geom).Area()
            confusion_matrix[label, predict] += area
        if i % 500 == 0:
            print("caculate fid:{}/{}, remain_time:{}".format(i, count, remain_time))
    return confusion_matrix
     
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
    overall_accuracy = np.sum(diagonal_values) / np.sum(confusion_matrix)

    return mean_producers_accuracy, mean_users_accuracy, overall_accuracy
    # get the precision and recall

def translate_DN_field(input_shp, output_shp):
    # 读取输入SHP文件
    gdf = gpd.read_file(input_shp, encoding='utf-8')

    # 先将euluc_osm字段的值转换为整数，然后再转换为字符串
    gdf['DN'] = gdf['DN'].astype(str)
    
    # 使用map函数和label_chinese_map字典来创建新的euluc_cn字段
    gdf['euluc_cn'] = gdf['DN'].map(DN_chinese_map)

    # 保存到输出SHP文件
    gdf.to_file(output_shp, encoding='utf-8')     
        
def save_to_excel(confusion_matrix, producers_accuracy, users_accuracy, overall_accuracy, file_name):
    # Create a DataFrame from the confusion matrix
    df_matrix = pd.DataFrame(confusion_matrix)
    
    # Create a DataFrame for accuracies
    df_producers = pd.DataFrame([producers_accuracy], columns=["Producer's Accuracy"])
    df_users = pd.DataFrame([users_accuracy], columns=["User's Accuracy"])
    df_overall = pd.DataFrame([overall_accuracy], columns=["Overall Accuracy"])

    with pd.ExcelWriter(file_name) as writer:
        df_matrix.to_excel(writer, sheet_name='confusion_matrix')
        df_producers.to_excel(writer, sheet_name='accuracy', startcol=0, startrow=0)
        df_users.to_excel(writer, sheet_name='accuracy', startcol=0, startrow=2)
        df_overall.to_excel(writer, sheet_name='accuracy', startcol=0, startrow=4)


def overlapEvaluate_geo(label_shp, predict_shp):
    # Load the data
    label_df = gpd.read_file(label_shp, encoding='utf-8')
    predict_df = gpd.read_file(predict_shp, encoding='utf-8')

    # trans the crs
    label_df = label_df.to_crs('EPSG:6933')
    predict_df = predict_df.to_crs('EPSG:6933')

    count = len(label_df)

    confusion_matrix = np.ones((11, 11)) * 1e-10

    ori_time = time.time()

    for i, row in label_df.iterrows():
        # if row["DN"] is None:
        #     continue
        now_time = time.time()
        avg_time = (now_time - ori_time) / (i + 1)
        remain_time = avg_time * (count - i)

        geom = row.geometry
        # 检测geom是否valid
        if not geom.is_valid:
            continue
        precise_matches = predict_df[predict_df.geometry.intersects(geom)]

        for idx, match_row in precise_matches.iterrows():
            if not match_row.geometry.is_valid:
                continue
            intersection = match_row.geometry.intersection(geom)
            if intersection.is_empty or isinstance(intersection, Polygon) and intersection.area <= 0.000000000000001:
                continue

            chinese_label = row["euluc_cn"]
            if chinese_label == 'None' or chinese_label is None:
                continue
            if chinese_label == '水体':
                continue
            label = chinese_label_map[chinese_label]

            code = label_chinese_map[match_row["euluc_osm"]]
            # if match_row["cls"] not in cell2euluc_v2.keys():
            #     continue
            # code = label_chinese_map[str(cell2euluc_v2[match_row["cls"]])]
            if code == '水体' or code == '道路':
                continue
            predict = chinese_label_map[code]

            area = intersection.area
            confusion_matrix[label, predict] += area

        # if i % 500 == 0:
        #     print(f"caculate fid:{i}/{count}, remain_time:{remain_time}")

    return confusion_matrix

if __name__ == '__main__':
    # translate_DN_field('/home/faye/DATA/nature/mid_data/all_data/2018/label/2018-gt.shp', '/home/faye/DATA/nature/mid_data/all_data/2018/label/2018_chinese_label.shp')
    # model_names = ["TransE", "GIE", "MurE", "RotE", "RefE", "AttE", "RotH", "RefH", "AttH", "ComplEx", "VecS_2_no_erro"]
    model_names = ["VecS_4"]
    for model_name in model_names:
        ori_dir = os.path.join('/media/dell/DATA/wy/code/CUKG/UrbanKG_Embedding_Model//logs_test', model_name)

        predict_dir = os.path.join(ori_dir, 'predict_shp_4')
        if not os.path.exists(predict_dir):
            os.makedirs(predict_dir)
        ori_seed_dir = "/media/dell/DATA/wy/data/nature_data/mid_data/5_city/2023/seed/"
        ori_label_dir = "/media/dell/DATA/wy/data/nature_data/mid_data/5_city/2023/label/new_label/标注"
        city_map = {
            # "WUHAN": "武汉市",
            # "GUANGZHOU": "广州市",
            # "SHANGHAI": "上海市",
            # "LANZHOU": "兰州市",
            # "YULIN": "榆林市",
            "WUDI": "五地"
        }
        for root, dirs, files in os.walk(ori_dir):
            for file in files:
                if file == 'predict_result.txt' and root.split('/')[-2] in city_map.keys() and 'experiment_4' in root:
                    city_name = root.split('/')[-2]
                    city_chinese_name = city_map[city_name]
                    seed_path = os.path.join(ori_seed_dir, city_chinese_name+'.shp')
                    seed_gdf = gpd.read_file(seed_path, encoding='utf-8')
                    # label_path = os.path.join(ori_label_dir, city_chinese_name+'.shp')
                    # label_gdf = gpd.read_file(label_path, encoding='utf-8')

                    result_txt_path = os.path.join(root, file)
                    txt_result = {}
                    with open(result_txt_path, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            predict_unit_id = int(line.strip().split('\t')[0].split('/')[-1])
                            predict_unit_euluc = line.strip().split('\t')[2].split('/')[-1]
                            txt_result[predict_unit_id] = predict_unit_euluc
                    # 添加euluc_osm字段
                    seed_gdf['euluc_osm'] = None
                    seed_gdf['source'] = None
                    seed_gdf['correct'] = None
                    for i, row in seed_gdf.iterrows():
                        unit_id = row['FID']
                        if not row['euluc'] < 10000:
                            # 新建一个euluc_osm字段，用来存储预测结果
                            row['euluc_osm'] = txt_result[unit_id]
                            row['source'] = 'predict'
                            # row['euluc_osm'] = random.choice(['101', '201', '202', '301', '402', '403', '501', '502', '503', '504', '505', '800', '401'])
                        else:
                            row['euluc_osm'] = str(int(row['euluc']))
                            row['source'] = 'seed'
                            
                        # if unit_id in label_gdf['FID'].values:
                        #     label_row = label_gdf[label_gdf['FID'] == unit_id].iloc[0]
                        #     label_euluc = label_row['euluc_osm']
                            # if label_euluc == row['euluc_osm']:
                            #     row['correct'] = 1
                            # else:
                            #     row['correct'] = 0
                                
                        if row['euluc_osm'] == '800':
                            a = 0
                        seed_gdf.loc[i] = row
                        
                    seed_gdf['euluc_cn'] = seed_gdf['euluc_osm'].map(label_chinese_map)
                    seed_gdf.to_file(os.path.join(predict_dir, city_chinese_name+'.shp'), encoding='utf-8')

        # combine_shp(predict_dir, os.path.join(predict_dir, 'all_result.shp'))
        
        label_path = "/media/dell/DATA/wy/data/nature_data/mid_data/all_data_old/2023/label/2023_chinese_label.shp"
        # label_path = "/home/faye/DATA/nature/mid_data/5_city/2023/label/final_label/all_label.shp"
        for city_names in city_map.values():
            predict_shp = os.path.join(predict_dir, city_names+'.shp')
            if not os.path.exists(predict_shp):
                continue
            city_matrix = overlapEvaluate_geo(label_path, predict_shp)
            df = pd.DataFrame(city_matrix)
            pa, ua, oa = calculate_accuracies(city_matrix)
            print("city_name:{}, mean producers_accuracy:{}, mean users_accuracy:{}, overall_accuracy:{}".format(city_names, pa, ua, oa))
            save_to_excel(city_matrix, pa, ua, oa, os.path.join(predict_dir, city_names+'.xlsx'))

        # # translate_DN_field('/media/dell/DATA/wy/label/2023-gt.shp', '/media/dell/DATA/wy/label/2023_chinese_label.shp')
        # matrix = overlapEvaluate_geo(label_path, os.path.join(predict_dir, 'all_result.shp'))
        # df = pd.DataFrame(matrix)

        # pa, ua, oa = calculate_accuracies(matrix)
        # print("mean producers_accuracy:{}, mean users_accuracy:{}, overall_accuracy:{}".format(pa, ua, oa))
        # # 写入excel
        # save_to_excel(matrix, pa, ua, oa, os.path.join(predict_dir, 'all_result.xlsx'))
