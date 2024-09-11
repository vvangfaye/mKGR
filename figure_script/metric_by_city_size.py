import os
import geopandas as gpd
from osgeo import gdal, ogr, osr
import pandas as pd
import json
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
def get_semantic_size():
    poi_dir = '/media/dell/DATA/wy/data/nature_data/mid_data/all_data/2023/poi_sample_0.1/'
    AOI_dir = '/media/dell/DATA/wy/data/nature_data/mid_data/all_data/2023/osm/'
    unit_dir = '/media/dell/DATA/wy/data/nature_data/mid_data/all_data/2023/unit/'
    city_list = os.listdir(poi_dir)
    # 后缀为.shp
    city_list = [city for city in city_list if city.endswith('.shp')]
    
    poi_num = {}
    for city in city_list:
        print(city)
        city_path = os.path.join(poi_dir, city)
        poi_ds = ogr.Open(city_path)
        aoi_ds = ogr.Open(os.path.join(AOI_dir, city))
        unit_ds = ogr.Open(os.path.join(unit_dir, city))
        poi_feature_num = poi_ds.GetLayer().GetFeatureCount()
        aoi_feature_num = aoi_ds.GetLayer().GetFeatureCount()
        
        unit_feature_num = unit_ds.GetLayer().GetFeatureCount()
        
        semantic_value = poi_feature_num + aoi_feature_num
        
        poi_num[city.replace('.shp', '')] = {
            'poi_num': poi_feature_num,
            'aoi_num': aoi_feature_num,
            'unit_num': unit_feature_num,
            'semantic_value': semantic_value
        }
    
    
    # 按照semantic_value排序
    poi_num = sorted(poi_num.items(), key=lambda x: x[1]['semantic_value'], reverse=True)
    json.dump(poi_num, open('semantic_num.json', 'w'), indent=4, ensure_ascii=False)
    print(poi_num)

def combine_shp(shp_list, save_path):
                
    df = gpd.GeoDataFrame()  # 初始化一个空的GeoDataFrame
    for shp in shp_list:
        print("{}/{}, {}".format(shp_list.index(shp), len(shp_list), shp))
        if not os.path.exists(shp):
            continue
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

if __name__ == "__main__":
    semantic_size = json.load(open('/media/dell/DATA/wy/code/CUKG/paper/metics/semantic_num.json', 'r'))
    city_list_1 = []
    city_list_2 = []
    city_list_3 = []
    city_list_4 = []
    for city, value in semantic_size:
        if value['semantic_value'] > 50000:
            city_list_1.append(city)
        elif value['semantic_value'] > 10000:
            city_list_2.append(city)
        elif value['semantic_value'] > 5000:
            city_list_3.append(city)
        else:
            city_list_4.append(city)
    
    # combine_shp([os.path.join('/media/dell/DATA/wy/code/CUKG/UrbanKG_Product/logs/VecS_4/predict_shp', city + '.shp') for city in city_list_1], '/media/dell/DATA/wy/code/CUKG/UrbanKG_Product/logs/VecS_4/predict_shp/city_1.shp')
    # combine_shp([os.path.join('/media/dell/DATA/wy/code/CUKG/UrbanKG_Product/logs/VecS_4/predict_shp', city + '.shp') for city in city_list_2], '/media/dell/DATA/wy/code/CUKG/UrbanKG_Product/logs/VecS_4/predict_shp/city_2.shp')
    # combine_shp([os.path.join('/media/dell/DATA/wy/code/CUKG/UrbanKG_Product/logs/VecS_4/predict_shp', city + '.shp') for city in city_list_3], '/media/dell/DATA/wy/code/CUKG/UrbanKG_Product/logs/VecS_4/predict_shp/city_3.shp')
    combine_shp([os.path.join('/media/dell/DATA/wy/code/CUKG/UrbanKG_Product/logs/VecS_4/predict_shp', city + '.shp') for city in city_list_4], '/media/dell/DATA/wy/code/CUKG/UrbanKG_Product/logs/VecS_4/predict_shp/city_4.shp')