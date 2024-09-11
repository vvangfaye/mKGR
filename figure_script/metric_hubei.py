import os
import pandas as pd
import geopandas as gpd

def sum_poi_by_pr(poi_list, poi_dir, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # 合并poi_list
    sum_gpd = gpd.GeoDataFrame()
    for i, poi in enumerate(poi_list):
        if i % 10 == 0:
            print("sum_poi_by_pr: {}/{}".format(int(i), len(poi_list)))
        poi_path = os.path.join(poi_dir, poi+".shp")
        poi_gdf = gpd.read_file(poi_path)
        sum_gpd = pd.concat([sum_gpd, poi_gdf], axis=0)
    
    # 重设FID
    sum_gpd['FID'] = range(len(sum_gpd))
    sum_gpd.reset_index(drop=True, inplace=True)
    
    # 保存
    sum_gpd.to_file(os.path.join(save_dir, "湖北省.shp"), encoding='utf-8')
    
def clip_predict(bound_path, result_path):
    # 读取边界
    bound_gdf = gpd.read_file(bound_path, encoding='utf-8')
    # 读取结果
    result_gdf = gpd.read_file(result_path, encoding='utf-8')
    # 合并边界和结果
    clip_gdf = gpd.overlay(bound_gdf, result_gdf)
    # 保存
    clip_gdf.to_file("/home/faye/DATA/nature/mid_data/all_data_pr/2023/result_concat/武汉市.shp", encoding='utf-8')
    
hubei_list = ["武汉市", "黄石市", "十堰市", "宜昌市", "襄阳市", "鄂州市", "荆门市", "孝感市", "荆州市", "黄冈市", "咸宁市", "随州市", "恩施土家族苗族自治州", "仙桃市", "潜江市", "天门市"]
# 合并predict
# sum_poi_by_pr(hubei_list, "/home/faye/DATA/code/CUKG/UrbanKG_Product/logs/VecS_4/predict_shp", "/home/faye/DATA/nature/mid_data/all_data_pr/2023/result_split")

# 切割predict
# clip_predict("/home/faye/DATA/nature/mid_data/all_data/2023_bound/split_bound/武汉市.shp", "/home/faye/DATA/nature/mid_data/all_data_pr/2023/result_concat/湖北省.shp")
