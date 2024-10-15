import os

import geopandas as gpd

def clip_result_by_boundary(bound_path, result_path, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    result_gdf = gpd.read_file(result_path, encoding='utf-8')

    bound_gdf = gpd.read_file(bound_path, encoding='utf-8')
    clipped_result = gpd.overlay(result_gdf, bound_gdf)
    clipped_result["euluc_result"] = clipped_result["euluc"]

    bound_id = clipped_result['ct_name'].unique()
    for i, ct_name in enumerate(bound_id):
        if os.path.exists(os.path.join(save_dir, "{}.shp".format(ct_name))):
            continue
        print("clip_result_by_boundary: {}/{}".format(i, len(bound_id)))
        row = clipped_result[clipped_result['ct_name'] == ct_name]
        # 添加FID列，FID从0开始
        row['FID'] = range(len(row))
        # 重设索引
        row = row.reset_index(drop=True)
        row.to_file(os.path.join(save_dir, "{}.shp".format(ct_name)), driver='ESRI Shapefile',
                    encoding='utf-8')
        
        
# 切割result
bound_path = "/home/wangyu/data/landuse/2023_bound.shp"
result_path = "/home/wangyu/data/landuse/ChinaLandUse.shp"
result_save_dir = "/home/wangyu/data/landuse/city"
clip_result_by_boundary(bound_path, result_path, result_save_dir)