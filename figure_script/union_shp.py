import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import MultiPolygon
from shapely.ops import polygonize
import pandas as pd
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
def merge_adjacent_polygons(gdf, field):
    print("Starting the merging process...")
    
    # 修复几何图形
    print("Fixing geometries using buffer(0)...")
    gdf['geometry'] = gdf['geometry'].buffer(0)

    # 创建一个空的 GeoDataFrame 来存储结果
    merged_gdf_list = []
    print("Initialized an empty list to collect the results.")

    # 按照 field 分组处理
    for name, group in gdf.groupby(field):
        print(f"Processing group: {name}")
        
        # 将每个组中的几何图形进行联合
        merged_geom = unary_union(group.geometry)
        print(f"Unified geometry for group '{name}': {merged_geom}")

        # 如果结果是 MultiPolygon，拆分成多个单个多边形
        if isinstance(merged_geom, MultiPolygon):
            print(f"Group '{name}' resulted in a MultiPolygon. Splitting into individual polygons...")
            polys = list(polygonize(merged_geom))
            for i, poly in enumerate(polys):
                print(f"Adding polygon {i+1} from MultiPolygon to the results list.")
                merged_gdf_list.append({field: group[field].iloc[0], 'geometry': poly})
        else:
            # 如果只是一个简单的多边形，直接添加到结果中
            print(f"Group '{name}' resulted in a single Polygon. Adding to the results list.")
            merged_gdf_list.append({field: group[field].iloc[0], 'geometry': merged_geom})
    
    # 将结果列表转换为 GeoDataFrame
    merged_gdf = gpd.GeoDataFrame(merged_gdf_list, columns=gdf.columns)
    
    print("Merging process completed.")
    return merged_gdf

def map_euluc(shp_path, save_path):
    ori_gdf = gpd.read_file(shp_path, encoding='utf-8')
    ori_gdf['euluc_cn'] = ori_gdf['euluc_osm'].apply(lambda x: label_chinese_map.get(x, '居民地'))
    
    ori_gdf.set_crs('epsg:4326', inplace=True)
    ori_gdf = ori_gdf.to_file(save_path, driver='ESRI Shapefile', encoding="utf-8")

if __name__ == '__main__':
    # shp_path = '/media/dell/DATA/wy/code/CUKG/UrbanKG_Product/logs/VecS_4/predict_shp/all_result.shp'
    # field = 'euluc_osm'
    # ori_gdf = gpd.read_file(shp_path, encoding='utf-8')
    # unioned_gdf = merge_adjacent_polygons(ori_gdf, field)
    # print(unioned_gdf)
    # unioned_gdf.to_file('/media/dell/DATA/wy/code/CUKG/UrbanKG_Product/logs/VecS_4/predict_shp/all_result_union.shp')
    # print('done!')
    
    shp_path = '/media/dell/DATA/wy/code/CUKG/UrbanKG_Product/logs/VecS_4/predict_shp/all_result_union.shp'
    save_path = '/media/dell/DATA/wy/code/CUKG/UrbanKG_Product/logs/VecS_4/predict_shp/all_result_union_cn.shp'
    map_euluc(shp_path, save_path)