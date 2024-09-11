# 获取中文poi的英文名称并保存
import json
import os
import geopandas as gpd
# from googletrans import Translator
# translator = Translator()

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

def get_poi_english_name(poi_file, save_file):
    poi_json = json.load(open(poi_file, 'r', encoding='utf-8'))
    poi_list = poi_json.keys()
    cn_eng_json = {}
    for poi in poi_list:
        print("Translating the poi: {}".format(poi))
        poi_english = translator.translate(poi, dest='en').text
        cn_eng_json[poi] = poi_english
    with open(save_file, 'w', encoding='utf-8') as f:
        json.dump(cn_eng_json, f, ensure_ascii=False, indent=4)
        
# 映射cell
def map_cell2euluc(cell_file, save_file):
    save_dir = os.path.dirname(save_file)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cell_gdf = gpd.read_file(cell_file, encoding='utf-8')
    cell_json = json.load(open("/media/dell/DATA/wy/code/CUKG/map_data/Cell_Class_Similar_to_EULUC_Class.json", 'r', encoding='utf-8'))

    cell_gdf["euluc"] = cell_gdf["cls"].map(cell_json)
    cell_gdf["euluc_cn"] = cell_gdf["euluc"].map(label_chinese_map)
    
    cell_gdf.to_file(save_file, encoding='utf-8')
        
def clip_urbanclip_result(urbanclip_result, save_file):
    bound_dir = '/media/dell/DATA/wy/data/nature_data/mid_data/all_data/2023_bound/split_bound/'
    bound_shp = os.path.join(bound_dir, urbanclip_result.split('/')[-1].split('.')[0] + '.shp')
    
    bound_gdf = gpd.read_file(bound_shp, encoding='utf-8')
    urbanclip_gdf = gpd.read_file(urbanclip_result, encoding='utf-8')
    
    # clip
    urbanclip_gdf = gpd.clip(urbanclip_gdf, bound_gdf)
    urbanclip_gdf.to_file(save_file, encoding='utf-8')
        
if __name__ == '__main__':
    # poi_file = '/media/dell/DATA/wy/code/graph-learning/CUKG/map_data/Fine_Class_Belong_to_Middle_Class.json'
    # poi_json = json.load(open(poi_file, 'r', encoding='utf-8'))
    # poi_list = poi_json.keys()
    # cn_eng_json = {}
    # for poi in poi_list:
    #     print("Translating the poi: {}".format(poi))
    #     cn_eng_json[poi] = ''
    # with open('/media/dell/DATA/wy/code/graph-learning/CUKG/paper/fine_poi_cn_en.json', 'w', encoding='utf-8') as f:
    #     json.dump(cn_eng_json, f, ensure_ascii=False, indent=4)
    city_list = os.listdir('/media/dell/DATA/wy/data/nature_data/mid_data/5_city/2023/cell/')
    # get shp end file
    city_list = [city for city in city_list if city.endswith('.shp')]
    # for city in city_list:
    #     city_name = city.split('.')[0]
    #     print("Processing the city: {}".format(city_name))
    #     cell_file = os.path.join('/media/dell/DATA/wy/data/nature_data/mid_data/5_city/2023/cell/', city)
    #     save_file = os.path.join('/media/dell/DATA/wy/data/nature_data/mid_data/5_city/2023/cell_result_grid/', city_name+'.shp')
    #     map_cell2euluc(cell_file, save_file)
    
    for city in city_list:
        city_name = city.split('.')[0]
        print("Processing the city: {}".format(city_name))
        urbanclip_result = os.path.join('/media/dell/DATA/wy/data/nature_data/mid_data/5_city/2023/output_data/result_urbanclip/', city_name+'.shp')
        save_file = os.path.join('/media/dell/DATA/wy/data/nature_data/mid_data/5_city/2023/output_data/result_urbanclip/', city_name+'_clip.shp')
        clip_urbanclip_result(urbanclip_result, save_file)
                
    