import geopandas as gpd
import pandas as pd
import os
import time
import multiprocessing as mp
import json
import random
from tqdm import tqdm
from utils_data import *
from osgeo import ogr, gdal
gdal.SetConfigOption("SHAPE_ENCODING", "")

DN_euluc_map = {
    '1': "101",
    '2': "201",
    '3': "202",
    '4': "301",
    '6': "402",
    '7': "403",
    '8': "501",
    '9': "502",
    '10': "503",
    '11': "504",
    '12': "505",
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
label_english_map = {
    '101': "Residential",
    '201': "Commercial and office area",
    '202': "Commercial service area",
    '301': "Industrial area",
    '402': "Transportation land",
    '403': "Airport",
    '501': "Administrative land",
    '502': "Educational land",
    '503':  "Hospital",
    '504': "Sports and culture",
    '505': "Park and green space",
    '800': "Water body",
    '401': "Road",
}
def random_sample_poi_worker(poi, source_dir, save_dir, sample_rate):
    if poi.endswith('.shp'):
        poi_path = source_dir + poi
        gdf = gpd.read_file(poi_path, encoding='utf-8')
        # 如果大于10000个POI，则随机抽取0.1的POI
        if len(gdf) > 10000:
            gdf = gdf.sample(frac=sample_rate)
        gdf['FID'] = range(len(gdf))
        gdf.reset_index(drop=True, inplace=True)
        gdf.to_file(save_dir + poi.split('.')[0] + '.shp', encoding='utf-8')

def random_sample_poi_parallel(source_dir, save_dir, sample_rate):
    ori_time = time.time()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    poi_list = os.listdir(source_dir)
    # 过滤掉非shp文件
    poi_list = list(filter(lambda x: x.endswith('.shp'), poi_list))
    args_list = [(poi, source_dir, save_dir, sample_rate) for poi in poi_list]
    
    with mp.Pool(processes=40) as pool:
        pool.starmap(random_sample_poi_worker, args_list)
        
    print(f"POI文件采样完成，保存在{save_dir}目录下")
    
def clip_cell_by_boundary(bound_path, cell_path, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cell_gdf = gpd.read_file(cell_path, encoding='utf-8')
    
    bound_gdf = gpd.read_file(bound_path, encoding='utf-8')
    clipped_cell = gpd.overlay(cell_gdf, bound_gdf)
    
    bound_id = clipped_cell['dt_name'].unique()
    for i, dt_name in enumerate(bound_id):
        print("clip_cell_by_boundary: {}/{}".format(i, len(bound_id)))
        row = clipped_cell[clipped_cell['dt_name'] == dt_name]
        ct_name = row['ct_name'].unique()
        for ct in ct_name:
            ct_row = row[row['ct_name'] == ct]
            ct_row.to_file(os.path.join(save_dir, "{}_{}.shp".format(ct, dt_name)), driver='ESRI Shapefile', encoding='utf-8')
            
def combine_target_cell(target_cities, cell_dir, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    cell_shp_list = os.listdir(cell_dir)
    cell_shp_list = [cell_shp for cell_shp in cell_shp_list if cell_shp.endswith(".shp")]
    for city in target_cities:
        city_shp_list = [cell_shp for cell_shp in cell_shp_list if city in cell_shp]
        for city_shp in city_shp_list:
            shp_path = os.path.join(cell_dir, city_shp)
            if city_shp_list.index(city_shp) == 0:
                df = gpd.read_file(shp_path, encoding='utf-8')
            else:
                df = pd.concat([df, gpd.read_file(shp_path, encoding='utf-8')], axis=0)
        save_path = os.path.join(save_dir, city + ".shp")

        df['FID'] = range(len(df))
        df.reset_index(drop=True, inplace=True)
        df.to_file(save_path, encoding='utf-8')
        
def make_map_data(poi_fine_to_middle_path, poi_middle_to_coarse_path, save_dir):
    # fliter poi_fine_to_middle
    with open(poi_fine_to_middle_path, 'r', encoding='utf-8') as f:
        poi_fine_to_middle = f.readlines()
    poi_fine_to_middle = poi_fine_to_middle[1:]
    poi_fine_to_middle_json = {}
    for i, line in enumerate(poi_fine_to_middle):
        fine, relation, middle = line.strip().split('\t')
        fine = fine.split(';')[-1]
        middle = middle.split(';')[-1]
        poi_fine_to_middle_json[fine] = middle
    
    json.dump(poi_fine_to_middle_json, open(os.path.join(save_dir, 'Fine_Class_Belong_to_Middle_Class.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    
    # fliter poi_middle_to_coarse
    with open(poi_middle_to_coarse_path, 'r', encoding='utf-8') as f:
        poi_middle_to_coarse = f.readlines()
    poi_middle_to_coarse = poi_middle_to_coarse[1:]
    poi_middle_to_coarse_json = {}
    for i, line in enumerate(poi_middle_to_coarse):
        middle, relation, coarse = line.strip().split('\t')
        middle = middle.split(';')[-1]
        coarse = coarse.split(';')[-1]
        poi_middle_to_coarse_json[middle] = coarse
        
    json.dump(poi_middle_to_coarse_json, open(os.path.join(save_dir, 'Middle_Class_Belong_to_Coarse_Class.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    
    # fliter poi_to_euluc
    poi_to_euluc_json = {}
    ori_maps = poi2euluc
    for poi_map in ori_maps:
        euluc = ori_maps[poi_map]
        poi_map = poi_map.split(';')[-1]
        poi_to_euluc_json[str(poi_map)] = str(euluc)
    
    json.dump(poi_to_euluc_json, open(os.path.join(save_dir, 'Fine_Class_Similar_to_EULUC_Class.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    
    # fliter cell_to_euluc
    cell_to_euluc_json = {}
    ori_maps = cell2euluc
    for cell_map in ori_maps:
        euluc = ori_maps[cell_map]
        cell_to_euluc_json[str(cell_map)] = str(euluc)
    
    json.dump(cell_to_euluc_json, open(os.path.join(save_dir, 'Cell_Class_Similar_to_EULUC_Class.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    
    # fliter osm_to_euluc
    osm_to_euluc_json = {}
    ori_maps = osm2euluc
    for osm_map in ori_maps:
        euluc = ori_maps[osm_map]
        osm_to_euluc_json[str(osm_map)] = str(euluc)
    
    # 格式化保存
    json.dump(osm_to_euluc_json, open(os.path.join(save_dir, 'OSM_Class_Similar_to_EULUC_Class.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

def add_FID_for_area(area_dir):
    area_list = os.listdir(area_dir)
    area_list = [area for area in area_list if area.endswith(".shp")]
    for area in area_list:
        area_path = os.path.join(area_dir, area)
        gdf = gpd.read_file(area_path, encoding='utf-8')
        # to 4326
        gdf = gdf.to_crs(epsg=4326)
        gdf.to_file(area_path, encoding='utf-8')
        
def split_train_val(h_r_t, rate):
    # 保证每个类别的三元组都在训练集/验证集中
    train_h_r_t = []
    val_h_r_t = []
    euluc_set = set()
    for i in range(len(h_r_t)):
        euluc_set.add(h_r_t[i][2])
    for euluc in euluc_set:
        temp = []
        for i in range(len(h_r_t)):
            if h_r_t[i][2] == euluc:
                temp.append(h_r_t[i])
        random.shuffle(temp)
        train = int(len(temp) * rate)
        val = len(temp) - train
        train_h_r_t = train_h_r_t + temp[:train]
        val_h_r_t = val_h_r_t + temp[train:]
    return train_h_r_t, val_h_r_t

def split_train_val_all(h_r_t, rate):
    random.shuffle(h_r_t)
    train = int(len(h_r_t) * rate)
    val = len(h_r_t) - train
    train_h_r_t = h_r_t[:train]
    val_h_r_t = h_r_t[train:]
    return train_h_r_t, val_h_r_t
    
    
def get_entity2id_relation2id(KG, unknow_path, entity2id, relation2id):
    entity = []
    relations = []
    with open(KG) as f:
        for line in f.readlines():
            temp = line.split()
            entity.append(temp[0])
            entity.append(temp[2])
            relations.append(temp[1])
    with open(unknow_path, encoding= 'utf-8') as f1:
        for line in f1.readlines():
            temp1 = line.split()
            entity.append(temp1[0])
            relations.append(temp1[1])
    entity = list(set(entity))
    relations = list(set(relations))
    f.close()

    with open(entity2id,'w') as f2:
        for i in range(len(entity)):
            f2.write(entity[i] + '\t')
            f2.write(str(i))
            f2.write('\n')
        f2.close()


    with open(relation2id,'w') as f3:
        for j in range(len(relations)):
            f3.write(relations[j]+'\t')
            f3.write(str(j))
            f3.write('\n')
        f3.close()


def produce_train_val_predict_test(KG, entity2id, realtion2id, triple, unknow_path, train_path, val_path, predict_path,
                                   seed_path, label_path, test_path, new_label_path):
    h_r_t = []

    h = []
    r = []
    t = []
    with open(KG) as f:
        for line in f.readlines():
            temp = line.split()
            h.append(temp[0])
            t.append(temp[2])
            r.append(temp[1])

    p_h = []
    p_r = []
    with open(unknow_path) as f1:
        for line in f1.readlines():
            temp1 = line.split()
            p_h.append(temp1[0])
            p_r.append(temp1[1])

    entity_category_dict = {}
    relation_category_dict = {}
    with open(entity2id) as f:
        for line in f.readlines():
            temp = line.split()
            entity_category_dict.update({temp[0]: temp[1]})
    with open(realtion2id) as f1:
        for line in f1.readlines():
            temp1 = line.split()
            relation_category_dict.update({temp1[0]: temp1[1]})

    with open(triple, 'w') as f2:
        for i in tqdm(range(len(h))):
            f2.write(str(entity_category_dict[h[i]]) + '\t')
            f2.write(str(relation_category_dict[r[i]]) + '\t')
            f2.write(str(entity_category_dict[t[i]]))
            f2.write('\n')

    with open(predict_path, 'w') as f3:
        for i in tqdm(range(len(p_h))):
            f3.write(str(entity_category_dict[p_h[i]]) + '\t')
            f3.write(str(relation_category_dict[p_r[i]]))
            f3.write('\n')

    # # 从unknow中选出label中存在的三元组，要求用空间分析
    # test_h_r_t = []
    # # Open the data sources
    # seed_ds = ogr.Open(seed_path, 1)
    # label_ds = ogr.Open(label_path, 1)
    # new_label_ds = ogr.Open(new_label_path, 1)

    # # Get the first layer
    # seed_layer = seed_ds.GetLayer()
    # label_layer = label_ds.GetLayer()
    # new_label_layer = new_label_ds.GetLayer()

    # graph = {}
    # count = seed_layer.GetFeatureCount()

    # # 创建空间索引
    # seed_sql = 'CREATE SPATIAL INDEX ON ' + seed_layer.GetName()
    # seed_ds.ExecuteSQL(seed_sql)
    # label_sql = 'CREATE SPATIAL INDEX ON ' + label_layer.GetName()
    # label_ds.ExecuteSQL(label_sql)
    # city_name = seed_path.split('/')[-1].split('.')[0]
    # ori_id_list = []
    # for i, feature in enumerate(seed_layer):
    #     label_layer.SetSpatialFilter(None)
    #     geom = feature.GetGeometryRef()
    #     # Find intersecting features using spatial index
    #     label_layer.SetSpatialFilter(geom)
    #     fid = feature.GetField("FID")
        
    #     # if feature.GetField("euluc") is not None:
    #     #     continue

    #     precise_matches = [f for f in label_layer if
    #                        f.GetGeometryRef().Intersection(geom) is not None and f.GetGeometryRef().Intersection(
    #                            geom).Area() > 0.000000000000001]
    #     for precise_match in precise_matches:
    #         intersect_area = precise_match.GetGeometryRef().Intersection(geom).Area()
    #         label_area = precise_match.GetGeometryRef().Area()
    #         seed_area = geom.Area()
    #         if intersect_area / label_area > 0.5 and intersect_area / seed_area > 0.5:
    #             unit_id = feature.GetField("FID")
    #             ori_id_list.append(unit_id)
    #             unit_dn = precise_match.GetField("DN")
    #             if str(unit_dn) not in DN_euluc_map:
    #                 continue
    #             unit_euluc_label = DN_euluc_map[str(unit_dn)]
    #             head = city_name + '/unit/' + str(unit_id)
    #             relation = 'Unit_Has_EULUC_Class'
    #             tail = 'eulucclass/' + unit_euluc_label
    #             test_h_r_t.append([head, relation, tail])
                
    # for i, feature in enumerate(new_label_layer):
    #     fid = feature.GetField("FID")
    #     euluc_label = feature.GetField("euluc_osm")
    #     if fid not in ori_id_list:
    #         head = city_name + '/unit/' + str(fid)
    #         relation = 'Unit_Has_EULUC_Class'
    #         tail = 'eulucclass/' + euluc_label
    #         test_h_r_t.append([head, relation, tail])

    # 挑选出关系为Unit_Has_EULUC_Class的三元组
    euluc_h_r_t = []
    other_h_r_t = []
    for i in range(len(h)):
        if r[i] == 'Unit_Has_EULUC_Class':
            euluc_h_r_t.append([h[i], r[i], t[i]])
        else:
            other_h_r_t.append([h[i], r[i], t[i]])
    random.shuffle(euluc_h_r_t)
    # 选取0.2的三元组作为val
    train_h_r_t, val_h_r_t = split_train_val(euluc_h_r_t, 0.8)
    # 选取0.5的val作为test
    test_h_r_t, val_h_r_t = split_train_val(val_h_r_t, 0.5)
    # 拼接其他关系的三元组
    train_h_r_t.extend(other_h_r_t)
    # 打乱
    random.shuffle(train_h_r_t)
    # 写入文件
    with open(train_path, 'w') as f3:
        for i in tqdm(range(len(train_h_r_t))):
            f3.write(str(entity_category_dict[train_h_r_t[i][0]]) + '\t')
            f3.write(str(relation_category_dict[train_h_r_t[i][1]]) + '\t')
            f3.write(str(entity_category_dict[train_h_r_t[i][2]]))
            f3.write('\n')
    f3.close()
    with open(val_path, 'w') as f4:
        for i in tqdm(range(len(val_h_r_t))):
            f4.write(str(entity_category_dict[val_h_r_t[i][0]]) + '\t')
            f4.write(str(relation_category_dict[val_h_r_t[i][1]]) + '\t')
            f4.write(str(entity_category_dict[val_h_r_t[i][2]]))
            f4.write('\n')
    f4.close()
    with open(test_path, 'w') as f5:
        for i in tqdm(range(len(test_h_r_t))):
            f5.write(str(entity_category_dict[test_h_r_t[i][0]]) + '\t')
            f5.write(str(relation_category_dict[test_h_r_t[i][1]]) + '\t')
            f5.write(str(entity_category_dict[test_h_r_t[i][2]]))
            f5.write('\n')
    f5.close()
    
    
def produce_train_val_predict(KG, entity2id, realtion2id, triple, unknow_path, train_path, val_path, predict_path):

    h = []
    r = []
    t = []
    with open(KG) as f:
        for line in f.readlines():
            temp = line.split()
            h.append(temp[0])
            t.append(temp[2])
            r.append(temp[1])

    p_h = []
    p_r = []
    with open(unknow_path, encoding= 'utf-8') as f1:
        for line in f1.readlines():
            temp1 = line.split()
            p_h.append(temp1[0])
            p_r.append(temp1[1])

    entity_category_dict = {}
    relation_category_dict = {}
    with open(entity2id) as f:
        for line in f.readlines():
            temp = line.split()
            entity_category_dict.update({temp[0]: temp[1]})
    with open(realtion2id) as f1:
        for line in f1.readlines():
            temp1 = line.split()
            relation_category_dict.update({temp1[0]: temp1[1]})

    with open(triple, 'w') as f2:
        for i in tqdm(range(len(h))):
            f2.write(str(entity_category_dict[h[i]]) + '\t')
            f2.write(str(relation_category_dict[r[i]]) + '\t')
            f2.write(str(entity_category_dict[t[i]]))
            f2.write('\n')

    with open(predict_path, 'w') as f3:
        for i in tqdm(range(len(p_h))):
            f3.write(str(entity_category_dict[p_h[i]]) + '\t')
            f3.write(str(relation_category_dict[p_r[i]]))
            f3.write('\n')

    # 挑选出关系为Unit_Has_EULUC_Class的三元组
    euluc_h_r_t = []
    other_h_r_t = []
    for i in range(len(h)):
        if r[i] == 'Unit_Has_EULUC_Class':
            euluc_h_r_t.append([h[i], r[i], t[i]])
        else:
            other_h_r_t.append([h[i], r[i], t[i]])
    random.shuffle(euluc_h_r_t)
    # 选取0.2的三元组作为val
    train_h_r_t, val_h_r_t = split_train_val(euluc_h_r_t, 0.8)
    # 拼接其他关系的三元组
    train_h_r_t.extend(other_h_r_t)
    # 打乱
    random.shuffle(train_h_r_t)
    # 写入文件
    with open(train_path, 'w') as f3:
        for i in tqdm(range(len(train_h_r_t))):
            f3.write(str(entity_category_dict[train_h_r_t[i][0]]) + '\t')
            f3.write(str(relation_category_dict[train_h_r_t[i][1]]) + '\t')
            f3.write(str(entity_category_dict[train_h_r_t[i][2]]))
            f3.write('\n')
    f3.close()
    with open(val_path, 'w') as f4:
        for i in tqdm(range(len(val_h_r_t))):
            f4.write(str(entity_category_dict[val_h_r_t[i][0]]) + '\t')
            f4.write(str(relation_category_dict[val_h_r_t[i][1]]) + '\t')
            f4.write(str(entity_category_dict[val_h_r_t[i][2]]))
            f4.write('\n')
    f4.close()

def get_train_val(triple, train_address, valid_address, test_address):
    h_r_t = []
    with open(triple) as f:
        for line in f.readlines():
            temp = line.split()
            h_r_t.append(temp)
    random.shuffle(h_r_t)
    train = int(len(h_r_t) * 0.9)
    valid = int(len(h_r_t) * 0.05)
    test = len(h_r_t) - train - valid

    with open(train_address, 'w') as f:
        for i in range(train):
            f.write(h_r_t[i][0] + '\t')
            f.write(h_r_t[i][1] + '\t')
            f.write(h_r_t[i][2])
            f.write('\n')
    f.close()

    with open(valid_address, 'w') as f:
        for i in range(valid):
            f.write(h_r_t[train + i][0] + '\t')
            f.write(h_r_t[train + i][1] + '\t')
            f.write(h_r_t[train + i][2])
            f.write('\n')
    f.close()

    with open(test_address, 'w') as f:
        for i in range(test):
            f.write(h_r_t[train + valid + i][0] + '\t')
            f.write(h_r_t[train + valid + i][1] + '\t')
            f.write(h_r_t[train + valid + i][2])
            f.write('\n')
    f.close()
def trans_euluc_to_chinese(input_shp, output_shp):
    # 读取输入SHP文件
    gdf = gpd.read_file(input_shp, encoding='utf-8')

    # 使用map函数和label_chinese_map字典来创建新的euluc_cn字段
    gdf['euluc'] = gdf['euluc'].astype(str)
    gdf['euluc_cn'] = gdf['euluc'].map(label_chinese_map)

    # 保存到输出SHP文件
    gdf.to_file(output_shp, encoding='utf-8')

def get_new_label(ori_label, new_label, unit_dir):
    if not os.path.exists(new_label):
        os.makedirs(new_label)
    
    city_list = os.listdir(ori_label)
    # 以shp结尾的
    city_list = [city for city in city_list if city.endswith('.shp')]
    # 通过unit和label的交集来获取新的label
    for city in city_list:
        city_name = city.split('.')[0]
        label_path = os.path.join(ori_label, city)
        unit_path = os.path.join(unit_dir, city_name + '.shp')
        label_gdf = gpd.read_file(label_path, encoding='utf-8')
        unit_gdf = gpd.read_file(unit_path, encoding='utf-8')
        # 设置坐标系
        new_label_gdf = gpd.GeoDataFrame(columns=['FID', 'geometry', 'euluc_osm', 'euluc_cn'])
        
        for i, row in unit_gdf.iterrows():
            unit_id = row['FID']
            unit_geom = row['geometry']
            # 通过unit和label的交集来获取新的label
            for j, label_row in label_gdf.iterrows():
                label_geom = label_row['geometry']
                if unit_geom.intersects(label_geom):
                    # 计算重叠
                    intersect_area = unit_geom.intersection(label_geom).area
                    if intersect_area / label_geom.area < 0.5 and intersect_area / unit_geom.area < 0.5: 
                        continue
                    new_label_gdf.loc[len(new_label_gdf)] = [unit_id, unit_geom, label_row['euluc_osm'], label_row['euluc_cn']]
                    break
                
        new_label_gdf = new_label_gdf.set_crs(epsg=4326)
        new_label_gdf.to_file(os.path.join(new_label, city), encoding='utf-8')
     
def produce_train_val_predict_test_all(KG, entity2id, realtion2id, triple, unknow_path, train_path, val_path, predict_path,
                                   seed_path, label_path, test_path, new_label_path):
    h_r_t = []

    h = []
    r = []
    t = []
    with open(KG) as f:
        for line in f.readlines():
            temp = line.split()
            h.append(temp[0])
            t.append(temp[2])
            r.append(temp[1])

    p_h = []
    p_r = []
    with open(unknow_path) as f1:
        for line in f1.readlines():
            temp1 = line.split()
            p_h.append(temp1[0])
            p_r.append(temp1[1])

    entity_category_dict = {}
    relation_category_dict = {}
    with open(entity2id) as f:
        for line in f.readlines():
            temp = line.split()
            entity_category_dict.update({temp[0]: temp[1]})
    with open(realtion2id) as f1:
        for line in f1.readlines():
            temp1 = line.split()
            relation_category_dict.update({temp1[0]: temp1[1]})

    with open(triple, 'w') as f2:
        for i in tqdm(range(len(h))):
            f2.write(str(entity_category_dict[h[i]]) + '\t')
            f2.write(str(relation_category_dict[r[i]]) + '\t')
            f2.write(str(entity_category_dict[t[i]]))
            f2.write('\n')

    with open(predict_path, 'w') as f3:
        for i in tqdm(range(len(p_h))):
            f3.write(str(entity_category_dict[p_h[i]]) + '\t')
            f3.write(str(relation_category_dict[p_r[i]]))
            f3.write('\n')

    all_h_r_t = []
    for i in range(len(h)):
        all_h_r_t.append([h[i], r[i], t[i]])

    # 选取0.2的三元组作为val,test
    train_h_r_t, val_h_r_t = split_train_val_all(all_h_r_t, 0.8)
    # 选取0.5的val作为test
    test_h_r_t, val_h_r_t = split_train_val_all(val_h_r_t, 0.5)

    # 写入文件
    with open(train_path, 'w') as f3:
        for i in tqdm(range(len(train_h_r_t))):
            f3.write(str(entity_category_dict[train_h_r_t[i][0]]) + '\t')
            f3.write(str(relation_category_dict[train_h_r_t[i][1]]) + '\t')
            f3.write(str(entity_category_dict[train_h_r_t[i][2]]))
            f3.write('\n')
    f3.close()
    with open(val_path, 'w') as f4:
        for i in tqdm(range(len(val_h_r_t))):
            f4.write(str(entity_category_dict[val_h_r_t[i][0]]) + '\t')
            f4.write(str(relation_category_dict[val_h_r_t[i][1]]) + '\t')
            f4.write(str(entity_category_dict[val_h_r_t[i][2]]))
            f4.write('\n')
    f4.close()
    with open(test_path, 'w') as f5:
        for i in tqdm(range(len(test_h_r_t))):
            f5.write(str(entity_category_dict[test_h_r_t[i][0]]) + '\t')
            f5.write(str(relation_category_dict[test_h_r_t[i][1]]) + '\t')
            f5.write(str(entity_category_dict[test_h_r_t[i][2]]))
            f5.write('\n')
    f5.close()
    
def combine_shp_and_reset_fid(shp_list, save_path):
    sum_gpd = gpd.GeoDataFrame()
    for i, shp in enumerate(shp_list):
        poi_gdf = gpd.read_file(shp, encoding='utf-8')
        sum_gpd = pd.concat([sum_gpd, poi_gdf], axis=0)
    
    # 重设FID
    sum_gpd['FID'] = range(len(sum_gpd))
    sum_gpd.reset_index(drop=True, inplace=True)
    
    # 保存
    sum_gpd.to_file(save_path, encoding='utf-8')
    
def combine_city(city_list, data_dir_list):
    for data_dir in tqdm(data_dir_list):
        shp_list = [os.path.join(data_dir, city + '.shp') for city in city_list]
        save_path = os.path.join(data_dir, '五地.shp')
        combine_shp_and_reset_fid(shp_list, save_path)
        
 
if __name__ == '__main__':
    # trans_euluc_to_chinese("/media/dell/DATA/wy/种子点/2023-result/2023-euluc.shp", "/media/dell/DATA/wy/种子点/2023-result/2023_euluc_cn.shp")
    # bound_path = "/home/faye/DATA/nature_data/pre_data/all_data/2023_bound/2023_bound.shp"
    # cell_path = "/home/faye/DATA/nature_data/ori_data/cell/城市功能区分类和场景分类结果/2023_scene_classifcation.shp"
    # save_dir = "/home/faye/DATA/nature_data/pre_data/all_data/2023/cell/"
    # clip_cell_by_boundary(bound_path, cell_path, save_dir)
    # random_sample_poi_parallel('/home/faye/DATA/nature/mid_data/all_data/2023/poi/', '/home/faye/DATA/nature/mid_data/all_data/2023/poi_sample_0.1/', 0.1)
    # target_cities = ['上海市', '广州市', '武汉市', '榆林市', '兰州市']
    # cell_dir = "/home/faye/DATA/nature_data/pre_data/all_data/2023/cell/"
    # save_dir = "/home/faye/DATA/nature_data/select_data/cell/"
    # combine_target_cell(target_cities, cell_dir, save_dir)

    # poi_fine_to_middle_path = "/home/faye/code/CUKG/map_data/ori_data/Fine_Class_Belong_to_Middle_Class.txt"
    # poi_middle_to_coarse_path = "/home/faye/code/CUKG/map_data/ori_data/Middle_Class_Belong_to_Coarse_Class.txt"
    # save_dir = "/home/faye/code/CUKG/map_data/"
    # make_map_data(poi_fine_to_middle_path, poi_middle_to_coarse_path, save_dir)

    # add_FID_for_area("/home/faye/DATA/nature_data/select_data/area/")
    
    # get_new_label("/home/faye/DATA/nature/mid_data/5_city/2023/label/new_label/标注", "/home/faye/DATA/nature/mid_data/5_city/2023/label/new_label/标注_new",
    #               "/home/faye/DATA/nature/mid_data/5_city/2023/unit")
    city_list = ['上海市', '广州市', '武汉市', '榆林市', '兰州市']
    unit_list = ["area", "cell", "poi_sample_0.1", "osm", "unit", "block", "seed"]
    data_dir = "/media/dell/DATA/wy/data/nature_data/mid_data/5_city/2023/"
    data_dir_list = [data_dir + unit for unit in unit_list]
    
    combine_city(city_list, data_dir_list)