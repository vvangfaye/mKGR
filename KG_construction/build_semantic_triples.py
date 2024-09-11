import geopandas as gpd
import os
# relation map
import json
fine_poi2middle_poi = json.load(open('./map_data/Fine_Class_Belong_to_Middle_Class.json', 'r', encoding='utf-8'))
middle_poi2coarse_poi = json.load(open('./map_data/Middle_Class_Belong_to_Coarse_Class.json', 'r', encoding='utf-8'))
fine_poi2euluc = json.load(open('./map_data/Fine_Class_Similar_to_EULUC_Class.json', 'r', encoding='utf-8'))
cell2euluc = json.load(open('./map_data/Cell_Class_Similar_to_EULUC_Class.json', 'r', encoding='utf-8'))
osm2euluc = json.load(open('./map_data/OSM_Class_Similar_to_EULUC_Class.json', 'r', encoding='utf-8'))

def Fine_Class_Belong_to_Middle_Class(save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, 'w') as save_file:
        for head in fine_poi2middle_poi:
            tail = fine_poi2middle_poi[head]
            line = f"fineclass/{head} Fine_Class_Belong_to_Middle_Class middleclass/{tail}"
            save_file.write(line + '\n')
    print('Fine_Class_Belong_to_Middle_Class done')
        

def Middle_Class_Belong_to_Coarse_Class(save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, 'w') as save_file:
        for head in middle_poi2coarse_poi:
            tail = middle_poi2coarse_poi[head]
            line = f"middleclass/{head} Middle_Class_Belong_to_Coarse_Class coarseclass/{tail}"
            save_file.write(line + '\n')
    print('Middle_Class_Belong_to_Coarse_Class done')

def Fine_Class_Similar_to_EULUC_Class(save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, 'w') as save_file:
        for head in fine_poi2euluc:
            tail = fine_poi2euluc[head]
            line = f"fineclass/{head} Fine_Class_Similar_to_EULUC_Class eulucclass/{tail}"
            save_file.write(line + '\n')
    print('Fine_Class_Similar_to_EULUC_Class done')

def Cell_Class_Similar_to_EULUC_Class(save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, 'w') as save_file:
        for head in cell2euluc:
            tail = cell2euluc[head]
            line = f"cellclass/{head} Cell_Class_Similar_to_EULUC_Class eulucclass/{tail}"
            save_file.write(line + '\n')
    print('Cell_Class_Similar_to_EULUC_Class done')

def OSM_Class_Similar_to_EULUC_Class(save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    with open(save_path, 'w') as save_file:
        for head in osm2euluc:
            tail = osm2euluc[head]
            line = f"osmclass/{head} OSM_Class_Similar_to_EULUC_Class eulucclass/{tail}"
            save_file.write(line + '\n')
    print('OSM_Class_Similar_to_EULUC_Class done')


def POI_Has_Fine_Class(poi_path, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    poi_gdf = gpd.read_file(poi_path, encoding='utf-8')
    city_name = poi_path.split('/')[-1].split('.')[0]
    for i, row in poi_gdf.iterrows():
        poi_id = row['FID']
        poi_class = str(row['type'])
        poi_fine_class = poi_class.split(';')[-1].replace(' ', '')
        if poi_fine_class not in fine_poi2middle_poi:
            continue
        line = f"{city_name}/poi/{poi_id} POI_Has_Fine_Class fineclass/{poi_fine_class}"
        with open(save_path, 'a') as save_file:
            save_file.write(line + '\n')
    print('POI_Has_Fine_Class done')

def Area_Has_Fine_Class(area_path, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    if not os.path.exists(area_path):
        return
    area_gdf = gpd.read_file(area_path, encoding='utf-8')
    city_name = area_path.split('/')[-1].split('.')[0]
    for i, row in area_gdf.iterrows():
        area_id = row['FID']
        area_class = str(row['type'])
        area_fine_class = area_class.split(';')[-1].replace(' ', '')
        if area_fine_class not in fine_poi2middle_poi:
            print(area_fine_class)
            continue
        line = f"{city_name}/area/{area_id} Area_Has_Fine_Class fineclass/{area_fine_class}"
        with open(save_path, 'a') as save_file:
            save_file.write(line + '\n')
    print('Area_Has_Fine_Class done')

def Cell_Has_Cell_Class(cell_path, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    cell_gdf = gpd.read_file(cell_path, encoding='utf-8')
    city_name = cell_path.split('/')[-1].split('.')[0]
    for i, row in cell_gdf.iterrows():
        cell_id = row['FID']
        cell_class = str(row['cls'])
        # if cell_class not in cell2euluc:
        #     continue
        line = f"{city_name}/cell/{cell_id} Cell_Has_Cell_Class cellclass/{cell_class}"
        with open(save_path, 'a') as save_file:
            save_file.write(line + '\n')
    print('Cell_Has_Cell_Class done')

def OSM_Has_OSM_Class(osm_path, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    osm_gdf = gpd.read_file(osm_path, encoding='utf-8')
    city_name = osm_path.split('/')[-1].split('.')[0]
    for i, row in osm_gdf.iterrows():
        osm_id = row['FID']
        osm_class = str(row['code'])
        if osm_class not in osm2euluc:
            continue
        line = f"{city_name}/osm/{osm_id} OSM_Has_OSM_Class osmclass/{osm_class}"
        with open(save_path, 'a') as save_file:
            save_file.write(line + '\n')
    print('OSM_Has_OSM_Class done')