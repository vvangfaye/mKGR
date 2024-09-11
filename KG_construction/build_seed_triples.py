import geopandas as gpd
import os
import json
osm2euluc = json.load(open('./map_data/OSM_Class_Similar_to_EULUC_Class.json', 'r', encoding='utf-8'))
def Unit_Has_EULUC_Class(seed_path, save_path, unknow_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    seed_gdf = gpd.read_file(seed_path, encoding='utf-8')
    city_name = seed_path.split('/')[-1].split('.')[0]
    
    predict_lines= []
    label_lines = []
    for i, row in seed_gdf.iterrows():
        unit_id = row['FID']
        if not row['euluc'] < 10000:
            line = f"{city_name}/unit/{unit_id} Unit_Has_EULUC_Class"
            predict_lines.append(line)
        else:
            osm_class = str(int(row['euluc']))
            if osm_class == '401':
                line = f"{city_name}/unit/{unit_id} Unit_Has_EULUC_Class"
                predict_lines.append(line)
            else:
                line = f"{city_name}/unit/{unit_id} Unit_Has_EULUC_Class eulucclass/{osm_class}"
                label_lines.append(line)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(label_lines))

    with open(unknow_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(predict_lines))
    print('Unit_Has_EULUC_Class done')