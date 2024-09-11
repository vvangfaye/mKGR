import os
from graph_maker import *
from utils import *
import multiprocessing as mp
import shutil
from pypinyin import lazy_pinyin
def main_computing_worker(data_dir, city_name, save_dir):
    # 地理实体（6）： unit(seed), block, area, cell, poi, osm
    unit_path = os.path.join(data_dir, 'unit', city_name + '.shp')
    block_path = os.path.join(data_dir, 'block', city_name + '.shp')
    area_path = os.path.join(data_dir, 'area', city_name + '.shp')
    cell_path = os.path.join(data_dir, 'cell', city_name + '.shp')
    poi_path = os.path.join(data_dir, 'poi_sample_0.1', city_name + '.shp')
    osm_path = os.path.join(data_dir, 'osm', city_name + '.shp')
    seed_path = os.path.join(data_dir, 'seed', city_name + '.shp')
    
    # 语义实体（5）： fineclass, middleclass, coarseclass, cellclass, osmclass
    
    # 空间关系（7）： Unit_Adjacent_to_Unit, Block_Adjacent_to_Block, Unit_In_Block, POI_In_Unit, Unit_Overlap_Area, Unit_Overlap_Cell, Unit_Overlap_OSM
    Unit_Adjacent_to_Unit_path = os.path.join(save_dir, city_name, 'Unit_Adjacent_to_Unit.txt')
    Block_Adjacent_to_Block_path = os.path.join(save_dir, city_name, 'Block_Adjacent_to_Block.txt')
    Unit_In_Block_path = os.path.join(save_dir, city_name, 'Unit_In_Block.txt')
    POI_In_Unit_path = os.path.join(save_dir, city_name, 'POI_In_Unit.txt')
    Unit_Overlap_Area_path = os.path.join(save_dir, city_name, 'Unit_Overlap_Area.txt')
    Unit_Overlap_Cell_path = os.path.join(save_dir, city_name, 'Unit_Overlap_Cell.txt')
    Unit_Overlap_OSM_path = os.path.join(save_dir, city_name, 'Unit_Overlap_OSM.txt')
    
    # 语义关系（9）：Fine_Class_Belong_to_Middle_Class, Middle_Class_Belong_to_Coarse_Class, Fine_Class_Similar_to_EULUC_Class, Cell_Class_Similar_to_EULUC_Class, OSM_Class_Similar_to_EULUC_Class, POI_Has_Fine_Class, Area_Has_Fine_Class, Cell_Has_Cell_Class, OSM_Has_OSM_Class
    Fine_Class_Belong_to_Middle_Class_path = os.path.join(save_dir, city_name, 'Fine_Class_Belong_to_Middle_Class.txt')
    Middle_Class_Belong_to_Coarse_Class_path = os.path.join(save_dir, city_name, 'Middle_Class_Belong_to_Coarse_Class.txt')
    Fine_Class_Similar_to_EULUC_Class_path = os.path.join(save_dir, city_name, 'Fine_Class_Similar_to_EULUC_Class.txt')
    Cell_Class_Similar_to_EULUC_Class_path = os.path.join(save_dir, city_name, 'Cell_Class_Similar_to_EULUC_Class.txt')
    OSM_Class_Similar_to_EULUC_Class_path = os.path.join(save_dir, city_name, 'OSM_Class_Similar_to_EULUC_Class.txt')
    POI_Has_Fine_Class_path = os.path.join(save_dir, city_name, 'POI_Has_Fine_Class.txt')
    Area_Has_Fine_Class_path = os.path.join(save_dir, city_name, 'Area_Has_Fine_Class.txt')
    Cell_Has_Cell_Class_path = os.path.join(save_dir, city_name, 'Cell_Has_Cell_Class.txt')
    OSM_Has_OSM_Class_path = os.path.join(save_dir, city_name, 'OSM_Has_OSM_Class.txt')
    
    # seed关系（1）： Unit_Has_EULUC_Class
    Unit_Has_EULUC_Class_path = os.path.join(save_dir, city_name, 'Unit_Has_EULUC_Class.txt')
    unknow_path = os.path.join(save_dir, city_name, 'unknow.txt')
    # 合并的图谱
    KG_dir = os.path.join(save_dir, city_name, '{}_KG'.format(city_name))
    if not os.path.exists(KG_dir):
        os.makedirs(KG_dir)
    KG_path = os.path.join(KG_dir, '{}_KG.txt'.format(city_name))
    
    
    graph_maker = BaseGraphMaker(unit_path, block_path, area_path, cell_path, poi_path, osm_path, seed_path,
                                 
                                 Unit_Adjacent_to_Unit_path,
                                 Block_Adjacent_to_Block_path, 
                                 Unit_In_Block_path,
                                 POI_In_Unit_path,
                                 Unit_Overlap_Area_path,
                                 Unit_Overlap_Cell_path, 
                                 Unit_Overlap_OSM_path,
                                
                                 Fine_Class_Belong_to_Middle_Class_path, 
                                 Middle_Class_Belong_to_Coarse_Class_path, 
                                 Fine_Class_Similar_to_EULUC_Class_path, 
                                 Cell_Class_Similar_to_EULUC_Class_path, 
                                 OSM_Class_Similar_to_EULUC_Class_path, 
                                 POI_Has_Fine_Class_path, 
                                 Area_Has_Fine_Class_path, 
                                 Cell_Has_Cell_Class_path, 
                                 OSM_Has_OSM_Class_path,
                                 Unit_Has_EULUC_Class_path,
                                 unknow_path,
                                 
                                 KG_path
                                 )
    graph_maker.make_graph()
    
    # 制造训练数据
    entity2id_path = os.path.join(KG_dir, 'entity2id.txt')
    relation2id_path = os.path.join(KG_dir, 'relation2id.txt')
    triple_path = os.path.join(KG_dir, 'triplets.txt')
    train_path = os.path.join(KG_dir, 'train.txt')
    val_path = os.path.join(KG_dir, 'valid.txt')
    test_path = os.path.join(KG_dir, 'test.txt')
    predict_path = os.path.join(KG_dir, 'predict.txt')
    new_label_path = os.path.join(data_dir, 'label', 'new_label', "标注", city_name + '.shp')
    get_entity2id_relation2id(KG_path, unknow_path, entity2id_path, relation2id_path)
    
    # seed_path = os.path.join(data_dir, 'seed', city_name + '.shp')
    label_path = os.path.join(data_dir, 'label', '2023_chinese_label.shp')
    
    # produce_train_val_predict_test(KG_path, entity2id_path, relation2id_path, triple_path, unknow_path, train_path, val_path, predict_path, seed_path, label_path, test_path, new_label_path)
    produce_train_val_predict_test_all(KG_path, entity2id_path, relation2id_path, triple_path, unknow_path, train_path, val_path, predict_path, seed_path, label_path, test_path, new_label_path)
    
    

def sum_rename(save_dir, target_dir):
    for dir_name, _, file_names in os.walk(save_dir):
        for file_name in file_names:
            if "_KG" in dir_name:
                source_path = os.path.join(dir_name, file_name)
                city_name = dir_name.split('/')[-1].split('_')[0][:-1]
                # 汉语转大写拼音
                city_name = ''.join(lazy_pinyin(city_name))
                city_name = city_name.upper()
                target_path = os.path.join(target_dir, city_name, file_name)
                if not os.path.exists(os.path.dirname(target_path)):
                    os.makedirs(os.path.dirname(target_path))
                # 复制文件
                shutil.copy(source_path, target_path)
                
                if file_name == "predict.txt" or file_name == "train.txt" or file_name == "valid.txt" or file_name == "test.txt":
                    # 修改文件名, 去除后缀
                    new_file_name = file_name.split('.')[0]
                    new_file_path = os.path.join(target_dir, city_name, new_file_name)
                    os.rename(target_path, new_file_path)
                    
def combine_graph(save_dir, target_dir):
    # 合并图谱
    KG_dir = os.path.join(target_dir, 'ALL_CITY')
    if not os.path.exists(KG_dir):
        os.makedirs(KG_dir)
    KG_path = os.path.join(KG_dir, 'ALL_KG.txt')
    unknow_path = os.path.join(KG_dir, 'unknow.txt')
    for dir_name, _, file_names in os.walk(save_dir):
        for file_name in file_names:
            if "_KG" in dir_name and file_name.endswith('_KG.txt'):
                source_path = os.path.join(dir_name, file_name)
                with open(source_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                with open(KG_path, 'a', encoding='utf-8') as f:
                    f.writelines(lines)
            if file_name == "unknow.txt":
                source_path = os.path.join(dir_name, file_name)
                with open(source_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                with open(unknow_path, 'a', encoding='utf-8') as f:
                    f.writelines(lines)
    # 制造训练数据
    entity2id_path = os.path.join(KG_dir, 'entity2id.txt')
    relation2id_path = os.path.join(KG_dir, 'relation2id.txt')
    triple_path = os.path.join(KG_dir, 'triplets.txt')
    train_path = os.path.join(KG_dir, 'train.txt')
    val_path = os.path.join(KG_dir, 'valid.txt')
    predict_path = os.path.join(KG_dir, 'predict.txt')
    get_entity2id_relation2id(KG_path, unknow_path, entity2id_path, relation2id_path)
    produce_train_val_predict_test(KG_path, entity2id_path, relation2id_path, triple_path, unknow_path, train_path, val_path, predict_path)

def get_city_list(data_dir):
    name_id_list = []
    for dir_name, _, file_names in os.walk(data_dir):
        for file_name in file_names:
            if file_name.endswith('.shp'):
                name_id_list.append(file_name.split('.')[0])
    return name_id_list

if __name__ == '__main__':
    
    data_dir = '/media/dell/DATA/wy/data/nature_data/mid_data/5_city/2023/'
    save_dir = '/media/dell/DATA/wy/data/nature_data/mid_data/5_city/2023/graph/'
    # name_id_list = ["榆林市", "上海市", "广州市", "武汉市", "兰州市"]
    name_id_list = ["五地"]
    # name_id_list = ["武汉市", "兰州市"]
    # name_id_list = get_city_list(data_dir + 'unit/')
    
    args_list = [(data_dir, name_id, save_dir) for name_id in name_id_list]

    with mp.Pool(5) as p:
        p.starmap(main_computing_worker, args_list)

    save_dir = '/media/dell/DATA/wy/data/nature_data/mid_data/5_city/2023/graph/'
    sum_rename(save_dir, target_dir="/media/dell/DATA/wy/data/nature_data/mid_data/5_city/2023/graph/wudi")
    
    # save_dir = '/home/faye/DATA/nature/mid_data/5_city/2023/graph/'
    # combine_graph(save_dir, target_dir="/home/faye/DATA/nature/mid_data/5_city/2023/graph/sum")
     
    
    
    # for name_id in name_id_list:
    #     main_computing_worker(data_dir, name_id, save_dir)
