import argparse
import json
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.optim
import pandas as pd
import numpy as np
import models
import optimizers.regularizers as regularizers
from datasets.kg_dataset import KGDataset
from models import all_models
DATA_PATH = './data_euluc'

parser = argparse.ArgumentParser(
    description="Urban Knowledge Graph Embedding"
)
parser.add_argument(
    "--dataset", default="WUHAN", choices=["NYC", "CHI", "WUHAN", "GUANGZHOU", "SHANGHAI", "LANZHOU", "YULIN"],
    help="Urban Knowledge Graph dataset"
)
parser.add_argument(
    "--model", default="TransE", choices=all_models, help='"TransE", "CP", "MurE", "RotE", "RefE", "AttE",'
                                                       '"ComplEx", "RotatE",'
                                                       '"RotH", "RefH", "AttH"'
                                                       '"GIE "VecS_2_neg_reg3"',
)
parser.add_argument(
    "--optimizer", choices=["Adagrad", "Adam", "SparseAdam"], default="Adam",
    help="Optimizer"
)
parser.add_argument(
    "--max_epochs", default=150, type=int, help="Maximum number of epochs to train for"
)
parser.add_argument(
    "--patience", default=10, type=int, help="Number of epochs before early stopping"
)
parser.add_argument(
    "--valid", default=3, type=float, help="Number of epochs before validation"
)
parser.add_argument(
    "--rank", default=32, type=int, help="Embedding dimension"
)
parser.add_argument(
    "--batch_size", default=4120, type=int, help="Batch size"
)
parser.add_argument(
    "--learning_rate", default=1e-3, type=float, help="Learning rate"
)
parser.add_argument(
    "--neg_sample_size", default=50, type=int, help="Negative sample size, -1 to not use negative sampling"
)
parser.add_argument(
    "--init_size", default=1e-3, type=float, help="Initial embeddings' scale"
)
parser.add_argument(
    "--multi_c", action="store_true", help="Multiple curvatures per relation"
)
parser.add_argument(
    "--regularizer", choices=["N3", "F2"], default="N3", help="Regularizer"
)
parser.add_argument(
    "--reg", default=0, type=float, help="Regularization weight"
)
parser.add_argument(
    "--dropout", default=0, type=float, help="Dropout rate"
)
parser.add_argument(
    "--gamma", default=0, type=float, help="Margin for distance-based losses"
)
parser.add_argument(
    "--bias", default="constant", type=str, choices=["constant", "learn", "none"],
    help="Bias type (none for no bias)"
)
parser.add_argument(
    "--dtype", default="double", type=str, choices=["single", "double"], help="Machine precision"
)
parser.add_argument(
    "--double_neg", action="store_true",
    help="Whether to negative sample both head and tail entities"
)
parser.add_argument(
    "--debug", action="store_true",
    help="Only use 1000 examples for debugging"
)


def get_embeddings(args):
    # create model
    dataset_path = os.path.join(DATA_PATH, args.dataset)
    dataset = KGDataset(dataset_path, args.debug)
    args.sizes = dataset.get_shape()
    
    relation_type_index = dataset.get_relation_type_index()
    if "VecS" in args.model:
        model = getattr(models, args.model)(args, relation_type_index)
    else:
        model = getattr(models, args.model)(args)
    weights = torch.load(os.path.join("/media/dell/DATA/wy/code/CUKG/UrbanKG_Embedding_Model/logs_euluc/logs", args.model,"{}/experiment_0/model.pt".format(args.dataset)))
    entity_embeddings = weights['entity.weight'].detach().numpy()

    idx = pd.read_csv(DATA_PATH + '/' + args.dataset + "/entity_idx_embedding.csv", header=None)
    entity_idx = np.array(idx)

    entity_final_embedddings = np.zeros([entity_embeddings.shape[0], entity_embeddings.shape[1]])
    for i in range(entity_embeddings.shape[0]):

        entity_final_embedddings[int(entity_idx[i])] = entity_embeddings[i]

    return entity_final_embedddings

def get_relation_embeddings(args):
    # create model
    dataset_path = os.path.join(DATA_PATH, args.dataset)
    dataset = KGDataset(dataset_path, args.debug)
    args.sizes = dataset.get_shape()
    i

    weights = torch.load(os.path.join("/media/dell/DATA/wy/code/graph-learning/CUKG/UrbanKG_Embedding_Model/logs_no_euluc/", args.model,"{}/experiment_0/model.pt".format(args.dataset)))
    # weights = torch.load(os.path.join("/media/dell/DATA/wy/code/graph-learning/CUKG/UrbanKG_Embedding_Model/logs/", "VecS_2_neg_bn_50","{}/experiment_0/model.pt".format(args.dataset)))

    relation_embeddings = weights['rel.weight'].detach().numpy()
    
    # 取前一半
    relation_embeddings = relation_embeddings[0:relation_embeddings.shape[0]//2]
    
    idx = pd.read_csv(DATA_PATH + '/' + args.dataset + "/relations_idx_embeddings.csv", header=None)
    entity_idx = np.array(idx)

    entity_final_embedddings = np.zeros([relation_embeddings.shape[0], relation_embeddings.shape[1]])
    for i in range(relation_embeddings.shape[0]):

        entity_final_embedddings[int(entity_idx[i])] = relation_embeddings[i]

    return entity_final_embedddings

def get_region_embeddings(grid_KG_id_path, entity_final_embedddings, save_path):

    grid = pd.read_csv(grid_KG_id_path)
    grid_KG_id = grid[["region_id", "KG_id"]].values
    grid_embeddings = np.zeros([260, 32])

    for i in range(grid_embeddings.shape[0]):
        grid_embeddings[i] = entity_final_embedddings[int(grid_KG_id[i][1])]

    print(grid_embeddings)
    np.save(save_path, grid_embeddings)

def get_POI_embedding(grid_KG_id_path, entity_final_embedddings, save_path):
    poi = pd.read_csv(grid_KG_id_path)
    poi_KG_id = poi[["poi_id", "KG_id", "Region_id"]].values
    poi_embeddings = np.zeros([1600, 33])
    for i in range(poi_embeddings.shape[0]):
        poi_embeddings[i] [0:32] = entity_final_embedddings[int(poi_KG_id[i][1])]
        poi_embeddings[i] [32] =  int(poi_KG_id[i][2])

    print(poi_embeddings)
    np.save(save_path, poi_embeddings)

def get_Road_embedding(grid_KG_id_path, entity_final_embedddings, save_path):
    road = pd.read_csv(grid_KG_id_path)
    road_KG_id = road[["road_id", "KG_id", "Region_id"]].values
    road_embeddings = np.zeros([2500, 33])
    for i in range(road_embeddings.shape[0]):
        road_embeddings[i] [0:32] = entity_final_embedddings[int(road_KG_id[i][1])]
        road_embeddings[i] [32] =  int(road_KG_id[i][2])

    print(road_embeddings)
    np.save(save_path, road_embeddings)

def read_entity2id(entity2id_path):
    entity2id = {}
    with open(entity2id_path, 'r') as f:
        for line in f:
            line = line.strip()
            entity, id = line.split('\t')
            entity2id[entity] = int(id)
    return entity2id
def trans_hex_to_mpl(color_list):
    for i in range(len(color_list)):
        color_list[i] = tuple(int(color_list[i][j:j+2], 16) for j in (0, 2, 4))
        color_list[i] = tuple([x/255 for x in color_list[i]])
    return color_list
args = parser.parse_args()

# 设置dataset

# args.model = "TransE"
# models_list = ["TransE", "MurE", "RotE", "RefE", "AttE", "RefH", "AttH", "GIE", "VecS_4"]
models_list = ["VecS_4"]
models_list = ["TransE", "GIE", "VecS_4"]
# dataset_list = ["LANZHOU", "YULIN", "SHANGHAI", "GUANGZHOU", "WUHAN"]
dataset_list = ["YULIN", "SHANGHAI", "GUANGZHOU", "WUHAN", "LANZHOU"]
# dataset_list = ["GUANGZHOU"]
for dataset in dataset_list:
# models_list = ["VecS_3"]
    for model in models_list:
        args.model = model
        args.dataset = dataset
        save_dir = "/media/dell/DATA/wy/code/CUKG/result_all/entity_{}".format(args.dataset.lower())
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # if os.path.exists(save_dir + '/{}_result.png'.format(args.model)):
        #     continue
        
        all_embedding = get_embeddings(args)

        entity2id = read_entity2id(DATA_PATH + '/' + args.dataset + "/entity2id.txt")

        # 获取矢量实体和语义实体
        semantic_entity = []
        vector_entity = []
        for i in entity2id.keys():
            if len(i.split('/')) == 3:
                vector_entity.append(i)
            else:
                semantic_entity.append(i)

        # 随机抽取100个矢量实体和语义实体
        import random
        sample_num = 200
        if len(vector_entity) > sample_num:
            vector_entity = random.sample(vector_entity, sample_num)
        else:
            vector_entity = vector_entity
        if len(semantic_entity) > sample_num:
            semantic_entity = random.sample(semantic_entity, sample_num)
        else:
            semantic_entity = semantic_entity
            
        unit_entity = []
        block_entity = []
        cell_entity = []
        area_entity = []
        osm_entity = []
        poi_entity = []
        for i in vector_entity:
            if i.split('/')[1] == 'poi':
                poi_entity.append(i)
            elif i.split('/')[1] == 'unit':
                unit_entity.append(i)
            elif i.split('/')[1] == 'block':
                block_entity.append(i)
            elif i.split('/')[1] == 'cell':
                cell_entity.append(i)
            elif i.split('/')[1] == 'area':
                area_entity.append(i)
            elif i.split('/')[1] == 'osm':
                osm_entity.append(i)
        print(len(poi_entity), len(unit_entity), len(block_entity), len(cell_entity), len(area_entity), len(osm_entity))
        
        eulucclass_entity = []
        fineclass_entity = []
        middleclass_entity = []
        coarseclass = []
        cellclass = []
        osmclass = []
        for i in semantic_entity:
            if i.split('/')[0] == 'eulucclass':
                eulucclass_entity.append(i)
            elif i.split('/')[0] == 'fineclass':
                fineclass_entity.append(i)
            elif i.split('/')[0] == 'middleclass':
                middleclass_entity.append(i)
            elif i.split('/')[0] == 'coarseclass':
                coarseclass.append(i)
            elif i.split('/')[0] == 'cellclass':
                cellclass.append(i)
            elif i.split('/')[0] == 'osmclass':
                osmclass.append(i)
        print(len(eulucclass_entity), len(fineclass_entity), len(middleclass_entity), len(coarseclass), len(cellclass), len(osmclass))

        # vector_entity_embedding = all_embedding[[entity2id[i] for i in vector_entity]]
        # semantic_entity_embedding = all_embedding[[entity2id[i] for i in semantic_entity]]
        unit_entity_embedding = all_embedding[[entity2id[i] for i in unit_entity]]
        block_entity_embedding = all_embedding[[entity2id[i] for i in block_entity]]
        cell_entity_embedding = all_embedding[[entity2id[i] for i in cell_entity]]
        area_entity_embedding = all_embedding[[entity2id[i] for i in area_entity]]
        osm_entity_embedding = all_embedding[[entity2id[i] for i in osm_entity]]
        poi_entity_embedding = all_embedding[[entity2id[i] for i in poi_entity]]
        eulucclass_entity_embedding = all_embedding[[entity2id[i] for i in eulucclass_entity]]
        fineclass_entity_embedding = all_embedding[[entity2id[i] for i in fineclass_entity]]
        middleclass_entity_embedding = all_embedding[[entity2id[i] for i in middleclass_entity]]
        coarseclass_entity_embedding = all_embedding[[entity2id[i] for i in coarseclass]]
        cellclass_entity_embedding = all_embedding[[entity2id[i] for i in cellclass]]
        osmclass_entity_embedding = all_embedding[[entity2id[i] for i in osmclass]]
        

        # 进行降维绘图
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        import matplotlib
        import matplotlib.font_manager as fm
        import matplotlib.cm as cm
        import matplotlib as mpl
        import seaborn as sns
        import pandas as pd
        import numpy as np
        import os

        # 设置字体 为 Atial
        myfont = fm.FontProperties(fname='/usr/share/fonts/truetype/arphic/ukai.ttc')
        sns.set(font=myfont.get_name())
        sns.set_style("whitegrid",{"font.sans-serif":['simhei', 'Arial']})

        # 设置颜色,区分矢量实体和语义实体，按红蓝色渐变
        # red = cm.get_cmap('Reds', 12)
        # blue = cm.get_cmap('Blues', 12)
        
        semantic_color_list = [
            "#50632a", "#789440", "#9dbb61", "#c4d6a0", "#d7e3bf", "#ebf1df"
        ]
        spatial_color_list = [
            "#9c4a09", "#ea700d", "#f59d56", "#f9c499", "#fbd7bb", "#fcebdd"
        ]
        # semantic_color_list = trans_hex_to_mpl(semantic_color_list)
        # spatial_color_list = trans_hex_to_mpl(spatial_color_list)
        
        
        # 设置颜色，两个不同色系
        color = cm.rainbow(np.linspace(0, 1, 12))

        # 图例label字号设置
        # plt.rcParams['legend.fontsize'] = 20
        # plt.rcParams['axes.labelsize'] = 20
        
        all_entity_embedding = np.concatenate((unit_entity_embedding, block_entity_embedding, cell_entity_embedding, area_entity_embedding, osm_entity_embedding, poi_entity_embedding, eulucclass_entity_embedding, fineclass_entity_embedding, middleclass_entity_embedding, coarseclass_entity_embedding, cellclass_entity_embedding, osmclass_entity_embedding), axis=0)
        # 降维
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        tsne.fit(all_entity_embedding)
        tsne_dim2_embeddings = tsne.embedding_
        unit_entity_embedding = tsne_dim2_embeddings[0:len(unit_entity_embedding)]
        block_entity_embedding = tsne_dim2_embeddings[len(unit_entity_embedding):len(unit_entity_embedding)+len(block_entity_embedding)]
        cell_entity_embedding = tsne_dim2_embeddings[len(unit_entity_embedding)+len(block_entity_embedding):len(unit_entity_embedding)+len(block_entity_embedding)+len(cell_entity_embedding)]
        area_entity_embedding = tsne_dim2_embeddings[len(unit_entity_embedding)+len(block_entity_embedding)+len(cell_entity_embedding):len(unit_entity_embedding)+len(block_entity_embedding)+len(cell_entity_embedding)+len(area_entity_embedding)]
        osm_entity_embedding = tsne_dim2_embeddings[len(unit_entity_embedding)+len(block_entity_embedding)+len(cell_entity_embedding)+len(area_entity_embedding):len(unit_entity_embedding)+len(block_entity_embedding)+len(cell_entity_embedding)+len(area_entity_embedding)+len(osm_entity_embedding)]
        poi_entity_embedding = tsne_dim2_embeddings[len(unit_entity_embedding)+len(block_entity_embedding)+len(cell_entity_embedding)+len(area_entity_embedding)+len(osm_entity_embedding):len(unit_entity_embedding)+len(block_entity_embedding)+len(cell_entity_embedding)+len(area_entity_embedding)+len(osm_entity_embedding)+len(poi_entity_embedding)]
        eulucclass_entity_embedding = tsne_dim2_embeddings[len(unit_entity_embedding)+len(block_entity_embedding)+len(cell_entity_embedding)+len(area_entity_embedding)+len(osm_entity_embedding)+len(poi_entity_embedding):len(unit_entity_embedding)+len(block_entity_embedding)+len(cell_entity_embedding)+len(area_entity_embedding)+len(osm_entity_embedding)+len(poi_entity_embedding)+len(eulucclass_entity_embedding)]
        fineclass_entity_embedding = tsne_dim2_embeddings[len(unit_entity_embedding)+len(block_entity_embedding)+len(cell_entity_embedding)+len(area_entity_embedding)+len(osm_entity_embedding)+len(poi_entity_embedding)+len(eulucclass_entity_embedding):len(unit_entity_embedding)+len(block_entity_embedding)+len(cell_entity_embedding)+len(area_entity_embedding)+len(osm_entity_embedding)+len(poi_entity_embedding)+len(eulucclass_entity_embedding)+len(fineclass_entity_embedding)]
        middleclass_entity_embedding = tsne_dim2_embeddings[len(unit_entity_embedding)+len(block_entity_embedding)+len(cell_entity_embedding)+len(area_entity_embedding)+len(osm_entity_embedding)+len(poi_entity_embedding)+len(eulucclass_entity_embedding)+len(fineclass_entity_embedding):len(unit_entity_embedding)+len(block_entity_embedding)+len(cell_entity_embedding)+len(area_entity_embedding)+len(osm_entity_embedding)+len(poi_entity_embedding)+len(eulucclass_entity_embedding)+len(fineclass_entity_embedding)+len(middleclass_entity_embedding)]
        coarseclass_entity_embedding = tsne_dim2_embeddings[len(unit_entity_embedding)+len(block_entity_embedding)+len(cell_entity_embedding)+len(area_entity_embedding)+len(osm_entity_embedding)+len(poi_entity_embedding)+len(eulucclass_entity_embedding)+len(fineclass_entity_embedding)+len(middleclass_entity_embedding):len(unit_entity_embedding)+len(block_entity_embedding)+len(cell_entity_embedding)+len(area_entity_embedding)+len(osm_entity_embedding)+len(poi_entity_embedding)+len(eulucclass_entity_embedding)+len(fineclass_entity_embedding)+len(middleclass_entity_embedding)+len(coarseclass_entity_embedding)]
        cellclass_entity_embedding = tsne_dim2_embeddings[len(unit_entity_embedding)+len(block_entity_embedding)+len(cell_entity_embedding)+len(area_entity_embedding)+len(osm_entity_embedding)+len(poi_entity_embedding)+len(eulucclass_entity_embedding)+len(fineclass_entity_embedding)+len(middleclass_entity_embedding)+len(coarseclass_entity_embedding):len(unit_entity_embedding)+len(block_entity_embedding)+len(cell_entity_embedding)+len(area_entity_embedding)+len(osm_entity_embedding)+len(poi_entity_embedding)+len(eulucclass_entity_embedding)+len(fineclass_entity_embedding)+len(middleclass_entity_embedding)+len(coarseclass_entity_embedding)+len(cellclass_entity_embedding)]
        osmclass_entity_embedding = tsne_dim2_embeddings[len(unit_entity_embedding)+len(block_entity_embedding)+len(cell_entity_embedding)+len(area_entity_embedding)+len(osm_entity_embedding)+len(poi_entity_embedding)+len(eulucclass_entity_embedding)+len(fineclass_entity_embedding)+len(middleclass_entity_embedding)+len(coarseclass_entity_embedding)+len(cellclass_entity_embedding):len(unit_entity_embedding)+len(block_entity_embedding)+len(cell_entity_embedding)+len(area_entity_embedding)+len(osm_entity_embedding)+len(poi_entity_embedding)+len(eulucclass_entity_embedding)+len(fineclass_entity_embedding)+len(middleclass_entity_embedding)+len(coarseclass_entity_embedding)+len(cellclass_entity_embedding)+len(osmclass_entity_embedding)]

        # 绘图
        plt.figure(figsize=(10, 10))
        
        s_size = 200
        # 透明度
        alpha = 0.5
        plt.scatter(unit_entity_embedding[:, 0], unit_entity_embedding[:, 1], c=spatial_color_list[4], label='Parcel', s=s_size, alpha=alpha)
        plt.scatter(block_entity_embedding[:, 0], block_entity_embedding[:, 1], c=spatial_color_list[3], label='Block', s=s_size, alpha=alpha)
        plt.scatter(cell_entity_embedding[:, 0], cell_entity_embedding[:, 1], c=spatial_color_list[5], label='Grid', s=s_size, alpha=alpha)
        plt.scatter(area_entity_embedding[:, 0], area_entity_embedding[:, 1], c=spatial_color_list[1], label='ROI', s=s_size, alpha=alpha)
        plt.scatter(osm_entity_embedding[:, 0], osm_entity_embedding[:, 1], c=spatial_color_list[2], label='AOI', s=s_size, alpha=alpha)
        plt.scatter(poi_entity_embedding[:, 0], poi_entity_embedding[:, 1], c=spatial_color_list[0], label='POI', s=s_size, alpha=alpha)
        plt.scatter(eulucclass_entity_embedding[:, 0], eulucclass_entity_embedding[:, 1], c=semantic_color_list[2], label='Land-use class', s=s_size, alpha=alpha)
        plt.scatter(fineclass_entity_embedding[:, 0], fineclass_entity_embedding[:, 1], c=semantic_color_list[3], label='POI class(f)', s=s_size, alpha=alpha)
        plt.scatter(middleclass_entity_embedding[:, 0], middleclass_entity_embedding[:, 1], c=semantic_color_list[4], label='POI class(m)', s=s_size, alpha=alpha)
        plt.scatter(coarseclass_entity_embedding[:, 0], coarseclass_entity_embedding[:, 1], c=semantic_color_list[5], label='POI class(c)', s=s_size, alpha=alpha)
        plt.scatter(cellclass_entity_embedding[:, 0], cellclass_entity_embedding[:, 1], c=semantic_color_list[1], label='Grid class', s=s_size, alpha=alpha)
        plt.scatter(osmclass_entity_embedding[:, 0], osmclass_entity_embedding[:, 1], c=semantic_color_list[0], label='AOI class', s=s_size, alpha=alpha)
        
        # plt.scatter(unit_entity_embedding[:, 0], unit_entity_embedding[:, 1], c=color[0], label='unit')
        # plt.scatter(block_entity_embedding[:, 0], block_entity_embedding[:, 1], c=color[1], label='block')
        # plt.scatter(cell_entity_embedding[:, 0], cell_entity_embedding[:, 1], c=color[2], label='cell')
        # plt.scatter(area_entity_embedding[:, 0], area_entity_embedding[:, 1], c=color[3], label='area')
        # plt.scatter(osm_entity_embedding[:, 0], osm_entity_embedding[:, 1], c=color[4], label='osm')
        # plt.scatter(poi_entity_embedding[:, 0], poi_entity_embedding[:, 1], c=color[5], label='poi')
        # plt.scatter(osmclass_entity_embedding[:, 0], osmclass_entity_embedding[:, 1], c=color[6], label='osmclass')
        # plt.scatter(fineclass_entity_embedding[:, 0], fineclass_entity_embedding[:, 1], c=color[7], label='fineclass')
        # plt.scatter(middleclass_entity_embedding[:, 0], middleclass_entity_embedding[:, 1], c=color[8], label='middleclass')
        # plt.scatter(coarseclass_entity_embedding[:, 0], coarseclass_entity_embedding[:, 1], c=color[9], label='coarseclass')
        # plt.scatter(cellclass_entity_embedding[:, 0], cellclass_entity_embedding[:, 1], c=color[10], label='cellclass')
        # plt.scatter(eulucclass_entity_embedding[:, 0], eulucclass_entity_embedding[:, 1], c=color[11], label='eulucclass')
        
        # 根据实体设置坐标轴范围，固定单位长度
        # plt.xlim(-100, 100)
        # plt.ylim(-100, 100)
        # 设置横纵坐标字号大小
        plt.rcParams['font.size'] = 24
        
        # 隐藏刻度线
        plt.xticks([])
        plt.yticks([])
        
        # 绘制图例 右下, 并设置字号
        # plt.legend(loc='lower right', borderaxespad=0.5, fontsize=16)
        
        # 设置边框为黑色
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        
        
        plt.savefig(save_dir + '/{}_result.png'.format(args.model), dpi=300)
        # plt.show()


    # for model in models_list:
    #     args.model = model
    #     all_embedding = get_relation_embeddings(args)

    #     entity2id = read_entity2id(DATA_PATH + '/' + args.dataset + "/relation2id.txt")

    #     # 获取矢量实体和语义实体
    #     semantic_entity = []
    #     vector_entity = []
    #     vec_semant_entity = []
    #     for i in entity2id.keys():
    #         if "Similar" in i or "Belong" in i:
    #             semantic_entity.append(i)
    #         elif "Has" in i:
    #             vec_semant_entity.append(i)
    #         else:
    #             vector_entity.append(i)

    #     vector_entity_embedding = all_embedding[[entity2id[i] for i in vector_entity]]
    #     semantic_entity_embedding = all_embedding[[entity2id[i] for i in semantic_entity]]
    #     vec_semant_entity_embedding = all_embedding[[entity2id[i] for i in vec_semant_entity]]

    #     # 进行降维绘图
    #     from sklearn.manifold import TSNE
    #     import matplotlib.pyplot as plt
    #     import matplotlib
    #     import matplotlib.font_manager as fm
    #     import matplotlib.cm as cm
    #     import matplotlib as mpl
    #     import seaborn as sns
    #     import pandas as pd
    #     import numpy as np
    #     import os

    #     # 设置字体
    #     myfont = fm.FontProperties(fname='/usr/share/fonts/truetype/arphic/ukai.ttc')
    #     sns.set(font=myfont.get_name())
    #     sns.set_style("whitegrid",{"font.sans-serif":['simhei', 'Arial']})

    #     # 设置颜色
    #     color = cm.rainbow(np.linspace(0, 1, 3))

    #     # 图例label字号设置
    #     plt.rcParams['legend.fontsize'] = 20
    #     plt.rcParams['axes.labelsize'] = 20

    #     # 降维
    #     tsne = TSNE(n_components=2, init='pca', random_state=0, perplexity=3)
    #     tsne.fit(vector_entity_embedding)
    #     tsne_dim2_embeddings = tsne.embedding_
    #     tsne.fit(semantic_entity_embedding)
    #     tsne_dim2_embeddings_semantic = tsne.embedding_
    #     tsne.fit(vec_semant_entity_embedding)
    #     tsne_dim2_embeddings_vec_semant = tsne.embedding_

    #     # 绘图
    #     plt.figure(figsize=(10, 10))

    #     plt.scatter(tsne_dim2_embeddings[:, 0], tsne_dim2_embeddings[:, 1], c=color[0], label='空间关系')
    #     plt.scatter(tsne_dim2_embeddings_semantic[:, 0], tsne_dim2_embeddings_semantic[:, 1], c=color[1], label='语义关系')
    #     plt.scatter(tsne_dim2_embeddings_vec_semant[:, 0], tsne_dim2_embeddings_vec_semant[:, 1], c=color[2], label='类别关系')
    #     plt.legend(prop=myfont)
    #     plt.savefig('/media/dell/DATA/wy/code/graph-learning/CUKG/result/relation/{}_result_relation.png'.format(args.model), dpi=300)
        # plt.show()















