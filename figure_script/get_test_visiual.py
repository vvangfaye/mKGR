import argparse
import json
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import torch.optim
import models
import optimizers.regularizers as regularizers
from datasets.kg_dataset import KGDataset
from models import all_models
from optimizers.kg_optimizer import KGOptimizerEuluc
from utils.train import get_savedir, avg_both, format_metrics, count_params, avg_metrics
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.cm as cm
import seaborn as sns
myfont = fm.FontProperties(fname='/usr/share/fonts/truetype/arphic/ukai.ttc')
sns.set(font=myfont.get_name())
sns.set_style("whitegrid",{"font.sans-serif":['simhei', 'Arial']})
DATA_PATH = './data_euluc'
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
label_eng_map = {
    '101': "Residential",
    '201': "Business office",
    '202': "Commercial service",
    '301': "Industrial",
    '402': "Transportation stations",
    '403': "Airport facilities",
    '501': "Administrative",
    '502': "Educational",
    '503': "Medical",
    '504': "Sport and cultural",
    '505': "Park and greenspace",
    '800': "Water",
    '401': "Road",
}   
def set_random():
    torch.manual_seed(8923)
    torch.cuda.manual_seed(8923)
    np.random.seed(8923)

def get_test_vis(args, experiment=0):

    # create dataset
    dataset_path = os.path.join(DATA_PATH, args.dataset)
    dataset = KGDataset(dataset_path, args.debug)
    args.sizes = dataset.get_shape()

    test_examples = dataset.get_examples("test")
    val_examples = dataset.get_examples("valid")
    
    relation_type_index = dataset.get_relation_type_index()
    idx2eulucclass = dataset.get_idx2eulucclass()
    idx2entity = dataset.get_idx2entity()
    fliters = dataset.get_filters()

    # create model
    if args.model == "GIE_euluc":
        model = getattr(models, 'GIE')(args)
    elif "VecS" in args.model:
        model = getattr(models, args.model)(args, relation_type_index)
    else:
        model = getattr(models, args.model)(args)
    device = "cuda"
    model.to(device)
    # load model
    model.load_state_dict(torch.load(args.best_model_path))
    results, val_embedding, wrong_examples, wrong_label = model.compute_embedding(val_examples, idx2eulucclass)
    
    wrong_examples = wrong_examples.cpu().numpy()
    wrong_examples = np.concatenate((wrong_examples, wrong_label.reshape(-1, 1)), axis=1)
    # 使用idx2eulucclass映射
    wrong_examples = np.vectorize(idx2entity.get)(wrong_examples)
    
    # 写入文件
    # with open("./test_vis/{}_{}_wrong_examples.txt".format(args.dataset, args.model), 'w') as f:
    #     for line in wrong_examples:
    #         f.write(" ".join(line) + "\n")
    
    category_embedding = {}
    for key, value in idx2eulucclass.items():
        result_index = np.where(val_examples[:, 2] == key)[0]
        category_embedding[value] = val_embedding[result_index]
    
    return category_embedding


parser = argparse.ArgumentParser(
    description="Urban Knowledge Graph Embedding"
)
parser.add_argument(
    "--dataset", default="WUHAN", choices=["NYC", "CHI", "YULIN", "WUHAN", "SHANGHAI", "GUANGZHOU", "LANZHOU"],
    help="Urban Knowledge Graph dataset"
)
# Trans
parser.add_argument(
    "--model", default="TransE", choices=all_models, help='Model name'
)
parser.add_argument(
    "--optimizer", choices=["Adagrad", "Adam", "SparseAdam"], default="Adam",
    help="Optimizer"
)
parser.add_argument(
    "--max_epochs", default=150, type=int, help="Maximum number of epochs to train for"
)
parser.add_argument(
    "--patience", default=20, type=int, help="Number of epochs before early stopping"
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
    "--euluc_batch_size", default=500, type=int, help="Batch size"
)
parser.add_argument(
    "--euluc_neg_sample_size", default=3, type=int, help="Negative sample size, -1 to not use negative sampling"
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
parser.add_argument(
    "--best_model_path", default="/home/faye/code/UUKG/UrbanKG_Embedding_Model/logs/12_30/CHI/GIE_00_31_27/model.pt", type=str, help="The best model path"
)

if __name__ == "__main__":
    set_random()
    # args = parser.parse_args()
    save_dir = "./test_vis"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    exp_times = 1
    # for model in ["VecS_4", "TransE", "GIE", "MurE", "RotE", "RefE", "AttE", "RotH", "RefH", "AttH", "ComplEx"]:
    for model in ["VecS_4", "GIE", "TransE"]:
        for dataset in ["GUANGZHOU", "WUHAN", "SHANGHAI", "LANZHOU", "YULIN"]:
        # for dataset in ["SHANGHAI"]:
            args = parser.parse_args(["--dataset", dataset, "--multi_c", "--model", model])
            args.best_model_path = "/media/dell/DATA/wy/code/CUKG/UrbanKG_Embedding_Model/logs_euluc/logs_normal/{}/{}/experiment_0/model.pt".format(model, dataset)
            # args.best_model_path = "/media/dell/DATA/wy/code/graph-learning/CUKG/UrbanKG_Embedding_Model/logs/VecS_2/{}/experiment_0/model.pt".format(dataset)
            if model in ["ComplEx", "RotatE"]:
                args.optimizer = "SparseAdam"
            metrics_list = []
            euluc_metrics_list = []
            save_dir_list = []
            test_metrics_list = []
            test_euluc_metrics_list = []
            
            try:
                category_embeddings = get_test_vis(args)
                
                # 合并category_embeddings中的tensor
                all_category_embeddings = []
                for key, tensor in category_embeddings.items():
                    all_category_embeddings.append(tensor.cpu().numpy())
                    
                all_category_embeddings = np.concatenate(all_category_embeddings, axis=0)
            
                tsne = TSNE(n_components=2, init='pca', random_state=0)
                tsne.fit(all_category_embeddings)
                embeddings = tsne.embedding_
                
                # 按keys绘图
                keys = list(category_embeddings.keys())
                keys = sorted(keys)
                
                
                myfont = fm.FontProperties(fname='/usr/share/fonts/truetype/arphic/ukai.ttc')
                sns.set(font=myfont.get_name())
                sns.set_style("whitegrid",{"font.sans-serif":['simhei', 'Arial']})
                
                # corlor = cm.rainbow(np.linspace(0, 1, len(label_eng_map.keys())))
                corlor_map = {
                    '101': '#fcecb6',
                    '201': '#f4c1bf',
                    '202': '#eb8682',
                    '301': '#e1c0fa',
                    '402': '#eaebdc',
                    '403': '#cccccc',
                    '501': '#96d5c4',
                    '502': '#bdd2ff',
                    '503': '#73fedf',
                    '504': '#87b0f2',
                    '505': '#cdedab',
                }
                # 绘图并保存
                plt.figure(figsize=(10, 10))
                start = 0
                for i, key in enumerate(keys):
                    label = label_eng_map[key.split("/")[1]]
                    corlor_index = key.split("/")[1]
                    if corlor_index not in corlor_map.keys():
                        continue    
                    
                    plt.scatter(embeddings[start:start+len(category_embeddings[key]), 0], embeddings[start:start+len(category_embeddings[key]), 1], c=corlor_map[corlor_index], label=label, alpha=0.5, s=200)
                    start += len(category_embeddings[key])
                
                # 隐藏坐标轴
                plt.xticks([])
                plt.yticks([])
                
                # 隐藏框线
                plt.box(False)
                # plt.legend(prop=myfont, loc='upper right')
                plt.savefig('./test_vis/{}_{}.png'.format(dataset, model))

            except Exception as e:
                print(e, dataset, model)
                
            