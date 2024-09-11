import argparse
import json
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import torch.optim
import shutil
import models
import optimizers.regularizers as regularizers
from datasets.kg_dataset import KGDataset
from models import all_models
from optimizers.kg_optimizer import KGOptimizerEuluc
from utils.train import get_savedir, avg_both, format_metrics, count_params, avg_metrics
import numpy as np
import random
DATA_PATH = './data'

def set_random(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def train(args, experiment=0, trained=False):
    if trained:
        save_dir = get_savedir(args.model, args.dataset, experiment, make_new=False)
        dataset_path = os.path.join(DATA_PATH, args.dataset)
        dataset = KGDataset(dataset_path, args.debug)
        args.sizes = dataset.get_shape()
        
        # load data
        logging.info("\t " + str(dataset.get_shape()))
        train_examples = dataset.get_examples("train")
        valid_examples = dataset.get_examples("valid")
        test_examples = dataset.get_examples("test")
        predict_examples = dataset.get_examples("predict")
        filters = dataset.get_filters()
        euluc_examples = dataset.get_euluc_examples()
        
        idx2eulucclass = dataset.get_idx2eulucclass()
        idx2entity = dataset.get_idx2entity()
        idx2relation = dataset.get_idx2relation()
        
        relation_type_index = dataset.get_relation_type_index()
        
        if args.model == "GIE_euluc":
            model = getattr(models, 'GIE')(args)
        elif "VecS" in args.model:
            model = getattr(models, args.model)(args, relation_type_index)
        else:
            model = getattr(models, args.model)(args)
        total = count_params(model)
        logging.info("Total number of parameters {}".format(total))
        device = "cuda"
        model.to(device)
        model.load_state_dict(torch.load(os.path.join(save_dir, "model.pt")))
        
        model.cuda()
        model.eval()

        # Validation metrics
        valid_metrics = model.compute_metrics(valid_examples, filters)
        valid_euluc_metrics = model.compute_euluc_metrics(valid_examples, filters, idx2eulucclass)
        
        # Test metrics
        test_metrics = model.compute_metrics(test_examples, filters)
        test_euluc_metrics = model.compute_euluc_metrics(test_examples, filters, idx2eulucclass)

        # Predict results and metrics
        predict_result, result_index = model.get_predict_results(predict_examples, idx2eulucclass)
        reuslt_euluc = [idx2eulucclass[int(i)] for i in result_index]
        
        predict_result_path = os.path.join(save_dir, "predict_result.txt")
        with open(predict_result_path, "w") as f:
            for result in predict_result:
                result_score = result[3:]
                f.write(idx2entity[int(result[0])] + "\t" + idx2relation[int(result[1])] + "\t" + idx2entity[int(result[2])] + "\t" + str(reuslt_euluc) + "\t" + str(result_score).replace('\n', '') + "\n")

        # 重置 logging
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        return valid_metrics, valid_euluc_metrics, test_metrics, test_euluc_metrics, save_dir
    else:
        save_dir = get_savedir(args.model, args.dataset, experiment, make_new=False)
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        save_dir = get_savedir(args.model, args.dataset, experiment)
        # file logger
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%d %H:%M:%S",
            filename=os.path.join(save_dir, "train.log")
        )

        # stdout logger
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
        console.setFormatter(formatter)
        logging.getLogger("").addHandler(console)
        logging.info("Saving logs in: {}".format(save_dir))

        # create dataset
        dataset_path = os.path.join(DATA_PATH, args.dataset)
        dataset = KGDataset(dataset_path, args.debug)
        args.sizes = dataset.get_shape()

        # load data
        logging.info("\t " + str(dataset.get_shape()))
        train_examples = dataset.get_examples("train")
        valid_examples = dataset.get_examples("valid")
        test_examples = dataset.get_examples("test")
        predict_examples = dataset.get_examples("predict")
        filters = dataset.get_filters()
        euluc_examples = dataset.get_euluc_examples()
        
        idx2eulucclass = dataset.get_idx2eulucclass()
        idx2entity = dataset.get_idx2entity()
        idx2relation = dataset.get_idx2relation()
        
        relation_type_index = dataset.get_relation_type_index()
        # save config
        with open(os.path.join(save_dir, "config.json"), "w") as fjson:
            json.dump(vars(args), fjson)

        # create model
        if args.model == "GIE_euluc":
            model = getattr(models, 'GIE')(args)
        elif "VecS" in args.model:
            model = getattr(models, args.model)(args, relation_type_index)
        else:
            model = getattr(models, args.model)(args)
        total = count_params(model)
        logging.info("Total number of parameters {}".format(total))
        device = "cuda"
        model.to(device)

        # get optimizer
        regularizer = getattr(regularizers, args.regularizer)(args.reg)
        # show all the parameters of the model
        for name, param in model.named_parameters():
            print(name, param.size())
        optim_method = getattr(torch.optim, args.optimizer)(model.parameters(), lr=args.learning_rate)
        optimizer = KGOptimizerEuluc(model, regularizer, optim_method, args.batch_size, args.neg_sample_size, args.euluc_batch_size, args.euluc_neg_sample_size,
                                bool(args.double_neg), idx2eulucclass)
        counter = 0
        best_mrr = None
        best_epoch = None
        logging.info("\t Start training")
        for step in range(args.max_epochs):
            # Train step
            model.train()
            train_loss = optimizer.epoch(train_examples, euluc_examples)
            # train_loss = optimizer.epoch_no_euluc(train_examples)
            logging.info("\t Epoch {} | average train loss: {:.4f}".format(step, train_loss))

            # Valid step
            model.eval()
            valid_loss = optimizer.calculate_valid_loss(valid_examples)
            logging.info("\t Epoch {} | average valid loss: {:.4f}".format(step, valid_loss))

            if (step + 1) % args.valid == 0:
                valid_metrics = model.compute_metrics(valid_examples, filters)
                # euluc_valid_metrics = model.compute_euluc_metrics(valid_examples, filters, idx2eulucclass)
                logging.info(format_metrics(valid_metrics, split="valid"))
                # logging.info(format_metrics(euluc_valid_metrics, split="euluc valid"))
                
                test_metrics = model.compute_metrics(test_examples, filters)
                logging.info(format_metrics(test_metrics, split="test"))
                # test_euluc_metrics = model.compute_euluc_metrics(test_examples, filters, idx2eulucclass)
                # logging.info(format_metrics(test_euluc_metrics, split="euluc test"))

                # predict_result, result_index = model.get_predict_results(predict_examples, idx2eulucclass)
                            
                valid_mrr = valid_metrics["MRR"]
                if not best_mrr or valid_mrr > best_mrr:
                    best_mrr = valid_mrr
                    counter = 0
                    best_epoch = step
                    logging.info("\t Saving model at epoch {} in {}".format(step, save_dir))
                    torch.save(model.cpu().state_dict(), os.path.join(save_dir, "model.pt"))
                    model.cuda()
                else:
                    counter += 1
                    if counter == args.patience:
                        logging.info("\t Early stopping")
                        break
                    elif counter == args.patience // 2:
                        pass

        logging.info("\t Optimization finished")
        if not best_mrr:
            torch.save(model.cpu().state_dict(), os.path.join(save_dir, "model.pt"))
        else:
            logging.info("\t Loading best model saved at epoch {}".format(best_epoch))
            model.load_state_dict(torch.load(os.path.join(save_dir, "model.pt")))
        model.cuda()
        model.eval()
        valid_euluc_metrics = None
        test_euluc_metrics = None
        # Validation metrics
        valid_metrics = model.compute_metrics(valid_examples, filters)
        logging.info(format_metrics(valid_metrics, split="valid"))
        # valid_euluc_metrics = model.compute_euluc_metrics(valid_examples, filters, idx2eulucclass)
        # logging.info(format_metrics(valid_euluc_metrics, split="euluc valid"))
        
        # Test metrics
        test_metrics = model.compute_metrics(test_examples, filters)
        logging.info(format_metrics(test_metrics, split="test"))
        # test_euluc_metrics = model.compute_euluc_metrics(test_examples, filters, idx2eulucclass)
        # logging.info(format_metrics(test_euluc_metrics, split="euluc test"))

        # Predict results and metrics
        # predict_result, result_index = model.get_predict_results(predict_examples, idx2eulucclass)
        # reuslt_euluc = [idx2eulucclass[int(i)] for i in result_index]
        
        # predict_result_path = os.path.join(save_dir, "predict_result.txt")
        # with open(predict_result_path, "w") as f:
        #     for result in predict_result:
        #         result_score = result[3:]
        #         f.write(idx2entity[int(result[0])] + "\t" + idx2relation[int(result[1])] + "\t" + idx2entity[int(result[2])] + "\t" + str(reuslt_euluc) + "\t" + str(result_score).replace('\n', '') + "\n")
        if valid_euluc_metrics is None:
            valid_euluc_metrics = {
            'MRR': 0,
            'hits@[1,3,10]': [0, 0, 0],
        }
        if test_euluc_metrics is None:
            test_euluc_metrics = {
            'MRR': 0,
            'hits@[1,3,10]': [0, 0, 0],
        }
        
        # 重置 logging
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        return valid_metrics, valid_euluc_metrics, test_metrics, test_euluc_metrics, save_dir
        # return valid_metrics, test_metrics, save_dir

parser = argparse.ArgumentParser(
    description="Urban Knowledge Graph Embedding"
)
parser.add_argument(
    "--dataset", default="WUHAN", choices=["NYC", "CHI", "YULIN", "WUHAN", "SHANGHAI", "GUANGZHOU", "LANZHOU", "WUDI"],
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
    "--euluc_neg_sample_size", default=50, type=int, help="Negative sample size, -1 to not use negative sampling"
)
parser.add_argument(
    "--init_size", default=1e-3, type=float, help="Initial embeddings' scale"
)
parser.add_argument(
    "--multi_c", action="store_true", help="Multiple curvatures per relation"
)
parser.add_argument(
    "--regularizer", choices=["N3", "F2", "Euluc", "hyperbolic_distance"], default="F2", help="Regularizer"
)
parser.add_argument(
    "--reg", default=100, type=float, help="Regularization weight"
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
    rand_seed = random.randint(0, 1000000)
    set_random(rand_seed)
    # args = parser.parse_args()
    exp_times = 5
    # for model in ["VecS", "TransE", "MurE", "CP", "RotE", "RefE", "AttE", "RotH", "RefH", "AttH", "ComplEx", "RotatE", "GIE"]:
    for model in ["VecS"]:
        # for dataset in ["YULIN", "LANZHOU", "WUHAN", "SHANGHAI",  "GUANGZHOU"]:
        for dataset in ["WUDI"]:
            args = parser.parse_args(["--dataset", dataset, "--multi_c", "--model", model])
            
            save_dir = get_savedir(args.model, args.dataset, 0, make_new=False)
            save_dir = os.path.dirname(save_dir)
            metrics_path = os.path.join(save_dir, "metrics.txt")
            # if os.path.exists(metrics_path):
            #     continue
            
            if model in ["ComplEx", "RotatE"]:
                args.optimizer = "SparseAdam"
            metrics_list = []
            euluc_metrics_list = []
            save_dir_list = []
            test_metrics_list = []
            test_euluc_metrics_list = []
            for exp in range(exp_times):
                trained = False
                save_dir = get_savedir(args.model, args.dataset, exp, make_new=False)
                train_log = os.path.join(save_dir, "train.log")
                print(train_log)
                if os.path.exists(train_log):
                    with open(train_log, "r") as f:
                        lines = f.readlines()
                        print(lines[-4])
                        if "finished" in lines[-4]:
                            trained = True
                metrics, euluc_metrics, test_metrics, test_euluc_metrics, save_dir = train(args, exp, trained)
                metrics_list.append(metrics)
                euluc_metrics_list.append(euluc_metrics)
                save_dir_list.append(save_dir)
                test_metrics_list.append(test_metrics)
                test_euluc_metrics_list.append(test_euluc_metrics)
            # get metrics mean and std
            metrics = avg_metrics(metrics_list)
            euluc_metrics = avg_metrics(euluc_metrics_list)

            test_metrics = avg_metrics(test_metrics_list)
            test_euluc_metrics = avg_metrics(test_euluc_metrics_list)
            # save metrics save dir在上级目录
            save_dir = os.path.dirname(save_dir_list[0])
            metrics_path = os.path.join(save_dir, "metrics.txt")
            with open(metrics_path, "w") as f:
                f.write("metrics:\n")
                for key, value in metrics.items():
                    f.write(key + ": " + str(value) + "\n")
                f.write("euluc_metrics:\n")
                for key, value in euluc_metrics.items():
                    f.write(key + ": " + str(value) + "\n")
                f.write("test_metrics:\n")
                for key, value in test_metrics.items():
                    f.write(key + ": " + str(value) + "\n")
                f.write("test_euluc_metrics:\n")
                for key, value in test_euluc_metrics.items():
                    f.write(key + ": " + str(value) + "\n")
            # save best model
            best_index = np.argmax([m['hits@[1,3,10]'][0] for m in metrics_list])
            ori_save_dir = save_dir_list[best_index]
            ori_model_path = os.path.join(ori_save_dir, "model.pt")
            best_model_path = os.path.join(save_dir, "model.pt")
            os.system("cp {} {}".format(ori_model_path, best_model_path))
            # save best result
            ori_best_result_path = os.path.join(ori_save_dir, "predict_result.txt")
            best_result_path = os.path.join(save_dir, "predict_result.txt")

            os.system("cp {} {}".format(ori_best_result_path, best_result_path))
