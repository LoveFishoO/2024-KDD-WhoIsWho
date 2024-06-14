import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import random
import json
import pickle
import logging
from torch.optim.lr_scheduler import _LRScheduler
from models import  GCNModel
from utils import *
from torch.backends import cudnn
# torch.autograd.set_detect_anomaly(True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def add_arguments(args):
    # essential paras eval_dir
    # args.add_argument('--train_dir', type=str, help="train_dir", default = "../out_data/e5_instruct_graph_train.pkl")
    # args.add_argument('--eval_dir', type=str, help="eval_dir", default = None)
    args.add_argument('--test_dir', type=str, help="test_dir", default = "../out_data/e5_instruct_graph_test.pkl")
    args.add_argument('--model_path', type=str, help="save_dir", default= "./graph_model/e5_instruct_gcn_model.pt")
    args.add_argument('--save_result_dir', type=str, help="save_dir", default= "../output/e5_instruct_gcn.json")
    args.add_argument('--log_name', type=str, help="log_name", default = "log")
    # training paras.
    # args.add_argument('--epochs', type=int, help="training #epochs", default=50)
    args.add_argument('--seed', type=int, help="seed", default=42)
    # args.add_argument('--lr', type=float, help="learning rate", default=5e-4)
    # args.add_argument('--min_lr', type=float, help="min lr", default=2e-4)
    # args.add_argument('--bs', type=int, help="batch size", default=1)
    # args.add_argument('--input_dim', type=int, help="input dimension", default=768)
    # args.add_argument('--output_dim', type=int, help="output dimension", default=768)
    # args.add_argument('--input_dim', type=int, help="input dimension", default=1024* 3) 
    # args.add_argument('--output_dim', type=int, help="output dimension", default=1024)
    args.add_argument('--verbose', type=int, help="eval", default=1)
    
    # dataset graph paras
    # args.add_argument('--usecoo', help="use co-organization edge", action='store_true')
    # args.add_argument('--usecov', help="use co-venue edge", action='store_true')
    args.add_argument('--usecoo', help="use co-organization edge", default=True)
    args.add_argument('--usecov', help="use co-venue edge", default=True)
    args.add_argument('--threshold', type=float, help="threshold of coo and cov", default=0.2)
    # args.add_argument('--threshold', type=float, help="threshold of coo and cov", default=0)
    
    args = args.parse_args()
    return args

device = torch.device('cpu')

args = argparse.ArgumentParser()
args = add_arguments(args)

if args.test_dir is not None:
    encoder = torch.load(f"{args.model_path}")
    encoder.eval()
    with open(args.test_dir, 'rb') as f:
        test_data = pickle.load(f)
    result = {}

    with torch.no_grad():
        for tmp_test in tqdm(test_data):

            each_sub, _ , author_id, pub_id  = tmp_test
            # each_sub = each_sub.cuda()
            each_sub = each_sub.to(device)
            
            node_outputs, adj_matrix, adj_weight, batch_item = each_sub.x, each_sub.edge_index, each_sub.edge_attr.squeeze(-1), each_sub.batch
            
            if args.threshold > 0:
                flag = adj_weight[:,1:]<args.threshold
                adj_weight[:,1:] = torch.where(flag,torch.tensor(0.0),adj_weight[:,1:])
            if args.usecoo and args.usecov:
                adj_weight = adj_weight.mean(dim = -1)
            elif args.usecoo:
                adj_weight = (adj_weight[:,0] + adj_weight[:,1])/2
            elif args.usecov:
                adj_weight = (adj_weight[:,0] + adj_weight[:,2])/2
            else:
                adj_weight = adj_weight[:,0]
            flag = torch.nonzero(adj_weight).squeeze(-1)
            adj_matrix = adj_matrix.T[flag].T
            adj_weight = adj_weight[flag]
            # edge_labels = edge_labels[flag]

            logit = encoder(node_outputs,adj_matrix)
            logit = logit.squeeze(-1)

            result[author_id] = {}
            for i in range(len(pub_id)):
                result[author_id][pub_id[i]]=logit[i].item()
    
    with open(args.save_result_dir, 'w') as f:
        json.dump(result, f)
    print(args.save_result_dir)