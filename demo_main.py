import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm


from model.resnet50 import ResNet50
# import tensorboard as tx

import torch
from torch import nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.utils.tensorboard as tx
import os
# from data.imagedata import imagedataset
from utils.penalty import *
from utils.getdata import *

def makeargs():
    parse=argparse.ArgumentParser()
    parse.add_argument('--image_dir',type=str,default="//exdata/data/train_align")
    parse.add_argument('--maad_path',type=str,default='/exdata/data/maad_id.csv')
    parse.add_argument('--save_path',type=str,default='checkpoints')
    parse.add_argument('--train_csv',type=str,default='data/train_id_sample.csv')
    parse.add_argument('--test_csv',type=str,default='data/test_id_sample.csv')
    parse.add_argument('--batch_size',type=int,default=32)
    parse.add_argument('-lr',type=float,default=0.01)
    parse.add_argument('--epoch',type=int,default=20)
    parse.add_argument('--idclass',type=int,default=8631)
    args=parse.parse_args()
    return args

# def loaddata(args):
#     train_csv=pd.read_csv(args.train_csv)
#     test_csv=pd.read_csv(args.test_csv)
#     train_dataset=imagedataset(args.image_dir,train_csv)
#     test_dataset=imagedataset(args.image_dir,test_csv)
#     train_dl=DataLoader(train_dataset,args.batch_size)
#     test_dl=DataLoader(test_dataset,args.batch_size)
#     return train_dl,test_dl
def train(train_dl,fr_model,fac_model,device,optimizer,):
    loss=0
    timelosses=0
    for d in tqdm(train_dl):
        y=fr_model(d[0].to(device))
        loss1=XE(y,d[1].to(device))
        optimizer.zero_grad()

train_dl