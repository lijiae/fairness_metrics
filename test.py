import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm


from model.resnet50 import ResNet50
import tensorboard as tx

import torch
from torch import nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
import os
from data.imagedata import imagedataset

def makeargs():
    parse=argparse.ArgumentParser()
    parse.add_argument('--image_dir',type=str,default="/exdata/data/train_align")
    parse.add_argument('--save_path',type=str,default='checkpoints')
    parse.add_argument('--train_csv',type=str,default='data/test_id_sample.csv')
    parse.add_argument('--batch_size',type=int,default=32)
    parse.add_argument('--idclass',type=int,default=8631)
    args=parse.parse_args()
    return args

def loadimage(args):
    train_csv=pd.read_csv(args.train_csv)
    dataset=imagedataset(args.image_dir,train_csv)
    train_size=int(0.8*len(dataset))
    test_size=len(dataset)-train_size
    train_dataset,test_dataset=torch.utils.data.random_split(dataset,[train_size,test_size])
    train_dl=DataLoader(train_dataset,args.batch_size)
    test_dl=DataLoader(test_dataset,args.batch_size)
    return train_dl,test_dl

def loadallimage(args):
    train_csv = pd.read_csv(args.train_csv)
    dataset = imagedataset(args.image_dir, train_csv)
    print("total data length is",len(dataset))
    dataloader=DataLoader(dataset,batch_size=32)
    return dataloader

def main():
    # 加载
    args=makeargs()
    device='cuda' if torch.cuda.is_available() else 'cpu'

    # 读取数据
    testdl=loadallimage(args)

    #读取模型
    model=ResNet50(args.idclass)
    model.load_state_dict(torch.load("checkpoints/classifier.pth.tar")['state_dict'])
    model.eval()
    model.to(device)

    result_pre=[]
    result_names=[]
    acc=0

    for d in tqdm(testdl):
        y=model(d[0].to(device))
        _,label=torch.max(y,1)
        acc+=(d[1].to(device)==label).sum()
        result_pre+=label.tolist()
        result_names+=list(d[2])

    print("accurate counts:",acc)
    data=pd.DataFrame({
        "Filename":result_names,
        "pre_id":result_pre
    })
    data.to_csv('data/test_pre_id.csv')

main()