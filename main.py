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
from data.imagedata import imagedataset


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

def loadimage(args):
    train_csv=pd.read_csv(args.train_csv)
    dataset=imagedataset(args.image_dir,train_csv)
    train_size=int(0.8*len(dataset))
    test_size=len(dataset)-train_size
    train_dataset,test_dataset=torch.utils.data.random_split(dataset,[train_size,test_size])
    train_dl=DataLoader(train_dataset,args.batch_size)
    test_dl=DataLoader(test_dataset,args.batch_size)
    return train_dl,test_dl

def loaddata(args):
    train_csv=pd.read_csv(args.train_csv)
    test_csv=pd.read_csv(args.test_csv)
    train_dataset=imagedataset(args.image_dir,train_csv)
    test_dataset=imagedataset(args.image_dir,test_csv)
    train_dl=DataLoader(train_dataset,args.batch_size)
    test_dl=DataLoader(test_dataset,args.batch_size)
    return train_dl,test_dl


def main():
    # 加载
    args=makeargs()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    writer=torch.utils.tensorboard.SummaryWriter('./log')

    # 读取数据
    # train_dl,test_dl=loadimage(args)
    train_dl,test_dl=loaddata(args)

    #读取模型
    model=ResNet50(args.idclass)
    model.load_state_dict(torch.load("checkpoints/2_cb.pth.tar")['state_dict'])
    optimizer=torch.optim.SGD(model.parameters(),args.lr,momentum=0.9)
    cel=nn.CrossEntropyLoss()
    model.to(device)

    bs=0
    for e in range(args.epoch):
        losses=0
        timelosses=0
        model.train()
        for d in tqdm(train_dl):
            bs += 1
            y=model(d[0].to(device))
            loss=cel(y,d[1].to(device))
            optimizer.zero_grad()
            losses=losses+loss
            timelosses+=loss
            loss.backward()
            optimizer.step()
            if bs%2000==0:
                writer.add_scalar('loss/train',timelosses,int(bs/2000))
                timelosses=0
        writer.add_scalar('loss/traine',losses,e)
        torch.save({'epoch': e, 'state_dict': model.state_dict()},
                   os.path.join(args.save_path, str(e) + '_fcbaseline.pth.tar'))
        total=0
        corr=0
        model.eval()
        for d in tqdm(test_dl):
            y=model(d[0].to(device))
            _,label=torch.max(y,1)
            total=total+label.size()[0]
            corr+=(d[1].to(device)==label).sum()
        print(float(corr)/float(total))
        writer.add_scalar('acc/test',float(corr)/float(total),e)

main()