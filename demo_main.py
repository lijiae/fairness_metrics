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
from utils.penalty import *
from utils.getdata import *

from model.CausalMerge import FR_model
from model.AttributeNet import AttributeNet

def makeargs():
    parse=argparse.ArgumentParser()
    parse.add_argument('--image_dir',type=str,default="/media/lijia/DATA/lijia/data/vggface2/train_align")
    parse.add_argument('--maad_path',type=str,default='/media/lijia/DATA/lijia/data/vggface2/anno/maad_id.csv')
    parse.add_argument('--save_path',type=str,default='/home/lijia/codes/202302/lijia/face-recognition/checkpoints/ingroup')
    parse.add_argument('--train_csv',type=str,default='/media/lijia/DATA/lijia/data/vggface2/anno/train_id_sample.csv')
    parse.add_argument('--test_csv',type=str,default='/media/lijia/DATA/lijia/data/vggface2/anno/test_id_sample.csv')

    # training setting
    parse.add_argument('--batch_size',type=int,default=32)
    parse.add_argument('-lr',type=float,default=0.0001)
    parse.add_argument('--epoch',type=int,default=20)
    parse.add_argument('--warmup_step',type=int,default=0)

    # model setting
    parse.add_argument('--backbone_type',type=str,choices=['resnet50','senet'],default='resnet50')
    parse.add_argument('--idclass',type=int,default=8631)
    parse.add_argument('--ckpt_path',type=str,default='/home/lijia/codes/202302/lijia/face-recognition/checkpoints/classifier.pth.tar')

    args=parse.parse_args()
    return args

def loaddata(args):
    train_csv=pd.read_csv(args.train_csv)
    test_csv=pd.read_csv(args.test_csv)
    train_dataset=imagedataset(args.image_dir,train_csv)
    test_dataset=imagedataset(args.image_dir,test_csv)
    train_dl=DataLoader(train_dataset,args.batch_size,shuffle=True)
    test_dl=DataLoader(test_dataset,args.batch_size,shuffle=True)
    return train_dl,test_dl

def train(train_dl,fr_model,fac_model,optimizer,scheduler):
    loss=0
    losses=0
    mu=0.5
    interNum=200
    fr_model.train()

    device='cuda' if torch.cuda.is_available() else 'cpu'
    for i_bz,d in enumerate(train_dl):
        # scheduler.step()
        feature,y=fr_model(d[0].to(device))
        # y=fr_model.classifier(feature)

        loss1=XE(y,d[1].to(device))

        race_pre=torch.argmax(fac_model(d[0].to(device)),dim=1)
        loss2=InGroupPenalty(feature,race_pre,len(attrlist))

        loss=loss1+mu*loss2
        losses=losses+loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar('mu0.5/train_loss',loss,i_bz)
        writer.add_scalar('mu0.5/train_loss_xe',loss1,i_bz)
        writer.add_scalar('mu0.5/train_loss_ingrouploss',loss2,i_bz)

        if i_bz %interNum==0:
            pre_label=torch.argmax(y,dim=1)
            acc=(pre_label==d[1].to(device)).sum()/d[1].shape[0]
            writer.add_scalar('mu0.5/train_losses',losses,int(i_bz/interNum))
            losses=0
            print("batch:{}/total batch:{}  loss:{}  total_loss acc:{}".format(str(i_bz),len(train_dl),loss,acc))

def test(test_dl,fr_model,epoch):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    acc_total=0
    fr_model.eval()

    for data,label,_ in test_dl:
        _,y=fr_model(data.to(device))
        pre_label=torch.argmax(y,dim=1)
        acc_total+=(label.to(device)==pre_label).sum()

    writer.add_scalar('mu0.5_loss/test',acc_total/len(test_dl.dataset),epoch)
    print("test result: {}".format(acc_total/len(test_dl.dataset)))


attrlist=["Asian","Black","White"]
args=makeargs()
writer=torch.utils.tensorboard.SummaryWriter('./log')


fr_model=FR_model(args.idclass)
if os.path.exists(args.ckpt_path):
    fr_model=load_state_dict(fr_model,args.ckpt_path)
fr_model.to('cuda' if torch.cuda.is_available() else 'cpu')


fac_model=AttributeNet()
fac_model.set_idx_list(attrlist)
fac_model.to('cuda' if torch.cuda.is_available() else 'cpu')

# dataset
train_dl,test_dl=loaddata(args)

# optimizer & scheduler
optimizer = torch.optim.SGD(fr_model.parameters(), args.lr, momentum=0.9)
if args.warmup_step>1:
    scheduler = WarmUpLR(optimizer, args.warmup_step)
else:
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,len(train_dl),0.5)

# training
for i in range(5):
    print("start the {}th training:")
    train(train_dl,fr_model,fac_model,optimizer,scheduler)
    torch.save({'epoch': i, 'state_dict': fr_model.state_dict()},
               os.path.join(args.save_path, str(i) + '_ingroup_baseline.pth.tar'))
    test(test_dl,fr_model,i)
