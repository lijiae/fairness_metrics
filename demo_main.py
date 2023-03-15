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
# from utils.gradcam_batch import GradFeature

from model.CausalMerge import FR_model,FR_model_classifier,FR_model_backbone
from model.AttributeNet import AttributeNet

def makeargs():
    parse=argparse.ArgumentParser()
    parse.add_argument('--image_dir',type=str,default="/media/lijia/DATA/lijia/data/vggface2/train_align")
    parse.add_argument('--maad_path',type=str,default='/media/lijia/DATA/lijia/data/vggface2/anno/maad_id.csv')
    parse.add_argument('--save_path',type=str,default='/home/lijia/codes/202302/lijia/face-recognition/checkpoints/ingroup')
    parse.add_argument('--train_csv',type=str,default='/media/lijia/DATA/lijia/data/vggface2/anno/train_id_sample.csv')
    parse.add_argument('--test_csv',type=str,default='/media/lijia/DATA/lijia/data/vggface2/anno/test_id_sample.csv')

    # training setting
    parse.add_argument('--batch_size',type=int,default=48)
    parse.add_argument('-lr',type=float,default=0.0001)
    parse.add_argument('--warmup_step',type=int,default=0)
    parse.add_argument('--epoch',type=int,default=50)
    parse.add_argument('--mu',type=float,default=0.5)
    parse.add_argument('--print_inter',type=int,default=200)
    parse.add_argument('--train_type',type=str,default='normal',choices=['causal','normal'])
    parse.add_argument('--ingroup_loss',type=bool,default=False)

    # model setting
    parse.add_argument('--backbone_type',type=str,choices=['resnet50','senet'],default='resnet50')
    parse.add_argument('--dataset',type=str,default="celeba",choices=["celeba","vggface2"])
    parse.add_argument('--idclass',type=int,default=10178)
    parse.add_argument('--ckpt_path',type=str,default='')
    parse.add_argument('--attr_net_path',type=str,default='/home/lijia/codes/202302/lijia/face-recognition/checkpoints/AttributeNet.pkl')


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

def train(train_dl,fr_model,optimizer,scheduler,fac_model=None):
    loss=0
    losses=0
    mu=1
    if isinstance(fr_model,list):
        for module in fr_model:
            module.train()
    else:
        fr_model.train()
    attrlen=len(attrlist)
    device='cuda' if torch.cuda.is_available() else 'cpu'

    if args.train_type == 'causal':
        for i_bz,d in enumerate(tqdm(train_dl)):
            feature=fr_model[0](d[0].to(device))
            pred_class_logits=fac_model(d[0].to(device))
            race_pre=torch.argmax(pred_class_logits,dim=1)
            get_feature_map=fac_model.get_feature_map()
            # cam is f,logit is c
            f_x_c=[]
            for i in range(attrlen):
                classid=torch.ones_like(race_pre)*i
                f_x_c.append(getGradCam(get_feature_map,pred_class_logits,classid))
            cams=torch.stack(f_x_c,0) # 将list中cam特征合并
            weights_race=torch.mul(cams,pred_class_logits.transpose(0,1).unsqueeze(2).unsqueeze(3)).sum(0) # 和logit加权
            feature=(torch.mul(weights_race.unsqueeze(1),feature)+feature)/2 # 更新新的feature

            y=fr_model[1](feature)

            if args.ingroup_loss:
                loss1=XE(y,d[1].to(device))
                loss2=InGroupPenalty(feature,race_pre,len(attrlist))
                loss=loss1+mu*loss2
                writer.add_scalar('mu0.5/train_loss_ingrouploss',loss2)
            else:
                loss1=XE(y,d[1].to(device))
                loss=loss1
            losses = losses + loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('mu0.5/train_loss',loss)
            writer.add_scalar('mu0.5/train_loss_xe',loss1)

            if i_bz %args.print_inter==0:
                pre_label=torch.argmax(y,dim=1)
                acc=(pre_label==d[1].to(device)).sum()/d[1].shape[0]
                writer.add_scalar('mu0.5/train_losses',losses)
                losses=0
                print("batch:{}/total batch:{}  loss:{}  total_loss acc:{}".format(str(i_bz),len(train_dl),loss,acc))
    elif args.train_type == 'normal':
        for i_bz,d in enumerate(tqdm(train_dl)):
            feature,y=fr_model(d[0].to(device))

            if args.ingroup_loss:
                pred_class_logits = fac_model(d[0].to(device))
                race_pre = torch.argmax(pred_class_logits, dim=1)
                loss1=XE(y,d[1].to(device))
                loss2=InGroupPenalty(feature,race_pre,len(attrlist))
                loss=loss1+mu*loss2
                writer.add_scalar('celeba-baseline/train_loss_ingrouploss',loss2)
            else:
                loss1=XE(y,d[1].to(device))
                loss=loss1
            losses = losses + loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('celeba-baseline/train_loss',loss)
            writer.add_scalar('celeba-baseline/train_loss_xe',loss1)

            if i_bz %args.print_inter==0:
                pre_label=torch.argmax(y,dim=1)
                acc=(pre_label==d[1].to(device)).sum()/d[1].shape[0]
                writer.add_scalar('celeba-baseline/train_losses',losses)
                losses=0
                print("batch:{}/total batch:{}  loss:{}  total_loss acc:{}".format(str(i_bz),len(train_dl),loss,acc))

def test(test_dl,fr_model,epoch):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    acc_total=0
    fr_model.eval()

    for data,label,_ in test_dl:
        if isinstance(fr_model,list):
            feature=fr_model[0](data.to(device))
            y=fr_model[1](feature)
        else:
            _,y=fr_model(data.to(device))
        pre_label=torch.argmax(y,dim=1)
        acc_total+=(label.to(device)==pre_label).sum()

    writer.add_scalar('mu0.5_loss/test',acc_total/len(test_dl.dataset),epoch)
    print("test result: {}".format(acc_total/len(test_dl.dataset)))


attrlist=["Asian","Black","White"]
args=makeargs()
writer=torch.utils.tensorboard.SummaryWriter('./log')
device='cuda' if torch.cuda.is_available() else 'cpu'

if args.train_type=='normal':
    fr_model=FR_model(args.idclass)
    if os.path.exists(args.ckpt_path):
        fr_model=load_state_dict(fr_model,args.ckpt_path)
    fr_model.to(device)
else:
    fr_model=[]
    fr_model.append(FR_model_backbone())
    fr_model.append(FR_model_classifier(args.idclass))
    if os.path.exists(args.ckpt_path):
        fr_model=load_state_dict(fr_model,args.ckpt_path)
    for module in fr_model:
        module.to(device)


if args.train_type=="normal" and not args.ingroup_loss:
    fac_model=None
    print("no use for face attributes classifier")
else:
    fac_model=AttributeNet(args.attr_net_path)
    fac_model.set_idx_list(attrlist)
    fac_model.to('cuda' if torch.cuda.is_available() else 'cpu')

# dataset
if args.dataset=="celeba":
    train_dl, test_dl = loaddata_celeba(args)
else:
    train_dl,test_dl=loaddata(args)

# optimizer & scheduler
if isinstance(fr_model,list):
    # optimizer = torch.optim.SGD(fr_model.parameters(), args.lr, momentum=0.9)
    optimizer=torch.optim.SGD(
        [{'params':fr_model[0].parameters(),},
         {'params':fr_model[1].parameters()}],
        lr=args.lr,
        momentum=0.9
    )
else :
    optimizer = torch.optim.SGD(fr_model.parameters(), args.lr, momentum=0.9)
if args.warmup_step>1:
    scheduler = WarmUpLR(optimizer, args.warmup_step)
else:
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,10,0.1)

# training
for i in range(args.epoch):
    print("start the {}th training:".format(str(i)))
    train(train_dl,fr_model,optimizer,scheduler,fac_model=None)
    if isinstance(fr_model,list):
        torch.save({'epoch': i, 'state_dict': fr_model[0].state_dict()},
               os.path.join(args.save_path, str(i) + '_causalnet_backbone.pth.tar'))
        torch.save({'epoch': i, 'state_dict': fr_model[1].state_dict()},
               os.path.join(args.save_path, str(i) + '_causalnet_classifier.pth.tar'))
    else:
        torch.save({'epoch': i, 'state_dict': fr_model.state_dict()},
               os.path.join(args.save_path, str(i) + '_causalnet.pth.tar'))
    # test(test_dl,fr_model,i)

