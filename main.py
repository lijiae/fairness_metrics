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
from model.CIAM import CIAM


def makeargs():
    parse=argparse.ArgumentParser()
    parse.add_argument('--image_dir',type=str,default="/media/lijia/DATA/lijia/data/vggface2/train_align")
    parse.add_argument('--maad_path',type=str,default='/media/lijia/DATA/lijia/data/vggface2/anno/maad_id.csv')
    parse.add_argument('--save_path',type=str,default='/home/lijia/codes/202302/lijia/face-recognition/checkpoints/arcface/baseline')
    parse.add_argument('--train_csv',type=str,default='/media/lijia/DATA/lijia/data/vggface2/anno/train_id_sample_8615.csv')
    parse.add_argument('--test_csv',type=str,default='/media/lijia/DATA/lijia/data/vggface2/anno/test_id_sample_8615.csv')

    # training setting
    parse.add_argument('--batch_size',type=int,default=32)
    parse.add_argument('-lr',type=float,default=0.0001)
    parse.add_argument('--warmup_step',type=int,default=0)
    parse.add_argument('--epoch',type=int,default=5)
    parse.add_argument('--print_inter',type=int,default=200)
    parse.add_argument('--train_type',type=str,default='normal',choices=['causal','normal'])
    parse.add_argument('--ingroup_loss',type=bool,default=False)

    # model setting
    parse.add_argument('--backbone_type',type=str,choices=['resnet50','senet'],default='resnet50')
    parse.add_argument('--dataset',type=str,default="vggface2",choices=["celeba","vggface2"])
    parse.add_argument('--idclass',type=int,default=8615)
    parse.add_argument('--ckpt_path',type=str,default='/home/lijia/codes/202302/lijia/face-recognition/checkpoints/normal/0_vggface2_resnet.pth.tar')
    parse.add_argument('--ckpt_path_backbone',type=str,default='')
    parse.add_argument('--ckpt_path_classifier',type=str,default='')
    parse.add_argument('--attr_net_path',type=str,default='/home/lijia/codes/202302/lijia/face-recognition/checkpoints/AttributeNet.pkl')
    parse.add_argument('--metric_type',type=str,choices=['arcface','cosface','softmax'],default='arcface')

    # concept setting
    parse.add_argument('--concept_dir',type=str,default="/media/lijia/DATA/lijia/data/vggface2/average_face/gender")
    parse.add_argument('--cluster_num',type=int,default=5)
    parse.add_argument('--pre_proto',type=bool,default=False)
    parse.add_argument('--save_concept',type=str,default="/home/lijia/codes/202302/lijia/face-recognition/data/prototype/cluster_race/concept_A_B_W_feature.npy")
    parse.add_argument('--save_prior',type=str,default="/home/lijia/codes/202302/lijia/face-recognition/data/prototype/cluster_race/prior_A_B_W_feature.npy")


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

def train(train_dl,fr_model,optimizer,scheduler,e,fac_model=None):
    # scheduler.step()
    loss=0
    losses=0
    mu=1
    critrtia=AngularPenaltySMLoss(2048,8615)
    if isinstance(fr_model,list):
        for module in fr_model:
            module.train()
    else:
        fr_model.train()
    attrlen=len(attrlist)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    # intercount=e * len(train_dl.dataset)
    intercount=e*(len(train_dl))
    for i_bz,d in enumerate(tqdm(train_dl)):
        intercount=intercount+1
        feature= fr_model[0](d[0].to(device))
        y=fr_model[1](feature)
        if args.ingroup_loss:
            loss1=XE(y,d[1].to(device))
            pred_class_logits = fac_model(d[0].to(device))
            race_pre = torch.argmax(pred_class_logits, dim=1)
            # loss2=InGroupPenalty(feature,race_pre,len(attrlist))
            loss2=FairnessPenalty((torch.argmax(y,dim=1)==d[1].to(device)),race_pre,len(attrlist))
            loss=loss1+mu*loss2
        else:
            # loss1=XE(y,d[1].to(device))
            loss1=critrtia.loss_caculate(y,d[1].to(device))
            loss=loss1
        losses = losses + loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i_bz %args.print_inter==0:
            pre_label=torch.argmax(y,dim=1)
            acc=(pre_label==d[1].to(device)).sum()/d[1].shape[0]
            writer.add_scalar('arcface/train_loss',losses,intercount)
            losses=0
            print("batch:{}/total batch:{}  loss:{}  total_loss acc:{}".format(str(i_bz), len(train_dl), loss,
                                                                                   acc))

def test(test_dl,fr_model,i):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    acc_total=0
    if isinstance(fr_model, list):
        fr_model[0].eval()
        fr_model[1].eval()
    else:
        fr_model.eval()
    result_pre=[]
    result_names=[]

    for data,label,name in tqdm(test_dl):
        if isinstance(fr_model,list):
            # feature=fr_model[0](data.to(device))
            y=fr_model[1](fr_model[0](data.to(device)))
        else:
            _,y=fr_model(data.to(device))

        pre_label=torch.argmax(y,dim=1)
        acc_total+=(label.to(device)==pre_label).sum()
        result_pre+=pre_label.tolist()
        result_names+=list(name)
    data = pd.DataFrame({
        "Filename": result_names,
        "pre_id": result_pre
    })
    data.to_csv(str(i)+'test.csv',index=None)
    acc=acc_total/len(test_dl.dataset)
    print("test result: {}".format(acc))

# attrlist=["Asian","Black","White"]
attrlist=["Male","Female"]

args=makeargs()
device='cuda' if torch.cuda.is_available() else 'cpu'

if args.dataset=="vggface2":
    args.idclass=8615
elif args.dataset=="celeba":
    args.idclass=10178

fr_model=[]
fr_model.append(FR_model_backbone())
fr_model.append(FR_model_classifier(args.idclass,args.metric_type))
if os.path.exists(args.ckpt_path):
    fr_model=load_state_dict(fr_model,args.ckpt_path)
elif os.path.exists(args.ckpt_path_backbone):
    fr_model=load_state_dict_seperate(fr_model,args.ckpt_path_backbone,args.ckpt_path_classifier)

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
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,5,0.4)

# training
intercount=0
for i in range(0,args.epoch):
    writer = torch.utils.tensorboard.SummaryWriter('./log')
    print("start the {}th training:".format(str(i)))
    train(train_dl,fr_model,optimizer,scheduler,i,fac_model)
    scheduler.step()
    if isinstance(fr_model,list):
        torch.save({'epoch': i, 'state_dict': fr_model[0].state_dict()},
               os.path.join(args.save_path,  'method3_vggface2_attention_backbone_prior_{}.pth.tar'.format(str(i))))
        torch.save({'epoch': i, 'state_dict': fr_model[1].state_dict()},
               os.path.join(args.save_path, 'method3_vggface2_attention_classifier_prior_{}.pth.tar'.format(str(i))))
    else:
        torch.save({'epoch': i, 'state_dict': fr_model.state_dict()},
               os.path.join(args.save_path, str(i) + 'celeba_baseline.pth.tar'))
    # test(test_dl,fr_model,i)

