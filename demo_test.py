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
    parse.add_argument('--train_csv',type=str,default='/media/lijia/DATA/lijia/data/vggface2/anno/train_id_sample_8615.csv')
    parse.add_argument('--test_csv',type=str,default='/media/lijia/DATA/lijia/data/vggface2/anno/test_id_sample_8615.csv')

    # training setting
    parse.add_argument('--batch_size',type=int,default=48)
    parse.add_argument('-lr',type=float,default=0.0001)
    parse.add_argument('--warmup_step',type=int,default=0)
    parse.add_argument('--epoch',type=int,default=5)
    parse.add_argument('--mu',type=float,default=0.5)
    parse.add_argument('--print_inter',type=int,default=200)
    parse.add_argument('--test_type',type=str,default='normal',choices=['causal','normal'])
    parse.add_argument('--ingroup_loss',type=bool,default=False)

    # model setting
    parse.add_argument('--backbone_type',type=str,choices=['resnet50','senet'],default='resnet50')
    parse.add_argument('--idclass',type=int,default=8615)
    parse.add_argument('--ckpt_path',type=str,default='/home/lijia/codes/202302/lijia/face-recognition/checkpoints/ingroup/2_baseline.pth.tar')
    parse.add_argument('--ckpt_path_backbone',type=str,default='/home/lijia/codes/202302/lijia/face-recognition/checkpoints/ingroup/0_causalnet_backbone.pth.tar')
    parse.add_argument('--ckpt_path_classifier',type=str,default='/home/lijia/codes/202302/lijia/face-recognition/checkpoints/ingroup/0_causalnet_classifier.pth.tar')
    parse.add_argument('--dataset',type=str,default="vggface2",choices=["celeba","vggface2"])


    args=parse.parse_args()
    return args

def test(test_dl,fr_model):
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
    data.to_csv('test.csv',index=None)
    print("test result: {}".format(acc_total/len(test_dl.dataset)))


attrlist=["Asian","Black","White"]
args=makeargs()
device='cuda' if torch.cuda.is_available() else 'cpu'

if args.dataset=="vggface2":
    args.idclass=8615
elif args.dataset=="celeba":
    args.idclass=10178

if args.test_type=='normal':
    print('load normal model...')
    fr_model=FR_model(args.idclass)
    if os.path.exists(args.ckpt_path):
        fr_model=load_state_dict(fr_model,args.ckpt_path)
    fr_model.to(device)
else:
    print('load causal model...')
    fr_model=[]
    fr_model.append(FR_model_backbone())
    fr_model.append(FR_model_classifier(args.idclass))
    fr_model=load_state_dict_seperate(fr_model,args.ckpt_path_backbone,args.ckpt_path_classifier)
    for module in fr_model:
        module.to(device)

# dataset
if args.dataset=="celeba":
    _,test_dl=loaddata_celeba(args)
else:
    _,test_dl=loaddata(args)

test(test_dl,fr_model)


