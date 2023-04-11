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
    parse.add_argument('--batch_size',type=int,default=16)
    parse.add_argument('-lr',type=float,default=0.0001)
    parse.add_argument('--warmup_step',type=int,default=0)
    parse.add_argument('--epoch',type=int,default=5)
    parse.add_argument('--mu',type=float,default=0.5)
    parse.add_argument('--print_inter',type=int,default=200)
    parse.add_argument('--test_type',type=str,default='causal',choices=['causal','normal'])
    parse.add_argument('--ingroup_loss',type=bool,default=False)

    # model setting
    parse.add_argument('--backbone_type',type=str,choices=['resnet50','senet'],default='resnet50')
    parse.add_argument('--train_type',type=str,default='causal',choices=['causal','normal'])
    parse.add_argument('--idclass',type=int,default=8615)
    parse.add_argument('--ckpt_path',type=str,default='/home/lijia/codes/202302/lijia/face-recognition/checkpoints/normal/5_vggface2_resnet.pth.tar')
    parse.add_argument('--ckpt_path_backbone',type=str,default='/home/lijia/codes/202302/lijia/face-recognition/checkpoints/vggface2-causal-2/causalnet_backbone_celeba_0.pth.tar')
    parse.add_argument('--ckpt_path_classifier',type=str,default='/home/lijia/codes/202302/lijia/face-recognition/checkpoints/vggface2-causal-2/causalnet_classifier_celeba_0.pth.tar')
    parse.add_argument('--attr_net_path',type=str,default='/home/lijia/codes/202302/lijia/face-recognition/checkpoints/AttributeNet.pkl')
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

def test_demo(test_dl,fr_model,fac_model=None):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    acc_total=0
    if isinstance(fr_model, list):
        fr_model[0].eval()
        fr_model[1].eval()
    else:
        fr_model.eval()
    result_pre=[]
    result_names=[]

    attrlen=len(attrlist)
    device='cuda' if torch.cuda.is_available() else 'cpu'

    if args.train_type == 'causal':
        ch=2048
        for data, label, name in tqdm(test_dl):
            feature=fr_model[0](data.to(device))
            pred_class_logits=fac_model(data.to(device))
            race_pre=torch.argmax(pred_class_logits,dim=1)
            get_feature_map=fac_model.get_feature_map()
            # cam is f,logit is c
            f_x_c=[]
            # for i in range(attrlen):
            #     classid=torch.ones_like(race_pre)*i
            #     f_x_c.append(getGradCam(get_feature_map,pred_class_logits,classid))
            # cams=torch.stack(f_x_c,0) # 将list中cam特征合并
            # weights_race=torch.mul(cams,pred_class_logits.transpose(0,1).unsqueeze(2).unsqueeze(3)).sum(0) # 和logit加权
            # feature=(torch.mul(weights_race.unsqueeze(1),feature)+feature)/2 # 更新新的feature

            # f is avg+feature,logit is c
            # f_x_c=[]
            for i in range(attrlen):
                classid=torch.ones_like(race_pre)*i
                f_x_c.append(getGradCam(get_feature_map,pred_class_logits,classid))
            cams=torch.stack(f_x_c,0) # 将list中cam特征合并
            concept_cams=torch.mul(cams.unsqueeze(2).repeat([1,1,ch,1,1]),concept.unsqueeze(1).repeat([1,cams.shape[1],1,1,1]))
            feature=(feature+torch.mul(concept_cams,pred_class_logits.transpose(0,1).unsqueeze(2).unsqueeze(3).unsqueeze(4)).sum(0))/2
            y=fr_model[1](feature)
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

if args.train_type=='normal':
    fr_model=FR_model(args.idclass)
    if os.path.exists(args.ckpt_path):
        fr_model=load_state_dict(fr_model,args.ckpt_path)
    fr_model.to(device)
else:
    fr_model=[]
    fr_model.append(FR_model_backbone())
    fr_model.append(FR_model_classifier(args.idclass))
    # if os.path.exists(args.ckpt_path):
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
    # load prototype
    dir = "/home/lijia/codes/202302/lijia/face-recognition/data/prototype/race"
    concept = load_proto(dir, fr_model[0], attrlist).detach()
    # concept.grad.zero_()

# dataset
if args.dataset=="celeba":
    _,test_dl=loaddata_celeba(args)
else:
    _,test_dl=loaddata(args)

# test_demo(test_dl,fr_model,fac_model)
test(test_dl,fr_model)

