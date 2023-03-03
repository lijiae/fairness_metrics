import pandas as pd
import torch
from torch.utils.data import DataLoader
from data.imagedata import *


def load_state_dict(model,dictpath):
    pre_ckpt=torch.load(dictpath)
    # print(model)
    # model.backbone.load_state_dict(pre_ckpt,False)
    # model.classifier.load_state_dict(pre_ckpt.fc,False)
    model.load_state_dict(pre_ckpt['state_dict'])
    return model


def get_image_attr(dir,maadpath,idpath,attrlist,bz):
    maad=pd.read_csv(maadpath).drop(["id"],axis=1)
    idfile=pd.read_csv(idpath)
    maad=maad.merge(idfile)
    # idlist=list(idfile["Filename"])
    dataset=image_attr_name(dir,maad,attrlist)
    dl=DataLoader(dataset,bz)
    return dl