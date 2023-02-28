import pandas as pd
from torch.utils.data import DataLoader
from data.imagedata import *


def get_image_attr(dir,maadpath,idpath,attrlist,bz):
    maad=pd.read_csv(maadpath).drop(["id"],axis=1)
    idfile=pd.read_csv(idpath)
    maad=maad.merge(idfile)
    # idlist=list(idfile["Filename"])
    dataset=image_attr_name(dir,maad,attrlist)
    dl=DataLoader(dataset,bz)
    return dl