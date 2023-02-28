import torch
from torch import nn
from model.resnet50 import ResNet50
from model.AttributeNet import AttributeNet

class CausalFairModel(nn.Module):
    def __init__(self,id_num,BackboneType="resnet"):
        super(CausalFairModel,self).__init__()
        if BackboneType=="resnet":
            self.fr_backbone=ResNet50(id_num)
        self.fac=AttributeNet()

