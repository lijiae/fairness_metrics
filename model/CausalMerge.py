import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet50 import *

class CI_model(nn.Module):
    def __init__(self,numclass,backbone_type="resnet"):
        super(CI_model,self).__init__()
        if backbone_type == "resnet50":
            self.backbone=iresnet50_backbone()

        self.classifier = nn.Linear(2048, numclass)

    def forward(self, input):
        output=self.backbone(input)
        output=self.classifier(output)
        return output


