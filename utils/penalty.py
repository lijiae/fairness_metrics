import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from numpy import mean
import torch.nn.functional as F

def XE(output,target):
    XEloss=nn.CrossEntropyLoss()
    return XEloss(output,target)

def InGroupPenalty(output,raceresult,length):
    sum=[]
    criterion=nn.Sigmoid()
    for i in range(length):
        groupfeature=output[raceresult==i]
        if len(groupfeature)>1:
            group=groupfeature.var()
            sum.append(group.item())
    score=criterion(torch.tensor(-mean(sum)))
    return score

def FairnessPenalty(pre_result,race_pre,group_len):
    sum=[]
    # criterion=nn.Sigmoid()
    for i in range(group_len):
        group_result=pre_result[race_pre==i]
        if len(group_result)>=1:
            group_acc=(group_result.sum()/len(group_result)).item()
            sum.append(group_acc)
    return np.std(np.array(sum))

class Similarity():
    def __init__(self):
        self.cos_sim=nn.CosineSimilarity(-1)

    def CosSim(self,input,matrix):
        return self.cos_sim(input,matrix)

class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

class XE_loss(nn.Module):
    def __init__(self):
        super(XE_loss).__init__()
        self.loss_fucntion=nn.CrossEntropyLoss()

    def loss_caculate(self, wf, labels):
        return self.loss_fucntion(wf,labels)

class AngularPenaltySMLoss(nn.Module):

    def __init__(self, in_features, out_features, loss_type='arcface', eps=1e-7, s=None, m=None):
        '''
        Angular Penalty Softmax Loss

        Three 'loss_types' available: ['arcface', 'sphereface', 'cosface']
        These losses are described in the following papers:

        ArcFace: https://arxiv.org/abs/1801.07698
        SphereFace: https://arxiv.org/abs/1704.08063
        CosFace/Ad Margin: https://arxiv.org/abs/1801.05599

        '''
        super(AngularPenaltySMLoss, self).__init__()
        loss_type = loss_type.lower()
        assert loss_type in ['arcface', 'sphereface', 'cosface']
        if loss_type == 'arcface':
            self.s = 64.0 if not s else s
            self.m = 0.5 if not m else m
        if loss_type == 'sphereface':
            self.s = 64.0 if not s else s
            self.m = 1.35 if not m else m
        if loss_type == 'cosface':
            self.s = 30.0 if not s else s
            self.m = 0.4 if not m else m
        self.loss_type = loss_type
        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps

    def loss_caculate(self, wf, labels):
        if self.loss_type == 'cosface':
            numerator = self.s * (torch.diagonal(wf.transpose(0, 1)[labels]) - self.m)
        elif self.loss_type == 'arcface':
            numerator = self.s * torch.cos(torch.acos(
                torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)) + self.m)
        elif self.loss_type == 'sphereface':
            numerator = self.s * torch.cos(self.m * torch.acos(
                torch.clamp(torch.diagonal(wf.transpose(0, 1)[labels]), -1. + self.eps, 1 - self.eps)))

        excl = torch.cat([torch.cat((wf[i, :y], wf[i, y + 1:])).unsqueeze(0) for i, y in enumerate(labels)], dim=0)
        denominator = torch.exp(numerator) + torch.sum(torch.exp(self.s * excl), dim=1)
        L = numerator - torch.log(denominator)
        return -torch.mean(L)