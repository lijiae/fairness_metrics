import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
from numpy import mean



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