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