import numpy as np
import torch
import torch.nn as nn


def XE(output,target):
    return nn.CrossEntropyLoss(output,target)

def InGroup():
    return