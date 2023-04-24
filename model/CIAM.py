import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn


class CIAM(nn.Module):
    def __init__(self,concept_n=10):
        super(CIAM,self).__init__()
        self.sim=nn.CosineSimilarity(dim=-1)
        self.concept_n=concept_n
        self.coff=0.5
    def forward(self,x,c,prior):
        # x: bz*h*w*c
        # c: n*h*w*c
        sim_socre=self.sim(x.reshape(x.size()[0],-1).unsqueeze(1).repeat(1,self.concept_n,1),c.reshape(x.size()[0],c,-1))
        sim_matrix=self.softmax(sim_socre,dim=1)
        causal_attention=torch.mm(sim_matrix,c)
        image_level_context=causal_attention*prior.unsqueeze(-1).unsqueeze(-1)
        return image_level_context
