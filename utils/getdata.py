import pandas as pd
import torch
from torch.utils.data import DataLoader
from data.imagedata import *


def load_state_dict(model,dictpath):
    pre_ckpt=torch.load(dictpath)
    if isinstance(model,list):
        model[0].backbone.load_state_dict(pre_ckpt['state_dict'],False)
        model[1].load_state_dict(pre_ckpt['state_dict'],False)
    else:
        model.load_state_dict(pre_ckpt['state_dict'])
    return model

def load_state_dict_seperate(model,backbone_ckpt_path,classifier_path):
    assert isinstance(model,list)
    backbone_ckpt=torch.load(backbone_ckpt_path)
    classifier_ckpt=torch.load(classifier_path)
    model[0].load_state_dict(backbone_ckpt['state_dict'])
    model[1].load_state_dict(classifier_ckpt['state_dict'])
    return model
def get_image_attr(dir,maadpath,idpath,attrlist,bz):
    maad=pd.read_csv(maadpath).drop(["id"],axis=1)
    idfile=pd.read_csv(idpath)
    maad=maad.merge(idfile)
    # idlist=list(idfile["Filename"])
    dataset=image_attr_name(dir,maad,attrlist)
    dl=DataLoader(dataset,bz)
    return dl

def CAM(feature, gradient):
    assert len(feature.shape) == 4
    assert len(gradient.shape) == 4

    weight = torch.mean(gradient, dim=(2, 3), keepdim=True)  # Shape [Batch, Channel]
    cam = torch.relu(torch.sum(feature * weight, dim=1))  # Shape [Batch, W, H]

    # Normalization
    cam_min = torch.min(
        torch.min(cam, dim=-1, keepdim=True).values,
        dim=-2, keepdim=True).values
    cam -= cam_min
    cam_max = torch.max(
        torch.max(cam, dim=-1, keepdim=True).values,
        dim=-2, keepdim=True).values
    cam /= cam_max
    cam_new=torch.where(torch.isnan(cam),torch.zeros_like(cam),cam)
    return cam_new
def getGradCam(gt_features_input, pred_class_logits, classes):
    gt_features_input.retain_grad()

    batch_index = np.arange(classes.shape[0])  # batch [0, 1, 2, 3, ...]
    classes_index = np.array([batch_index, classes])  # 选择每个batch每个样本对应选择类别的响应值
    # counter_classes = self.counter_class[classes]

    scores = pred_class_logits[classes_index].sum()  # 响应值相加
    scores.backward(retain_graph=True)  # 你如果只回传一次不需要设置retain_graph为True

    gradient = gt_features_input.grad

    gt_cam = CAM(gt_features_input, gradient)  # 输入为抓取的特征和梯度

    gt_features_input.grad.zero_()  # 记得梯度清空，按你自己之前demo，我的模型参数清空没写在这里

    return gt_cam