import pandas as pd
import torch
from torch.utils.data import DataLoader
from data.imagedata import *
from utils.prototype import *


def load_state_dict(model,dictpath,fine_tune=False):
    pre_ckpt=torch.load(dictpath)
    if isinstance(model,list):
        model[0].backbone.load_state_dict(pre_ckpt['state_dict'],False)
        if not fine_tune:
            if isinstance(model[1].fc,torch.nn.Linear):
                model[1].load_state_dict(pre_ckpt['state_dict'],False)
    else:
        model.load_state_dict(pre_ckpt['state_dict'])
    return model

def load_state_dict_seperate(model,backbone_ckpt_path,classifier_path):
    assert isinstance(model,list)
    backbone_ckpt=torch.load(backbone_ckpt_path)
    classifier_ckpt=torch.load(classifier_path)
    model[0].load_state_dict(backbone_ckpt['state_dict'])
    try:
        model[1].load_state_dict(classifier_ckpt['state_dict'])
    except:
        print("pre-trained not load")
    return model
def get_image_attr(dir,maadpath,idpath,attrlist,bz):
    maad=pd.read_csv(maadpath).drop(["id"],axis=1)
    idfile=pd.read_csv(idpath)
    maad=maad.merge(idfile)
    # idlist=list(idfile["Filename"])
    dataset=image_attr_name(dir,maad,attrlist)
    dl=DataLoader(dataset,bz)
    return dl

def loaddata(args):
    train_csv=pd.read_csv(args.train_csv)
    test_csv=pd.read_csv(args.test_csv)
    train_dataset=imagedataset(args.image_dir,train_csv)
    test_dataset=imagedataset(args.image_dir,test_csv)
    train_dl=DataLoader(train_dataset,args.batch_size)
    test_dl=DataLoader(test_dataset,args.batch_size)
    return train_dl,test_dl

def load_data_yaml(yaml_dict):
    train_csv=pd.read_csv(yaml_dict['path']['train_csv_path'])
    test_csv=pd.read_csv(yaml_dict['path']['test_csv_path'])
    train_dataset=imagedataset(yaml_dict['path']['dir'],train_csv)
    test_dataset=imagedataset(yaml_dict['path']['dir'],test_csv)
    return train_dataset,test_dataset


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

def load_proto(dir,model,names):
    protos=Protos(dir,model)
    concept_dict=protos.extract_feature()
    concept_list=[]
    assert len(names)==len(concept_dict.keys())
    for name in names:
        concept_list.append(concept_dict[name])
    return torch.cat(concept_list)

def load_proto_cluster(dir,model,names,cluster_n):
    protos=Cluster_Proto(dir,model,cluster_n)
    concept_dict,prior_dict=protos.get_cluster_concept()
    concept_list=[]
    prior_list=[]
    assert len(names)==len(concept_dict.keys())
    for name in names:
        concept_list.append(concept_dict[name].unsqueeze(0))
        prior_list.append(torch.tensor(prior_dict[name]).unsqueeze(0))
    return torch.cat(concept_list),torch.cat(prior_list)