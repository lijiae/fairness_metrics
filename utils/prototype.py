import os
import numpy as np
import torchvision
from PIL import Image
import torch
from data.imagedata import image_from_dir
from torch.utils.data import DataLoader
from utils.kmeans_pytorch.kmeans_pytorch import kmeans


class Protos():
    # dir
    # --groups
    def __init__(self,dir,extractor):
        self.dir=dir
        self.groupnames=os.listdir(dir)
        self.extractor=extractor
        self.extractor.to("cuda")

    def extract_feature(self):
        mean_bgr = np.array([91.4953, 103.8827, 131.0912])
        proto_dict={}
        for name in self.groupnames:
            filenames=os.listdir(os.path.join(self.dir,name))
            filelist=[]
            for file in filenames:
                data = torchvision.transforms.Resize((224,224))(Image.open(os.path.join(self.dir,name,file)))
                data = np.array(data, dtype=np.uint8)
                img = data[:, :, ::-1]  # RGB -> BGR
                img = img.astype(np.float32)
                img -= mean_bgr
                img = img.transpose(2, 0, 1)  # C x H x W
                img = torch.from_numpy(img).float()
                feature=self.extractor(img.to("cuda").unsqueeze(0))
                filelist.append(feature)
            proto_dict[name]=torch.cat(filelist)
        return proto_dict

class cluster_Proto():
    def __init__(self,dir,extractor,num_cluster=10):
        self.device="cuda"
        self.dir=dir
        self.groupnames=os.listdir(dir)
        self.extractor=extractor
        self.extractor.to(self.device)
        self.mean_bgr = np.array([91.4953, 103.8827, 131.0912])
        self.cluster_num=num_cluster
        self.bz=32
        self.inter_size=5

    def extract_feature(self,img_batch):
        feature = self.extractor(img_batch.to(self.device))
        return feature

    def make_cluster_n(self,name):
        img_dataset = image_from_dir(os.path.join(self.dir,name))
        img_dataloader = DataLoader(img_dataset,self.bz)
        data=iter(iter(img_dataloader))

        cluster_ids_x, cluster_centers = kmeans(
            X=self.extract_feature(next(data)[0]), num_clusters=self.cluster_num, distance='euclidean',
            device=self.device
        )

        for _ in range(self.inter_size * img_dataloader.__len__() - 1):
            try:
                x = next(data)
            except:
                data = iter(img_dataloader)
                x = next(data)
            x=self.extract_feature(x)
            cluster_ids_x, cluster_centers = kmeans(
                X=x[0], num_clusters=self.cluster_num, cluster_centers=cluster_centers, distance='euclidean',
                device=self.device,tqdm_flag=False
            )

        for img_each_batch in img_dataloader:
            cluster_ids_x, cluster_centers = kmeans(
                X=img_each_batch[0], cluster_centers=cluster_centers, distance='euclidean',
                device=self.device,tqdm_flag=False
            )
        return cluster_centers

    def get_cluster_concept(self,path=""):
        # cluster_list=[]
        proto_dict={}
        for name in self.groupnames:
            cluster_n=self.make_cluster_n(name)
            # cluster_list.append(cluster_n)
            proto_dict[name] = torch.cat(cluster_n)
        return proto_dict