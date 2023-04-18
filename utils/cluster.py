import glob
import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import torch
from utils.kmeans_pytorch.kmeans_pytorch import kmeans
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader
import numpy as np


class get_img_batch(Dataset):
    def __init__(self,imgdir):
        self.imgdir = glob.glob(os.path.join(imgdir, '*.jpg'))

    def __getitem__(self, item):
        imgname=self.imgdir[item]
        img=cv2.imread(imgname).reshape(-1)
        return img,imgname
    def __len__(self):
        return len(self.imgdir)


def plot_blobs(x, cluster_ids_x):
    pca = PCA(2)
    y_2d = pca.fit_transform(y)
    x_2d = pca.transform(x)

    plt.figure(figsize=(4, 3), dpi=160)
    plt.scatter(x_2d[:, 0], x_2d[:, 1], c=cluster_ids_x, cmap='cool')
    # plt.scatter(y_2d[:, 0], y_2d[:, 1], c=cluster_ids_y, cmap='cool', marker='X')
    plt.scatter(
        cluster_centers[:, 0], cluster_centers[:, 1],
        c='white',
        alpha=0.6,
        edgecolors='black',
        linewidths=2
    )
    # plt.axis([-1, 1, -1, 1])
    plt.tight_layout()
    plt.show()


# config
dir="/media/lijia/DATA/lijia/data/vggface2/average_face/race/1" #image dir
# 消融
num_clusters = 10
device='cuda' if torch.cuda.is_available() else 'cpu'
bz=32

img_dataset=get_img_batch(dir)
img_dataloader=DataLoader(img_dataset,bz)
y=torch.tensor([])


data=iter(iter(img_dataloader))
cluster_ids_x, cluster_centers = kmeans(
    X=next(data)[0], num_clusters=num_clusters, distance='euclidean',
    device=device
)
# global cluster_centers


iteration=20
for _ in range(iteration*img_dataloader.__len__()-1):
    try:
        x=next(data)
    except:
        data=iter(img_dataloader)
        x=next(data)
    cluster_ids_x, cluster_centers = kmeans(
        X=x[0], num_clusters=num_clusters, cluster_centers=cluster_centers, distance='euclidean',
        device=device
    )
print(iteration*img_dataloader.__len__()-1)
print(cluster_centers.shape)

names=[]
cluster_ids=[]
for img_each_batch in img_dataloader:
    cluster_ids_x, cluster_centers = kmeans(
        X=img_each_batch[0], num_clusters=num_clusters, distance='euclidean', device=device
    )
    cluster_ids+=cluster_ids_x.tolist()
    names+=list(img_each_batch[1])

print(cluster_centers.shape)
print(cluster_ids)
print(names)

np.save("../data/cluster/class/10-white_cluster.npy", cluster_centers.numpy())
pd.DataFrame({
    "Filename":names,
    "cluster":cluster_ids
}).to_csv("../data/cluster/class/10-white_cluster.csv", index=None)
