import torch
from torch.utils.data import Dataset,DataLoader
import cv2
import numpy as np
import matplotlib.pyplot as plt
from kmeans_pytorch import kmeans, kmeans_predict
from sklearn.decomposition import PCA
from sklearn import datasets
import os
import glob

class get_img_batch(Dataset):
    def __init__(self,imgdir):
        self.imgdir = glob.glob(os.path.join(imgdir, '*.jpg'))

    def __getitem__(self, item):
        imgname=self.imgdir[item]
        img=cv2.imread(imgname)
        return img,imgname
    def __len__(self):
        return len(self.imgdir)


def plot_blobs(x, cluster_ids_x):
    pca = PCA(2)
    y_2d = pca.fit_transform(y)
    x_2d = pca.transform(x)

    plt.figure(figsize=(4, 3), dpi=160)
    plt.scatter(x_2d[:, 0], x_2d[:, 1], c=cluster_ids_x, cmap='cool')
    plt.scatter(y_2d[:, 0], y_2d[:, 1], c=cluster_ids_y, cmap='cool', marker='X')
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
dir="" #image dir
num_clusters = 6
device='cuda' if torch.cuda.is_available() else 'cpu'
bz=32

img_dataset=get_img_batch(dir)
img_dataloader=DataLoader(img_dataset,bz)
y=torch.zeros((img_dataset.__len__(),1))

for img_each_batch in img_dataloader:
    cluster_ids_x, cluster_centers = kmeans(
        X=img_each_batch, num_clusters=num_clusters, distance='euclidean', device=device
    )

cluster_ids_y = kmeans_predict(
        y, cluster_centers, 'euclidean', device=device
    )