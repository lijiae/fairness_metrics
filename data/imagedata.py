from torch.utils.data import Dataset
import numpy as np
import torch
import torchvision
from PIL import Image
import os

class image_attr_name(Dataset):
    def __init__(self,dir,maadfile,attrlist,isName=False):
        self.namelist=list(maadfile["Filename"])
        try:
            maadfile=maadfile[attrlist]
        except:
            print("your elements in attributes list must be in"+str(attrlist))
        device='cuda' if torch.cuda.is_available() else 'cpu'
        self.dir=dir
        self.attrlist= attrlist
        self.maad=torch.tensor(maadfile.values).to(device)
        self.mean_bgr=np.array([91.4953, 103.8827, 131.0912])

        # if isId:
        #     self.idlist=torch.tensor(idlist)
        # self.namelist=namelist
        # self.isId=isId
        self.isName=isName

    def transform(self, img):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float32)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)  # C x H x W
        img = torch.from_numpy(img).float()
        return img

    def __getitem__(self, item):
        imgname=self.namelist[item]
        if not os.path.exists(os.path.join(self.dir,imgname)):
            print("not exist for path{}".format(os.path.join(self.dir,imgname)))
        data = torchvision.transforms.Resize(224)(Image.open(os.path.join(self.dir,imgname)))
        data = np.array(data, dtype=np.uint8)
        data = self.transform(data)

        attr=self.maad[item]

        if self.isName:
            return data,attr,self.namelist[item]
        return data,attr

    def __len__(self):
        return len(self.namelist)

    def get_attrname(self):
        return len(self.attrlist)

class imagedataset(Dataset):
    def __init__(self,imagepath,idfile):
        device='cuda' if torch.cuda.is_available() else 'cpu'
        self.mean_bgr=np.array([91.4953, 103.8827, 131.0912])
        self.idtensor=torch.tensor(idfile["id"].values).to(device)
        self.namelist=list(idfile["Filename"])
        self.dir=imagepath

    def __len__(self):
        return len(self.namelist)

    def __getitem__(self, index):
        # Sample
        # sample=self.idfile.iloc[index]

        # data and label information
        imgname=self.namelist[index]
        id=self.idtensor[index]
        label=torch.tensor(int(id)).long()
        data = torchvision.transforms.Resize(224)(Image.open(os.path.join(self.dir,imgname)))
        data = np.array(data, dtype=np.uint8)
        data = self.transform(data)
        # label = np.int32(label)

        return data.float(), label,imgname

    def transform(self, img):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float32)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)  # C x H x W
        img = torch.from_numpy(img).float()
        return img