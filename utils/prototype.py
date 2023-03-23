import os
import numpy as np
import torchvision
from PIL import Image
import torch

class protos():
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
                data = torchvision.transforms.Resize((112, 112))(Image.open(os.path.join(self.dir,name,file)))
                img = data[:, :, ::-1]  # RGB -> BGR
                img = img.astype(np.float32)
                img -= mean_bgr
                img = img.transpose(2, 0, 1)  # C x H x W
                img = torch.from_numpy(img).float()
                feature,_=self.extract_feature(img.to("cuda").squeeze(0))
                filelist.append(feature)
            proto_dict[name]=filelist
        return proto_dict
