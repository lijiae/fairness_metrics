import torch
import torch.nn as nn
import torch.nn.functional as F

from model.resnet50 import *

class FR_model(nn.Module):
    def __init__(self,numclass):
        super(FR_model, self).__init__()
        self.stem = nn.Sequential(*[
            Conv(3, 64, k_size=7, stride=2),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])
        self.stages = nn.Sequential(*[
            self.make_stages(64, 256, down_samples=False, num_blocks=3),
            self.make_stages(256, 512, True, 4),
            self.make_stages(512, 1024, True, 6),
            self.make_stages(1024, 2048, True, 3)
        ])
        self.average = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, numclass)

    def make_stages(self, in_ch, out_ch, down_samples, num_blocks):
        layers = [BottleBlock(in_ch, out_ch, down_samples)]
        for _ in range(num_blocks - 1):
            layers.append(BottleBlock(out_ch, out_ch, down_sample=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.stem(x)
        output = self.stages(output)
        output = self.average(output)
        y = self.fc(output.reshape(output.shape[:2]))
        return output,y