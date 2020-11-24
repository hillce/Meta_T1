# Generator and Discriminator models
# Charles E Hill
# 21/10/2020

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from torch.utils.tensorboard import SummaryWriter
# Classic Generator Models

class Braided_AutoEncoder(nn.Module):
    """
    Generator network for recreating the missing image in the sequence of T1,T2,PDFF maps etc.
    """

    def __init__(self,nCImg,nCMeta,xDim=288,yDim=384):
        super(Braided_AutoEncoder,self).__init__()
        self.nCImg = nCImg
        self.nCMeta = nCMeta

        self.block1 = Braided_Block(self.nCImg,self.nCMeta,8)
        self.block2 = Braided_Block(8,8,16)
        self.block3 = Braided_Block(16,16,32)
        self.block4 = Braided_Block(32,32,16)
        self.block5 = Braided_Block(16,16,8)
        self.block6 = Braided_Block(8,8,self.nCImg)

    def forward(self,img,meta):
        img,meta = self.block1(img,meta)
        img,meta = self.block2(img,meta)
        img,meta = self.block3(img,meta)
        img,meta = self.block4(img,meta)
        img,meta = self.block5(img,meta)
        img,meta = self.block6(img,meta)

        return (img,meta)

        # Image: Instance Normalisation -> x (scaling from meta) -> Conv -> + (with fully connected meta) -> RelU
        # Meta: Concat with Metrics from instance normalisation -> Batch Normalisation -> FC -> ReLU


class Braided_Block(nn.Module):

    def __init__(self,inCImg,inCMeta,outC):
        super(Braided_Block,self).__init__()

        self.inNorm = nn.InstanceNorm2d(inCImg)
        self.conv = nn.Conv2d(inCImg,outC,3,padding=1)

        self.scalingFC = nn.Linear(inCMeta+2,1)
        self.bnMeta = nn.BatchNorm1d(inCMeta+2)
        self.fc = nn.Linear(inCMeta+2,outC)

    def mean_n_std(self,tensor):
        mean = tensor.mean(dim=(1,2,3)).unsqueeze(1)
        std = tensor.std(dim=(1,2,3)).unsqueeze(1)
        out = torch.tensor([(x,y) for x,y in zip(mean,std)])
        return out

    def forward(self,img,meta):
        x = self.inNorm(img)
        
        mNS = self.mean_n_std(img)

        tempMeta = torch.zeros((meta.size()[0],meta.size()[1]+2))
        for i in range(meta.size()[0]):
            tempMeta[i] = torch.cat((meta[i],mNS[i]))
        meta = tempMeta

        tempMeta = self.scalingFC(meta)

        x *= tempMeta[:,:,None,None] # THIS IS AMAZING: Its how you add or multiply an array of (A,B,C,D) and (A,B)

        x = self.conv(x)
        meta = self.bnMeta(meta)
        meta = self.fc(meta)

        x += meta[:,:,None,None]
                
        img = F.relu(x)
        meta = F.relu(meta)

        return img,meta



if __name__ == "__main__":
    img = torch.randn((4,7,288,384))
    meta = torch.randn((4,7))

    loss = nn.MSELoss()

    net = Braided_AutoEncoder(7,7,xDim=288,yDim=384)

    imgOut,metaOut = net(img,meta)

    err = loss(imgOut,img)

    err.backward()