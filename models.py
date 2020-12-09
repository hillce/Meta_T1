# Generator and Discriminator models
# Charles E Hill
# 21/10/2020

from ast import increment_lineno
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from collections import OrderedDict
import matplotlib.pyplot as plt 

from torch.utils.tensorboard import SummaryWriter
from datasets import T1_Train_Dataset, ToTensor, collate_fn, Normalise
# Classic Generator Models

class Braided_AutoEncoder(nn.Module):
    """
    Generator network for recreating the missing image in the sequence of T1,T2,PDFF maps etc.
    """

    def __init__(self,nCImg,nCMeta,xDim=288,yDim=384,device=device):
        super(Braided_AutoEncoder,self).__init__()
        self.device = device
        self.nCImg = nCImg
        self.nCMeta = nCMeta

        self.block1 = Braided_Block(self.nCImg,self.nCMeta,8,device=self.device)
        self.block2 = Braided_Block(8,8,16,device=self.device)
        self.block3 = Braided_Block(16,16,32,device=self.device)
        self.block4 = Braided_Block(32,32,16,device=self.device)
        self.block5 = Braided_Block(16,16,8,device=self.device)
        self.block6 = Braided_Block(8,8,self.nCImg,device=self.device)

        self.conv = nn.Conv2d(self.nCImg,self.nCImg,1)
        self.fc = nn.Linear(self.nCMeta,self.nCMeta)

    def forward(self,img,meta):
        img,meta = self.block1(img,meta)
        img,meta = self.block2(img,meta)
        img,meta = self.block3(img,meta)
        img,meta = self.block4(img,meta)
        img,meta = self.block5(img,meta)
        img,meta = self.block6(img,meta)
        img = self.conv(img)
        meta = self.fc(meta)

        return (img,meta)

class Braided_UNet(nn.Module):
    """
    Generator network for recreating the missing image in the sequence of T1,T2,PDFF maps etc.
    """

    def __init__(self,nCImg,nCMeta,xDim=288,yDim=384,device=device):
        super(Braided_UNet,self).__init__()
        self.device = device
        self.nCImg = nCImg
        self.nCMeta = nCMeta

        self.block1 = Braided_Block(self.nCImg,self.nCMeta,8,device=self.device)
        self.ds1 = nn.MaxPool2d(2)

        self.block2 = Braided_Block(8,8,16,device=self.device)
        self.ds2 = nn.MaxPool2d(2)

        self.block3 = Braided_Block(16,16,32,device=self.device)
        self.ds3 = nn.MaxPool2d(2)

        self.block4 = Braided_Block(32,32,64,device=self.device)
        self.ds4 = nn.MaxPool2d(2)

        self.block5 = Braided_Block(64,64,128,device=self.device)
        self.ds5 = nn.MaxPool2d(2)

        self.block6 = Braided_Block(128,128,64,device=self.device)
        self.us1 = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)

        self.block7 = Braided_Block(64,64,32,device=self.device)
        self.us2 = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)

        self.block8 = Braided_Block(32,32,16,device=self.device)
        self.us3 = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)

        self.block9 = Braided_Block(16,16,8,device=self.device)
        self.us4 = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)

        self.block10 = Braided_Block(8,8,self.nCImg,device=self.device)
        self.us5 = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)

        self.conv1 = nn.Conv2d(self.nCImg,self.nCImg,1)
        self.conv2 = nn.Conv2d(self.nCImg,self.nCImg,1)
        self.fc1 = nn.Linear(self.nCMeta,self.nCMeta)
        self.fc2 = nn.Linear(self.nCMeta,self.nCMeta)

    def forward(self,img,meta):
        img,meta = self.block1(img,meta)
        img = self.ds1(img)
        img,meta = self.block2(img,meta)
        img = self.ds2(img)
        img,meta = self.block3(img,meta)
        img = self.ds3(img)
        img,meta = self.block4(img,meta)
        img = self.ds4(img)
        img,meta = self.block5(img,meta)
        img = self.ds5(img)
        img,meta = self.block6(img,meta)
        img = self.us1(img)
        img,meta = self.block7(img,meta)
        img = self.us2(img)
        img,meta = self.block8(img,meta)
        img = self.us3(img)
        img,meta = self.block9(img,meta)
        img = self.us4(img)
        img,meta = self.block10(img,meta)
        img = self.us5(img)

        img = F.relu(self.conv1(img))
        img = self.conv2(img)
        
        meta = self.fc1(meta)
        meta = self.fc2(meta)

        return (img,meta)
        # Image: Instance Normalisation -> x (scaling from meta) -> Conv -> + (with fully connected meta) -> RelU
        # Meta: Concat with Metrics from instance normalisation -> Batch Normalisation -> FC -> ReLU

class Braided_Block(nn.Module):

    def __init__(self,inCImg,inCMeta,outC,device=torch.device("cuda:0")):
        super(Braided_Block,self).__init__()
        self.device = device

        self.inNorm = nn.InstanceNorm2d(inCImg)
        self.conv = nn.Conv2d(inCImg,outC,3,padding=1)
        self.bnImg = nn.BatchNorm2d(outC)

        # self.scalingFC = nn.Linear(inCMeta*2,1)
        self.bnMeta = nn.BatchNorm1d(inCMeta+inCImg)
        self.fc = nn.Linear(inCMeta+inCImg,outC)

    def mean_n_std(self,tensor):
        mean = tensor.mean(dim=(2,3))
        std = tensor.std(dim=(2,3))
        out = torch.ones((mean.size()[0],mean.size()[1]),device=self.device)
        out *= mean[:,:]
        out *= std[:,:]
        return out

    def forward(self,img,meta):
        x = self.inNorm(img)
        
        # print("img pre mean_n_std: ",img.size())
        mNS = self.mean_n_std(img)
        # print("MNS output: ",mNS.size())
        # print("Meta pre mean cat: ",meta.size())

        meta = torch.cat((meta,mNS),dim=1)

        # tempMeta = self.scalingFC(meta)

        # x *= tempMeta[:,:,None,None] # THIS IS AMAZING: Its how you add or multiply an array of (A,B,C,D) and (A,B)

        x = self.conv(x)
        # print("Meta pre batch norm: ",meta.size())
        meta = self.bnMeta(meta)
        meta = self.fc(meta)

        x += meta[:,:,None,None]

        img = F.relu(self.bnImg(x))
        meta = F.relu(meta)

        return img,meta

class Down_Conv_Braided(nn.Module):

    def __init__(self,inCImg,inCMeta,outC,device=torch.device("cuda:0")):
        super(Down_Conv_Braided,self).__init__()

        self.braided1 = Braided_Block(inCImg,inCMeta,outC,device=device)
        self.braided2 = Braided_Block(outC,outC,outC,device=device)
        self.ds1 = nn.MaxPool2d(2)

    def forward(self,img,meta):

        img,meta = self.braided1(img,meta)
        img,meta = self.braided2(img,meta)
        img = self.ds1(img)

        return img, meta

class Up_Conv_Braided(nn.Module):

    def __init__(self,inCImg,inCMeta,outC,device=torch.device("cuda:0")):
        super(Up_Conv_Braided,self).__init__()

        self.us1 = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)

        self.braided1 = Braided_Block(inCImg,inCMeta,outC,device=device)
        self.braided2 = Braided_Block(outC,outC,outC,device=device)

    def forward(self, img0, img1, meta):
        img0 = self.us1(img0)

        diffY = torch.tensor([img1.size()[2] - img0.size()[2]])
        diffX = torch.tensor([img1.size()[3] - img0.size()[3]])

        img0 = F.pad(img0, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        img = torch.cat([img1, img0], dim=1)
        # print(img.size())

        img, meta = self.braided1(img,meta)
        img, meta = self.braided2(img,meta)

        return img,meta

class Braided_UNet_Complete(nn.Module):

    def __init__(self,nCImg,nCMeta,xDim=288,yDim=384,device=device):
        super(Braided_UNet_Complete,self).__init__()


        self.braidedBlockIn = Braided_Block(nCImg,nCMeta,nCImg,device=device)
        self.down1 = Down_Conv_Braided(nCImg,nCImg,8,device=device)
        self.down2 = Down_Conv_Braided(8,8,16,device=device)
        self.down3 = Down_Conv_Braided(16,16,32,device=device)
        self.down4 = Down_Conv_Braided(32,32,64,device=device)

        self.up1 = Up_Conv_Braided(64+32,64,32,device=device)
        self.up2 = Up_Conv_Braided(32+16,32,16,device=device)
        self.up3 = Up_Conv_Braided(16+8,16,8,device=device)
        self.up4 = Up_Conv_Braided(8+nCImg,8,nCImg,device=device)

        self.outConv1 = nn.Conv2d(nCImg,nCImg,kernel_size=1)
        self.outConv2 = nn.Conv2d(nCImg,nCImg,kernel_size=1)

        self.outfc1 = nn.Linear(nCImg,nCMeta)
        self.outfc2 = nn.Linear(nCMeta,nCMeta)

    def forward(self,img,meta):

        # print(img.size())
        img, meta = self.braidedBlockIn(img,meta)
        img0,meta0 = self.down1(img,meta)
        # print("Img size 0: ",img0.size())
        img1,meta1 = self.down2(img0,meta0)
        # print("Img size 1: ",img1.size())
        img2,meta2 = self.down3(img1,meta1)
        # print("Img size 2: ",img2.size())
        img3,meta3 = self.down4(img2,meta2)
        # print("Img size 3: ",img3.size())
        img4,meta4 = self.up1(img3,img2,meta3)
        # print("Img size 4: ",img4.size())
        img5,meta5 = self.up2(img4,img1,meta4)
        # print("Img size 5: ",img5.size())
        img6,meta6 = self.up3(img5,img0,meta5)
        # print("Img size 6: ",img6.size())
        img7,meta7 = self.up4(img6,img,meta6)
        # print("Img size 7: ",img7.size())
        # print("Final Meta Size: ",meta7.size())

        img = self.outConv1(img7)
        img = self.outConv2(img)

        meta = self.outfc1(meta7)
        meta = self.outfc2(meta)

        return img,meta

if __name__ == "__main__":

    img = torch.randn((4,7,288,384))
    meta = torch.randn((4,3))

    net = Braided_UNet_Complete(7,3,device=torch.device("cpu"))

    imgOut, metaOut = net(img,meta)

    print(imgOut.size(),metaOut.size())
    # toT = ToTensor()
    # norm = Normalise()
    # trnsIn = transforms.Compose([toT])
    # bSize = 4

    # datasetTrain = T1_Train_Dataset(fileDir="C:/fully_split_data/",t1MapDir="C:/T1_Maps/",size=10000,transform=trnsIn,load=False)
    # loaderTrain = DataLoader(datasetTrain,batch_size=bSize,shuffle=True,collate_fn=collate_fn,pin_memory=False)

    # testBatch = next(iter(loaderTrain))
    # img = testBatch["Images"]
    # print(img.size())
    # meta = testBatch["InvTime"].type(torch.FloatTensor)
    # print(meta.size(),meta)

    # plt.imshow(img[0,0,:,:].numpy())
    # plt.show()

    # loss = nn.MSELoss()

    # net = Braided_UNet(7,7,device=torch.device("cpu"))

    # imgOut,metaOut = net(img,meta)

    # print(imgOut.size(),metaOut.size())

    # err = loss(imgOut,img)

    # plt.figure()
    # plt.imshow(imgOut[0,0,:,:].detach().numpy())
    # plt.show()

    # plt.figure()
    # plt.imshow(img[0,0,:,:].detach().numpy())
    # plt.show()
    
    # print(err.item())

    # err.backward()