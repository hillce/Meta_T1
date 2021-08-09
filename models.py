# Generator and Discriminator models
# Charles E Hill
# 21/10/2020
import sys
import torch
from torch import device
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg16_bn

from train_utils import get_device_metrics

# Braided Block and Building blocks

class Braided_Block(nn.Module):

    def __init__(self,inCImg,inCMeta,outC,device=torch.device("cuda:0")):
        super(Braided_Block,self).__init__()
        self.device = device

        self.inNorm = nn.InstanceNorm2d(inCImg)
        self.conv = nn.Conv2d(inCImg,outC,3,padding=1)
        self.bnImg = nn.BatchNorm2d(outC)
        self.conv2 = nn.Conv2d(outC,outC,3,padding=1)
        self.bnImg2 = nn.BatchNorm2d(outC)

        self.scalingFC = nn.Linear(inCMeta+inCImg,1)
        self.bnMeta = nn.BatchNorm1d(inCMeta+inCImg)
        self.fc = nn.Linear(inCMeta+inCImg,outC)

    def mean_n_std(self,tensor):
        mean = tensor.mean(dim=(2,3))
        std = tensor.std(dim=(2,3))
        out = torch.ones((mean.size()[0],mean.size()[1]),device=self.device)
        out *= mean[:,:]
        out *= std[:,:]
        # out.unsqueeze_(1)
        return out

    def forward(self,img,meta):
        x = self.inNorm(img)
        
        mNS = self.mean_n_std(img)
        meta = torch.cat((meta,mNS),dim=1)

        tempMeta = self.scalingFC(meta)
        x *= tempMeta[:,:,None,None] # THIS IS AMAZING: Its how you add or multiply an array of (A,B,C,D) and (A,B)

        x = self.conv(x)
        meta = self.bnMeta(meta)
        meta = self.fc(meta)

        x += meta[:,:,None,None]

        img = F.relu(self.bnImg(x))
        img = F.relu(self.bnImg2(self.conv2(img)))

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

# Braided UNet and Classifier Complete

class Braided_UNet(nn.Module):

    def __init__(self,nCImg,nCMeta,outCImg,outCMeta,xDim=288,yDim=384,device=device):
        super(Braided_UNet,self).__init__()


        self.braidedBlockIn = Braided_Block(nCImg,nCMeta,nCImg,device=device)
        self.down1 = Down_Conv_Braided(nCImg,nCImg,8,device=device)
        self.down2 = Down_Conv_Braided(8,8,16,device=device)
        self.down3 = Down_Conv_Braided(16,16,32,device=device)
        self.down4 = Down_Conv_Braided(32,32,64,device=device)

        self.up1 = Up_Conv_Braided(64+32,64,32,device=device)
        self.up2 = Up_Conv_Braided(32+16,32,16,device=device)
        self.up3 = Up_Conv_Braided(16+8,16,8,device=device)
        self.up4 = Up_Conv_Braided(8+nCImg,8,nCImg,device=device)

        self.outConv1 = nn.Conv2d(nCImg,outCImg,kernel_size=1)
        self.outConv2 = nn.Conv2d(outCImg,outCImg,kernel_size=1)

        self.outfc1 = nn.Linear(nCImg,nCMeta)
        self.outfc2 = nn.Linear(nCMeta,outCMeta)

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
        # print("Meta size 3: ",meta3.size())
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

class Braided_Classifier(nn.Module):

    def __init__(self,inCImg,inCMeta,outC,xDim,yDim,device=torch.device("cuda:0")):
        super(Braided_Classifier,self).__init__()
        self.device = device

        self.bb1 = Braided_Block(inCImg,inCMeta,16,device=self.device)
        self.bb2 = Down_Conv_Braided(16,16,32,device=self.device)
        self.bb3 = Down_Conv_Braided(32,32,64,device=self.device)
        self.bb4 = Down_Conv_Braided(64,64,128,device=self.device)
        self.bb5 = Down_Conv_Braided(128,128,256,device=self.device)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(256*(xDim//16)*(yDim//16)+256,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,256)
        self.fc4 = nn.Linear(256,outC)



    def forward(self,img,meta):

        img,meta = self.bb1(img,meta)
        img,meta = self.bb2(img,meta)
        img,meta = self.bb3(img,meta)
        img,meta = self.bb4(img,meta)
        img,meta = self.bb5(img,meta)

        x = self.flatten(img)
        x = torch.cat((x,meta),dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x

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

# AutoEncoder with central loss:

class Double_Conv(nn.Module):

    def __init__(self,inC,outC):
        super(Double_Conv,self).__init__()

        self.conv1 = nn.Conv2d(inC,outC,3,1,1)
        self.bn1 = nn.BatchNorm2d(outC)
        self.conv2 = nn.Conv2d(outC,outC,3,1,1)
        self.bn2 = nn.BatchNorm2d(outC)

    def forward(self,x):

        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))

        return x

class Down_Conv(nn.Module):

    def __init__(self,inC,outC):
        super(Down_Conv,self).__init__()

        self.down1 = Double_Conv(inC,outC)
        self.mp1 = nn.MaxPool2d(2,2)

    def forward(self,x):

        x = self.mp1(self.down1(x))

        return x

class Encoder(nn.Module):

    def __init__(self,xDim,yDim,inC=7):
        super(Encoder,self).__init__()

        self.down1 = Down_Conv(inC,16)
        self.down2 = Down_Conv(16,32)
        self.down3 = Down_Conv(32,64)
        self.down4 = Down_Conv(64,128)
        self.down5 = Down_Conv(128,256)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(256*(xDim//32)*(yDim//32),1024)

    def forward(self,x):

        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)

        x = self.flatten(x)

        x = self.fc1(x)

        return x

class Up_Conv(nn.Module):

    def __init__(self,inC,outC,kernel=4,stride=2,padding=1):
        super(Up_Conv,self).__init__()

        self.upConv = nn.ConvTranspose2d( inC, outC, kernel, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(outC)

    def forward(self,x):

        x = self.upConv(x)
        x = self.bn(x)
        x = F.leaky_relu(x)
        print("Up conv ",x.size())

        return x

class Decoder(nn.Module):

    def __init__(self,inC=1024,outC=7,xDim=288,yDim=384):
        super(Decoder,self).__init__()

        self.up1 = Up_Conv(inC,256,kernel=(xDim//32,yDim//32),stride=1,padding=0)
        self.up2 = Up_Conv(256,128)
        self.up3 = Up_Conv(128,64)
        self.up4 = Up_Conv(64,32)
        self.up5 = Up_Conv(32,16)
        self.upFinal = nn.ConvTranspose2d(16,outC,4,2,1)

    def forward(self,x):

        x = x.unsqueeze(-1)
        x = x.unsqueeze(-1)

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)

        x = torch.tanh(self.upFinal(x))

        return(x)

class AutoEncoder_W_Central_Loss(nn.Module):

    def __init__(self,inC,xDim,yDim,outC):
        super(AutoEncoder_W_Central_Loss,self).__init__()

        self.encoder = Encoder(xDim,yDim,inC=inC)
        self.decoder = Decoder(outC=outC,xDim=xDim,yDim=yDim)

    def forward(self,x):

        x = self.encoder(x)

        y = self.decoder(x)

        return x, y

# DCGAN_ESQ

class DCGAN_ESG(nn.Module):

    def __init__(self,inMeta=694,inImg=7,outMeta=7,outSize=12,trainMode=True):
        super(DCGAN_ESG,self).__init__()
        self.trainMode = trainMode
        self.meta_arm = Meta_Arm(inMeta,outMeta)
        self.vgg_arm = vgg16_bn(pretrained=True)

        numFtrs = self.vgg_arm.classifier[6].in_features
        self.vgg_arm.classifier[6] = nn.Linear(numFtrs,outSize)
        numChannels = self.vgg_arm.features[0].out_channels
        self.vgg_arm.features[0] = nn.Conv2d(inImg+outMeta, numChannels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self,x_img,x_meta):

        x_meta = self.meta_arm(x_meta)
        x_img = torch.cat([x_img,x_meta],dim=1)

        out = self.vgg_arm(x_img)
        if self.trainMode:
            return out
        else:
            return out,x_meta

class Meta_Arm(nn.Module):

    def __init__(self,inMeta,nC,nGF=16):
        super(Meta_Arm,self).__init__()

        self.convTrans0 = nn.ConvTranspose2d(inMeta,nGF*16, (3**2,4**2), 1, 0, bias=False)
        self.bn0 = nn.BatchNorm2d(nGF * 16)
        
        self.convTrans1 = nn.ConvTranspose2d(nGF*16,nGF*8, 4, (2,3), 1, bias=False)
        self.bn1 = nn.BatchNorm2d(nGF * 8)

        self.convTrans2 = nn.ConvTranspose2d(nGF*8,nGF*4, 4, (2,1), 1, bias=False)
        self.bn2 = nn.BatchNorm2d(nGF * 4)     

        self.convTrans3 = nn.ConvTranspose2d(nGF*4,nGF*2, 4, 2, 1,bias=False)
        self.bn3 = nn.BatchNorm2d(nGF * 2)     

        self.convTrans4 = nn.ConvTranspose2d(nGF*2,nGF, 4, 2 ,1,bias=False)
        self.bn4 = nn.BatchNorm2d(nGF)     

        self.convTrans5 = nn.ConvTranspose2d(nGF,nC, 4, 2, 1, bias=False)

    def forward(self,x):
        x = x.unsqueeze_(2)
        x = x.unsqueeze_(2)

        x = F.leaky_relu(self.bn0(self.convTrans0(x)))
        x = F.leaky_relu(self.bn1(self.convTrans1(x)))
        x = F.leaky_relu(self.bn2(self.convTrans2(x)))
        x = F.leaky_relu(self.bn3(self.convTrans3(x)))
        x = F.leaky_relu(self.bn4(self.convTrans4(x)))

        x = torch.tanh(self.convTrans5(x))

        return x

# VGG_Concat

class Triple_Conv(nn.Module):

    def __init__(self,inC,outC):
        super(Triple_Conv,self).__init__()

        self.conv1 = nn.Conv2d(inC,outC,3,1,1)
        self.bn1 = nn.BatchNorm2d(outC)
        self.conv2 = nn.Conv2d(outC,outC,3,1,1)
        self.bn2 = nn.BatchNorm2d(outC)
        self.conv3 = nn.Conv2d(outC,outC,3,1,1)
        self.bn3 = nn.BatchNorm2d(outC)

    def forward(self,x):

        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))

        return x

class Down_Conv_Triple(nn.Module):

    def __init__(self,inC,outC):
        super(Down_Conv_Triple,self).__init__()

        self.down1 = Triple_Conv(inC,outC)
        self.mp1 = nn.MaxPool2d(2,2)

    def forward(self,x):

        x = self.mp1(self.down1(x))

        return x

class VGG_Concat(nn.Module):

    def __init__(self,inC,inCMeta,outSize,xDim=288,yDim=384):
        super(VGG_Concat,self).__init__()

        self.dConv0 = Down_Conv(inC,64)
        self.dConv1 = Down_Conv(64,128)
        self.dConv2 = Down_Conv_Triple(128,256)
        self.dConv3 = Down_Conv_Triple(256,512)
        self.dConv4 = Down_Conv_Triple(512,512)

        self.flatten = nn.Flatten()

        self.fc0 = nn.Linear(512*(xDim//32)*(yDim//32),4096)
        self.fc1 = nn.Linear(4096+inCMeta,4096)
        self.fc2 = nn.Linear(4096,1000)
        self.fc3 = nn.Linear(1000,outSize)

    def forward(self,x,meta):

        x = self.dConv0(x)
        x = self.dConv1(x)
        x = self.dConv2(x)
        x = self.dConv3(x)
        x = self.dConv4(x)

        x = self.flatten(x)

        x = F.leaky_relu(self.fc0(x))

        x = torch.cat([x,meta],dim=1)

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)

        return x


if __name__ == "__main__":
    deviceMetrics, sD = get_device_metrics(memLimit=3000)
    device = torch.device(sD)

    img = torch.randn((4,7,288,384)).to(device)
    meta = torch.randn((4,694)).to(device)

    net = VGG_Concat(inC=7,inCMeta=694,outSize=12)
    # net = Braided_Classifier(7,694,1,288,384,device=device)

    net.to(device)
    net.eval()

    with torch.no_grad():
        out = net(img,meta)

    print(out.size())