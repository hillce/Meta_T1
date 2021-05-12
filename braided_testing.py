import os, sys, json
import argparse
import platform
import shutil

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
from scipy.io import loadmat
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.nn.init import xavier_uniform_, zeros_
from PyQt5.QtWidgets import QApplication

from models import Braided_Classifier
from datasets import Train_Meta_Dataset, Val_Meta_Dataset, Test_Meta_Dataset, Random_Affine, ToTensor, Normalise, collate_fn
from meta_function import meta_loading

# Arg parser so I can test out different model parameters
parser = argparse.ArgumentParser(description="Training program for T1 map generation")
parser.add_argument("--dir",help="File directory for numpy images",type=str,default="C:/fully_split_data/",dest="fileDir")
parser.add_argument("--t1dir",help="File directory for T1 matfiles",type=str,default="C:/T1_Maps/",dest="t1MapDir")
parser.add_argument("--model_name",help="Name for saving the model",type=str,default="Debug",dest="modelName")
# parser.add_argument("--load",help="Load the preset trainSets, or redistribute (Bool)",default=True,action='store_true',dest="load")
# parser.add_argument("-lr",help="Learning rate for the optimizer",type=float,default=1e-3,dest="lr")
# parser.add_argument("-b1",help="Beta 1 for the Adam optimizer",type=float,default=0.5,dest="b1")
parser.add_argument("-bSize","--batch_size",help="Batch size for dataloader",type=int,default=4,dest="batchSize")
# parser.add_argument("-nE","--num_epochs",help="Number of Epochs to train for",type=int,default=50,dest="numEpochs")
# parser.add_argument("--step_size",help="Step size for learning rate decay",type=int,default=5,dest="stepSize")
parser.add_argument("--norm",help="Normalise the data",default=False,action='store_true',dest="normalise")

args = parser.parse_args()

fileDir = args.fileDir
t1MapDir = args.t1MapDir
modelName = args.modelName
bSize = args.batchSize
normQ = args.normalise

modelDir = "./TrainingLogs/{}/".format(modelName)
assert os.path.isdir(modelDir), "Model Directory is not found, please check your model name!"


figDir = "{}Test_Figures/".format(modelDir)
try:
    os.makedirs(figDir)
except FileExistsError as e:
    print(e, "This means you will be overwriting previous results!")


toT = ToTensor()

if normQ:
    norm = Normalise()
    trnsInTest = transforms.Compose([toT,norm])
else:
    trnsInTest = transforms.Compose([toT])

datasetTest = Test_Meta_Dataset(modelDir=modelDir,fileDir=fileDir,t1MapDir=t1MapDir,size=1000,transform=trnsInTest,load=True)
loaderTest = DataLoader(datasetTest,batch_size=bSize,shuffle=False,collate_fn=collate_fn,pin_memory=False)

batch = next(iter(loaderTest))
inpMetaSize = batch['Meta'].size()[1]
outSize = batch["Tag"].size()[1]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
netB = Braided_Classifier(7,inpMetaSize,outSize,xDim=288,yDim=384,device=device)
modelDict = torch.load("{}model.pt".format(modelDir))
netB.load_state_dict(modelDict["state_dict"])
netB = netB.to(device)


testLen = datasetTest.__len__()

testLossCnt = 0

sig = torch.sigmoid()

pred = np.zeros((1000,outSize))

with torch.no_grad():
    netB.eval()
    for i,data in enumerate(loaderTest):
        inpData = data["Images"].to(device)
        eid = data["eid"]
        inpMeta = data["Meta"].to(device)
        outTag = data["Tag"].to(device)

        out = netB(inpData,inpMeta)

        out = sig(out)

        pred[i*bSize:(i*bSize),:] = out.cpu().numpy()

        sys.stdout.write("\r\tSubj {}/{}".format(i*bSize,testLen))
 






