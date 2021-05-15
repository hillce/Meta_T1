import os, sys, json
import argparse
import shutil

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.nn.init import xavier_uniform_, zeros_

from models import Braided_Classifier
from datasets import Train_Meta_Dataset, Val_Meta_Dataset, Test_Meta_Dataset, ToTensor, Normalise, collate_fn

# Arg parser so I can test out different model parameters
parser = argparse.ArgumentParser(description="Training program for T1 map generation")
parser.add_argument("--dir",help="File directory for numpy images",type=str,default="C:/fully_split_data/",dest="fileDir")
parser.add_argument("--t1dir",help="File directory for T1 matfiles",type=str,default="C:/T1_Maps/",dest="t1MapDir")
parser.add_argument("--model_name",help="Name for saving the model",type=str,default="Debug",dest="modelName")
parser.add_argument("--load",help="Load the preset trainSets, or redistribute (Bool)",default=False,action='store_true',dest="load")
parser.add_argument("-lr",help="Learning rate for the optimizer",type=float,default=1e-3,dest="lr")
parser.add_argument("-b1",help="Beta 1 for the Adam optimizer",type=float,default=0.5,dest="b1")
parser.add_argument("-bSize","--batch_size",help="Batch size for dataloader",type=int,default=4,dest="batchSize")
parser.add_argument("-nE","--num_epochs",help="Number of Epochs to train for",type=int,default=50,dest="numEpochs")
parser.add_argument("--step_size",help="Step size for learning rate decay",type=int,default=5,dest="stepSize")
parser.add_argument("--norm",help="Normalise the data",default=False,action='store_true',dest="normalise")
parser.add_argument("--con","-condense",help="Whether to condense Tags to single include/exclude",default=False,action='store_true',dest="condense")

args = parser.parse_args()

fileDir = args.fileDir
t1MapDir = args.t1MapDir
modelName = args.modelName
load = args.load
lr = args.lr
b1 = args.b1
bSize = args.batchSize
numEpochs = args.numEpochs
stepSize = args.stepSize
normalise = args.normalise
condense = args.condense

modelDir = "./models/Braided_Net/{}/".format(modelName)

if modelName == "Debug":
    try:
        shutil.rmtree(modelDir)
    except:
        pass

os.makedirs(modelDir)

hParamDict = {}
hParamDict["fileDir"] = fileDir
hParamDict["t1MapDir"] = t1MapDir
hParamDict["modelName"] = modelName
hParamDict["load"] = load
hParamDict["lr"] = lr
hParamDict["b1"] = b1
hParamDict["batchSize"] = bSize
hParamDict["numEpochs"] = numEpochs
hParamDict["stepSize"] = stepSize
hParamDict["normalise"] = normalise

with open("{}hparams.json".format(modelDir),"w") as f:
    json.dump(hParamDict,f)

figDir = "{}Training_Figures/".format(modelDir)
os.makedirs(figDir)

# Make writer
writer = SummaryWriter("{}tensorboard".format(modelDir))

# rA = Random_Affine(degreesRot=5,trans=(0.01,0.01),shear=5)
toT = ToTensor()

if normalise:
    norm = Normalise()
    trnsInTrain = transforms.Compose([toT,norm])
    trnsInVal = transforms.Compose([toT,norm])
else:
    trnsInTrain = transforms.Compose([toT])
    trnsInVal = transforms.Compose([toT])

datasetTrain = Train_Meta_Dataset(modelDir=modelDir,fileDir=fileDir,t1MapDir=t1MapDir,size=20000,transform=trnsInTrain,load=load,condense=condense)
datasetVal = Val_Meta_Dataset(modelDir=modelDir,fileDir=fileDir,t1MapDir=t1MapDir,size=5000,transform=trnsInVal,load=load,condense=condense)
datasetTest = Test_Meta_Dataset(modelDir=modelDir,fileDir=fileDir,t1MapDir=t1MapDir,size=5000,transform=trnsInVal,load=load,condense=condense)

loaderTrain = DataLoader(datasetTrain,batch_size=bSize,shuffle=True,collate_fn=collate_fn,pin_memory=False)
loaderVal = DataLoader(datasetVal,batch_size=bSize,shuffle=False,collate_fn=collate_fn,pin_memory=False)
loaderTest = DataLoader(datasetTest,batch_size=bSize,shuffle=False,collate_fn=collate_fn,pin_memory=False)

batch = next(iter(loaderTrain))
inpMetaSize = batch['Meta'].size()[1]
if condense:
    outSize = 1
else:
    outSize = batch["Tag"].size()[1]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
netB = Braided_Classifier(7,inpMetaSize,outSize,xDim=288,yDim=384,device=device)
netB = netB.to(device)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier_uniform_(m.weight.data)
        if m.bias.data is not None:
            zeros_(m.bias.data)
    if isinstance(m, nn.Linear):
        xavier_uniform_(m.weight.data)
        if m.bias.data is not None:
            zeros_(m.bias.data)   

netB.apply(weights_init)

loss = nn.BCEWithLogitsLoss()
optimB = optim.Adam(netB.parameters(),lr=lr,betas=(b1,0.999))

lrScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimB,"min",verbose=True)

trainLen = datasetTrain.__len__()
valLen = datasetVal.__len__()

lowestLoss = 1000000000.0
minMeta = False
trainLossCnt = 0
valLossCnt = 0

for nE in range(numEpochs):
    print("\nEpoch [{}/{}]".format(nE+1,numEpochs))
    print("\nTraining:")
    runningLoss = 0.0
    netB.train()
    for i,data in enumerate(loaderTrain):

        inpData = data["Images"].to(device)
        eid = data["eid"]
        inpMeta = data["Meta"].to(device)
        outTag = data["Tag"].to(device)

        optimB.zero_grad()
        netB.zero_grad()

        out = netB(inpData,inpMeta)

        err = loss(out,outTag)
        runningLoss += err.item()

        err.backward()
        optimB.step()

        sys.stdout.write("\r\tSubj {}/{}: Loss = {:.5f}".format(i*bSize,trainLen,runningLoss/((i+1)*bSize)))

        writer.add_scalar('Loss/train',err.item(),trainLossCnt)

        trainLossCnt += 1
 
    valLoss = 0.0
    with torch.no_grad():
        print("\nValidation:")
        netB.eval()
        for i,data in enumerate(loaderVal):

            inpData = data["Images"].to(device)
            eid = data["eid"]
            inpMeta = data["Meta"].to(device)
            outTag = data["Tag"].to(device)

            out = netB(inpData,inpMeta)

            for j in range(out.shape[0]):
                if np.isnan(out[j][0].item()):
                    for k in range(out.shape[1]):
                        out[j][k] = 1.0

            err = loss(out,outTag)

            if np.isnan(err.item()):
                valLoss += 100.0
            elif err.item() > 100.0:
                valLoss += 100.0
            else:
                valLoss += err.item()

            writer.add_scalar("Loss/val",err.item(),valLossCnt)

            sys.stdout.write("\r\tSubj {}/{}".format(i*bSize,valLen))

            valLossCnt += 1

        valLoss = valLoss/valLen

        print("\n\tVal Loss: {}".format(valLoss))

        lrScheduler.step(valLoss)

        if valLoss < lowestLoss:
            print("\n Saving Model!")
            torch.save({"Epoch":nE+1,
            "state_dict":netB.state_dict(),
            "optimizer":optimB.state_dict(),
            "lr_scheduler":lrScheduler.state_dict()
            },"{}model.pt".format(modelDir))
            lowestLoss = valLoss
        else:
            print("\n Val Loss {:.5f} > {:.5f} lowest Loss".format(valLoss,lowestLoss))

        
        torch.save({"Epoch":nE+1,
        "state_dict":netB.state_dict(),
        "optimizer":optimB.state_dict(),
        "lr_scheduler":lrScheduler.state_dict()
        },"{}model_latest.pt".format(modelDir))

    print("-"*50)