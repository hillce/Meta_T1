import os, sys, json
import argparse

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
from scipy.io import loadmat
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PyQt5.QtWidgets import QApplication

from models import Braided_AutoEncoder, Braided_UNet, Braided_UNet_Complete
from datasets import T1_Train_Meta_Dataset, T1_Val_Meta_Dataset, T1_Test_Meta_Dataset, Random_Affine, ToTensor, Normalise, collate_fn
from param_gui import Param_GUI
from train_utils import plot_images, plot_images_meta


# Arg parser so I can test out different model parameters
parser = argparse.ArgumentParser(description="Training program for T1 map generation")
parser.add_argument("--dir",help="File directory for numpy images",type=str,default="C:/fully_split_data/",dest="fileDir")
parser.add_argument("--t1dir",help="File directory for T1 matfiles",type=str,default="C:/T1_Maps/",dest="t1MapDir")
parser.add_argument("--model_name",help="Name for saving the model",type=str,dest="modelName",required=True)
parser.add_argument("--load",help="Load the preset trainSets, or redistribute (Bool)",default=False,action='store_true',dest="load")
parser.add_argument("-lr",help="Learning rate for the optimizer",type=float,default=1e-3,dest="lr")
parser.add_argument("-b1",help="Beta 1 for the Adam optimizer",type=float,default=0.5,dest="b1")
parser.add_argument("-bSize","--batch_size",help="Batch size for dataloader",type=int,default=4,dest="batchSize")
parser.add_argument("-nE","--num_epochs",help="Number of Epochs to train for",type=int,default=50,dest="numEpochs")
parser.add_argument("--step_size",help="Step size for learning rate decay",type=int,default=5,dest="stepSize")
parser.add_argument("--gui",help="Use GUI to pick out parameters (WIP)",default=False,action='store_true',dest="gui")
parser.add_argument("--norm",help="Normalise the data",default=False,action='store_true',dest="normalise")

args = parser.parse_args()

if args.gui:
    # Overrides the argparse parameters
    models = os.listdir("./TrainingLogs/")

    app = QApplication(sys.argv)
    main_win = Param_GUI()
    main_win.show()
    app.exec_()

    newModels = os.listdir("./TrainingLogs/")

    for mod in models:
        newModels.remove(mod)
    
    modelDir = "./TrainingLogs/{}/".format(newModels[0])
    dirHParam = "{}hparams.json".format(modelDir)

    with open(dirHParam,'r') as f:
        hParamDict = json.load(f)

    fileDir = hParamDict["fileDir"]
    t1MapDir = hParamDict["t1MapDir"]
    modelName = hParamDict["modelName"]
    load = hParamDict["load"]
    lr = hParamDict["lr"]
    b1 = hParamDict["b1"]
    bSize = hParamDict["batchSize"]
    numEpochs = hParamDict["numEpochs"]
    stepSize = hParamDict["stepSize"]
    normalise = hParamDict["normalise"]

else:

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

    modelDir = "./TrainingLogs/{}/".format(modelName)

    if modelName == "Temp":
        try:
            os.unlink(modelDir)
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

# rA = Random_Affine(degreesRot=5,trans=(0.01,0.01),shear=5)
toT = ToTensor()
norm = Normalise()

if normalise:
    trnsInTrain = transforms.Compose([toT,norm])
    trnsInVal = transforms.Compose([toT,norm])
else:
    trnsInTrain = transforms.Compose([toT])
    trnsInVal = transforms.Compose([toT])

datasetTrain = T1_Train_Meta_Dataset(modelDir,fileDir=fileDir,t1MapDir=t1MapDir,size=10000,transform=trnsInTrain,load=load)
datasetVal = T1_Val_Meta_Dataset(modelDir,fileDir=fileDir,t1MapDir=t1MapDir,size=1000,transform=trnsInVal,load=load)
datasetTest = T1_Test_Meta_Dataset(modelDir,fileDir=fileDir,t1MapDir=t1MapDir,size=1000,transform=trnsInVal,load=load)

loaderTrain = DataLoader(datasetTrain,batch_size=bSize,shuffle=True,collate_fn=collate_fn,pin_memory=False)
loaderVal = DataLoader(datasetVal,batch_size=bSize,shuffle=False,collate_fn=collate_fn,pin_memory=False)
loaderTest = DataLoader(datasetTest,batch_size=bSize,shuffle=False,collate_fn=collate_fn,pin_memory=False)

with open("./jsonFiles/bmi.json") as f:
    bmis = json.load(f)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
netB = Braided_UNet_Complete(7,7,1,1,device=device)
netB = netB.to(device)

loss1 = nn.SmoothL1Loss()
w1 = 1
loss3 = nn.SmoothL1Loss()
w3 = 1
optimB = optim.Adam(netB.parameters(),lr=lr,betas=(b1,0.999))

lrSchedulerG = torch.optim.lr_scheduler.StepLR(optimB,step_size=stepSize,gamma=0.1,verbose=True)

trainLen = datasetTrain.__len__()
valLen = datasetVal.__len__()

lowestLoss = 1000000000.0
trainLossArr = []
valLossArr = []
missingBMIS = set()
for nE in range(numEpochs):
    print("\nEpoch [{}/{}]".format(nE+1,numEpochs))
    print("\nTraining:")
    runningLoss = 0.0
    epochLossArr = []
    # netB.train()
    for i,data in enumerate(loaderTrain):
        try:
            inpData = data["Images"].to(device)
            inpInvTime = data["InvTime"].type(torch.FloatTensor)
            inpInvTime = inpInvTime.to(device)
            outGT = data["T1Map"].to(device)
            eids = data["eid"]

            bmi = [bmis[x] for x in eids]
            bmi = torch.tensor(bmi,device=device)
            bmi.unsqueeze_(1)

            inpInvTime = torch.zeros(inpInvTime.size(),device=device)

            optimB.zero_grad()
            netB.zero_grad()

            outImg, outMeta = netB(inpData,inpInvTime)

            err1 = loss1(outGT,outImg) * w1
            err3 = loss3(bmi,outMeta) * w3

            epochLossArr.append(err1.item())# + err3.item())
            runningLoss += err1.item()# + err3.item()

            err = err1 + err3
            err.backward()
            optimB.step()

            if i % (trainLen // (bSize*4)) == 0:
                outImg = outImg.detach().cpu().numpy()[0,:,:,:]
                inpData = inpData.cpu().numpy()[0,:,:,:]
                outGT = outGT.cpu().numpy()[0,:,:,:]

                outMeta = outMeta.detach().cpu().numpy()[0]
                bmi = bmi.cpu().numpy()[0]
                
                plot_images_meta(outGT,outImg,np.array([bmi,outMeta]),figDir,nE,i)

                if i != 0:
                    plt.figure()
                    plt.plot(epochLossArr)
                    plt.savefig("{}Epoch_{}_i_{}_loss.png".format(figDir,nE+1,i+1))
                plt.close("all")
        except KeyError as e:
            missingBMIS.add(str(e))

        sys.stdout.write("\r\tSubj {}/{}: Loss = {:.4f}".format(i*bSize,trainLen,runningLoss/((i+1)*4)))


    trainLossArr.append(epochLossArr)
    epochLossArr = []
    with torch.no_grad():
        print("\nValidation:")
        # netB.eval()
        for i,data in enumerate(loaderVal):
            try:
                inpData = data["Images"].to(device)
                inpInvTime = data["InvTime"].type(torch.FloatTensor)
                inpInvTime = inpInvTime.to(device)
                outGT = data["T1Map"].to(device)
                eids = data["eid"]

                bmi = [bmis[x] for x in eids]
                bmi = torch.tensor(bmi,device=device)
                bmi.unsqueeze_(1)

                inpInvTime = torch.zeros(inpInvTime.size(),device=device)

                outImg, outMeta = netB(inpData,inpInvTime)

                err1 = loss1(outGT,outImg)
                # err3 = loss3(bmi,outMeta)

                epochLossArr.append(err1.item())# + err3.item())

                if i % (valLen // (bSize*3)) == 0:
                    outImg = outImg.detach().cpu().numpy()[0,:,:,:]
                    inpData = inpData.cpu().numpy()[0,:,:,:]
                    outGT = outGT.cpu().numpy()[0,:,:,:]

                    outMeta = outMeta.detach().cpu().numpy()[0]
                    bmi = bmi.cpu().numpy()[0]
                    
                    plot_images_meta(outGT,outImg,np.array([bmi,outMeta]),figDir,nE,i,val=True)

            except KeyError as e:
                missingBMIS.add(str(e))

            sys.stdout.write("\r\tSubj {}/{}".format(i*bSize,valLen))

        valLossArr.append(epochLossArr)
        valLoss = sum(epochLossArr)/valLen
        print("\n\tVal Loss: {}".format(valLoss))


        if valLoss < lowestLoss:
            print("\n\tSaving Model!")
            torch.save({"Epoch":nE+1,
            "Generator_state_dict":netB.state_dict(),
            "Generator_loss_function1":loss1.state_dict(),
            "Generator_loss_function2":loss3.state_dict(),
            "Generator_optimizer":optimB.state_dict(),
            "Generator_lr_scheduler":lrSchedulerG.state_dict()
            },"{}model.pt".format(modelDir))
            lowestLoss = valLoss

    lrSchedulerG.step()
    print("-"*50)

    with open("{}Training_Loss.json".format(modelDir),"w") as f:
        json.dump(trainLossArr,f)

    with open("{}Val_Loss.json".format(modelDir),"w") as f:
        json.dump(valLossArr,f)

    with open("{}Missing_BMIs.json".format(modelDir),"w") as f:
        json.dump(list(missingBMIS),f)


