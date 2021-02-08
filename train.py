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
from PyQt5.QtWidgets import QApplication

from models import Braided_AutoEncoder, Braided_UNet, Braided_UNet_Complete
from datasets import T1_Train_Meta_Dataset, T1_Val_Meta_Dataset, T1_Test_Meta_Dataset, Random_Affine, ToTensor, Normalise, collate_fn
from param_gui import Param_GUI
from train_utils import plot_images, plot_images_meta


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
parser.add_argument("--gui",help="Use GUI to pick out parameters (WIP)",default=False,action='store_true',dest="gui")
parser.add_argument("--norm",help="Normalise the data",default=False,action='store_true',dest="normalise")

figPerEpoch = 40


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

if platform.system() == "Linux":
    fileDir = "/home/shug4421/Data/fully_split_data/"
    t1MapDir = "/home/shug4421/Data/T1_Maps/"


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

datasetTrain = T1_Train_Meta_Dataset(modelDir=modelDir,fileDir=fileDir,t1MapDir=t1MapDir,size=10000,transform=trnsInTrain,load=load)
datasetVal = T1_Val_Meta_Dataset(modelDir=modelDir,fileDir=fileDir,t1MapDir=t1MapDir,size=1000,transform=trnsInVal,load=load)
datasetTest = T1_Test_Meta_Dataset(modelDir=modelDir,fileDir=fileDir,t1MapDir=t1MapDir,size=1000,transform=trnsInVal,load=load)

loaderTrain = DataLoader(datasetTrain,batch_size=bSize,shuffle=True,collate_fn=collate_fn,pin_memory=False)
loaderVal = DataLoader(datasetVal,batch_size=bSize,shuffle=False,collate_fn=collate_fn,pin_memory=False)
loaderTest = DataLoader(datasetTest,batch_size=bSize,shuffle=False,collate_fn=collate_fn,pin_memory=False)

# testBatch = next(iter(loaderTrain))
# inpData = testBatch["Images"]
# print(inpData.size())
# plt.imshow(inpData[0,0,:,:].numpy())
# plt.show()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# netB = Braided_UNet(7,7,288,384,device=device)
netB = Braided_UNet_Complete(7,7,1,1,device=device)

netB = netB.to(device)

# for param in netB.parameters():
#     print(param)

loss1 = nn.SmoothL1Loss()
w1 = 1000
loss3 = nn.SmoothL1Loss()
w3 = 1
optimB = optim.Adam(netB.parameters(),lr=lr,betas=(b1,0.999))

lrSchedulerG = torch.optim.lr_scheduler.StepLR(optimB,step_size=stepSize,gamma=0.1,verbose=True)

trainLen = datasetTrain.__len__()
valLen = datasetVal.__len__()

lowestLoss = 1000000000.0
lossArr = np.zeros((numEpochs,trainLen,2))
for nE in range(numEpochs):
    print("\nEpoch [{}/{}]".format(nE+1,numEpochs))
    print("\nTraining:")
    runningLoss = 0.0
    for i,data in enumerate(loaderTrain):

        inpData = data["Images"].to(device)
        inpInvTime = data["InvTime"].type(torch.FloatTensor)
        inpInvTime = inpInvTime.to(device)
        outGT = data["T1Map"].to(device)
        eid = data["eid"]

        

        optimB.zero_grad()
        netB.zero_grad()

        outImg, outMeta = netB(inpData,inpInvTime)

        err1 = loss1(inpData,outImg) * w1
        err3 = loss3(inpInvTime,outMeta) * w3

        lossArr[nE,i,0] = i+nE*trainLen
        lossArr[nE,i,1] = err1.item() + err3.item()
        runningLoss += err1.item() + err3.item()

        err = err1 + err3
        err.backward()
        optimB.step()

        # inpData = inpData.cpu().numpy()[0,0,:,:]
        # plt.imshow(inpData)
        # plt.show()
        
        if i % (trainLen // (bSize*figPerEpoch)) == 0:
            outImg = outImg.detach().cpu().numpy()[0,:,:,:]
            inpData = inpData.cpu().numpy()[0,:,:,:]

            outMeta = outMeta.detach().cpu().numpy()[0]
            inpInvTime = inpInvTime.cpu().numpy()[0]
            
            plot_images_meta(inpData,outImg,np.array([inpInvTime,outMeta]),figDir,nE,i)

            plt.figure()
            plt.plot(lossArr[nE,:i,0],lossArr[nE,:i,1])
            plt.savefig("{}Epoch_{}_i_{}_loss.png".format(figDir,nE+1,i+1))
            plt.close("all")

        sys.stdout.write("\r\tSubj {}/{}: Loss = {:.4f}".format(i*bSize,trainLen,runningLoss/((i+1)*4)))

    valLoss = []
    with torch.no_grad():
        print("\nValidation:")
        for i,data in enumerate(loaderVal):
            sys.stdout.write("\r\tSubj {}/{}".format(i*bSize,valLen))

            inpData = data["Images"].to(device)
            inpInvTime = data["InvTime"].type(torch.FloatTensor)
            inpInvTime = inpInvTime.to(device)
            outGT = data["T1Map"].to(device)

            netB.zero_grad()

            outImg, outMeta = netB(inpData,inpInvTime)
            err1 = loss1(inpData,outImg)
            err3 = loss1(inpInvTime,outMeta)

            valLoss.append(err1.item() + err3.item())

            if i % (valLen // (bSize*3)) == 0:
                outImg = outImg.detach().cpu().numpy()[0,:,:,:]
                inpData = inpData.cpu().numpy()[0,:,:,:]

                outMeta = outMeta.detach().cpu().numpy()[0]
                inpInvTime = inpInvTime.cpu().numpy()[0]
                print("\n Output inversion times: {}, input inversion times: {}".format(outMeta,inpInvTime))

                plot_images(inpData,outImg,np.array([inpInvTime,outMeta]),figDir,nE,i,val=True)

                x = np.arange(1,8)
                ax = plt.subplot(111)
                ax.bar(x-0.2, outMeta, width=0.2, color='b', align='center')
                ax.bar(x+0.2, inpInvTime, width=0.2, color='r', align='center')
                plt.savefig("{}Epoch_{}_i_{}_InvTime_val.png".format(figDir,nE+1,i+1))
                plt.close("all")



        valLoss = sum(valLoss)/valLen
        print("\n\tVal Loss: {}".format(valLoss))

        if valLoss < lowestLoss:
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


