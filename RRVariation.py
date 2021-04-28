import os, sys
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from models import Braided_Classifier

def RR_variation(invTime):
    rr_intervals = np.zeros(len(invTime))

    for i in range(len(invTime)):
        if i < 4:
            rr_intervals[i] = invTime[i+1] - invTime[i]
        else:
            rr_intervals[i] = np.mean(rr_intervals[0:4])

    return rr_intervals

def RR_test(rrInt):

    boolArr = rrInt[rrInt < 600]
    if boolArr.any():
        return 0
        
    boolArr = rrInt[rrInt > 1300]
    if boolArr.any():
        return 0

    maxRR = np.max(rrInt)
    minRR = np.min(rrInt)

    if maxRR > 1.5*minRR:
        return 0

    return 1

dataDir = "/home/shug4421/Data/fully_split_data/"
invTimes = [os.path.join(dataDir,x) for x in os.listdir(dataDir) if x.endswith("_inv_times.npy")]

device = torch.device("cuda:0")

net = Braided_Classifier(1,7,1,128,128,device=device)
net.to(device)

loss = nn.BCELoss()
loss.to(device)
opt = optim.Adam(net.parameters(),lr=1e-6)

for epoch in range(5):
    print("#"*50)
    print("Epoch {}".format(epoch))
    print("#"*50)
    print("Validation: ")

    with torch.no_grad():
        valAcc = []
        # net.eval()
        for i,fN in enumerate(invTimes[1000:2000]):
            invTime = np.load(fN)
            rrInt = RR_variation(invTime)
            rrTest = RR_test(rrInt)

            outGT = torch.tensor([float(rrTest)])
            outGT.unsqueeze_(0)
            outGT = outGT.to(device)

            inImg = torch.randn((1,1,128,128))
            inMeta = torch.from_numpy(invTime)
            inMeta = inMeta.float()
            inMeta.unsqueeze_(0)

            inImg = inImg.to(device)
            inMeta = inMeta.to(device)

            out = net(inImg,inMeta)

            if out[0] > 0.5:
                if outGT[0] == 1.0:
                    valAcc.append(1)
                else:
                    valAcc.append(0)
            else:
                if outGT[0] == 0.0:
                    valAcc.append(1)
                else:
                    valAcc.append(0)

            sys.stdout.write("\r[{}/{}] RRTest: {} Predicted: {:.2f} Acc: {:.3f}".format(i,len(invTimes),rrTest,out.cpu().numpy()[0,0],np.mean(valAcc)))

    acc = []
    print("\nTraining: ")
    net.train()
    for i,fN in enumerate(invTimes[0:1000]):
        opt.zero_grad()

        invTime = np.load(fN)
        rrInt = RR_variation(invTime)
        rrTest = RR_test(rrInt)

        outGT = torch.tensor([float(rrTest)])
        outGT.unsqueeze_(0)
        outGT = outGT.to(device)

        inImg = torch.zeros((1,1,128,128))
        inMeta = torch.from_numpy(invTime)
        inMeta = inMeta.float()
        inMeta.unsqueeze_(0)

        inImg = inImg.to(device)
        inMeta = inMeta.to(device)

        out = net(inImg,inMeta)

        if not rrTest:
            w = 10
        else:
            w = 1

        err = w*loss(out,outGT)

        err.backward()

        opt.step()

        if out[0] > 0.5:
            if outGT[0] == 1.0:
                acc.append(1)
            else:
                acc.append(0)
        else:
            if outGT[0] == 0.0:
                acc.append(1)
            else:
                acc.append(0)

        sys.stdout.write("\r[{}/{}] RRTest: {} Loss: {:.4f} Acc: {:.3f}".format(i,len(invTimes),rrTest,err.item(),np.mean(acc)))


