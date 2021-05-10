import os, copy, json, sys, re, warnings

import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms as transforms

from meta_function import meta_loading

class Base_Dataset(Dataset):

    def __init__(self,modelDir,fileDir="C:/fully_split_data/",t1MapDir="C:/T1_Maps/",transform=None,size=2000):
        self.modelDir = modelDir
        self.fileDir = fileDir
        self.t1MapDir = t1MapDir
        self.transform = transform
        self.size = size
        self.metaData = meta_loading()

    def get_itm(self,index,dataset):

        inpData = np.load("{}{}_20204_2_0.npy".format(self.fileDir,dataset[index]))
        inpDataInvTime = np.load("{}{}_20204_2_0_inv_times.npy".format(self.fileDir,dataset[index]))
        inpMeta = self.metaData[dataset[index]]

        try:
            outGT = loadmat("{}{}_20204_2_0.mat".format(self.t1MapDir,dataset[index]))['results']
        except:
            outGT = loadmat("{}{}_20204_2_0.mat".format(self.t1MapDir,dataset[index]))['x']

        sample = {"Images":inpData,"T1Map":outGT}

        if self.transform:
            sample = self.transform(sample)

        sample = {"Images":sample["Images"],"T1Map":sample["T1Map"],"InvTime":inpDataInvTime,"eid":dataset[index],"Meta":inpMeta}
        return sample

    def trim_meta(self,dataset):
        metaCopy = copy.deepcopy(self.metaData)
        for k in metaCopy.keys():
            if k not in dataset.keys():
                del self.metaData[k]

        del metaCopy

class Train_Meta_Dataset(Base_Dataset):

    def __init__(self,modelDir,size=2000,fileDir="C:/fully_split_data/",t1MapDir="C:/T1_Maps/",load=True,transform=None):
        Base_Dataset.__init__(self,modelDir,fileDir=fileDir,t1MapDir=t1MapDir,transform=transform,size=size)

        if not load:
            subjList = [x[:7] for x in os.listdir(self.fileDir) if x.endswith("_2_0.npy") if os.path.isfile("{}{}_20204_2_0.mat".format(self.t1MapDir,x[:7]))]
            self.trainSet = np.random.choice(subjList,self.size)
            np.save("{}trainSet.npy".format(self.modelDir),self.trainSet)
        else:
            self.trainSet = np.load("trainSet.npy".format(modelDir))
            np.save("{}trainSet.npy".format(modelDir),self.trainSet)

        self.trim_meta(self.trainSet)

    def __getitem__(self, index):
        return self.get_itm(index,self.trainSet)

    def __len__(self):
        return len(self.trainSet)

class Val_Meta_Dataset(Base_Dataset):

    def __init__(self,modelDir,size=2000,fileDir="C:/fully_split_data/",t1MapDir="C:/T1_Maps/",load=True,transform=None):
        Base_Dataset.__init__(self,modelDir,fileDir=fileDir,t1MapDir=t1MapDir,transform=transform,size=size)

        if not load:
            subjList = [x[:7] for x in os.listdir(self.fileDir) if x.endswith("_2_0.npy") if os.path.isfile("{}{}_20204_2_0.mat".format(self.t1MapDir,x[:7]))]
            self.trainSet = np.load("{}trainSet.npy".format(self.modelDir))
            subjList = [x for x in subjList if x not in self.trainSet]
            self.valSet = np.random.choice(subjList,self.size)
            np.save("{}valSet.npy".format(self.modelDir),self.valSet)
        else:
            self.valSet = np.load("valSet.npy".format(self.modelDir))
            np.save("{}valSet.npy".format(self.modelDir),self.valSet)

        self.trim_meta(self.valSet)

    def __getitem__(self, index):
        return self.get_itm(index,self.valSet)

    def __len__(self):
        return len(self.valSet)

class Test_Meta_Dataset(Base_Dataset):

    def __init__(self,modelDir,size=2000,fileDir="C:/fully_split_data/",t1MapDir="C:/T1_Maps/",load=True,transform=None):
        Base_Dataset.__init__(self,modelDir,fileDir=fileDir,t1MapDir=t1MapDir,transform=transform,size=size)

        if not load:
            subjList = [x[:7] for x in os.listdir(self.fileDir) if x.endswith("_2_0.npy") if os.path.isfile("{}{}_20204_2_0.mat".format(self.t1MapDir,x[:7]))]
            self.trainSet = np.load("{}trainSet.npy".format(self.modelDir))
            self.valSet = np.load("{}valSet.npy".format(self.modelDir))
            subjList = [x for x in subjList if x not in self.trainSet and x not in self.valSet]
            self.testSet = np.random.choice(subjList,self.size)
            np.save("{}testSet.npy".format(self.modelDir),self.testSet)
        else:
            self.testSet = np.load("testSet.npy".format(self.modelDir))
            np.save("{}testSet.npy".format(self.modelDir),self.testSet)

        self.trim_meta(self.testSet)

    def __getitem__(self, index):
        return self.get_itm(index,self.testSet)

    def __len__(self):
        return len(self.testSet)

class Random_Affine(object):

    def __init__(self,degreesRot=10,trans=(0.1,0.1),shear=10):

        self.rA = transforms.RandomAffine(degreesRot,translate=trans,shear=shear)

    def __call__(self,sample):
        inpData = sample["Images"]
        outGT = sample["T1Map"]

        allData = torch.cat((inpData,outGT))

        images = self.rA(allData)

        sample = {"Images":images[:7,:,:],"T1Map":images[7,:,:].unsqueeze_(0)}
        return sample

class ToTensor(object):
    """ convert ndarrays in sample to Tensors"""

    def __call__(self,sample):
        inpData = sample["Images"]
        outGT = sample["T1Map"]

        inpData = np.transpose(inpData,axes=(2,0,1))
        inpData = torch.from_numpy(inpData).float() 
        
        outGT = outGT[:,:,0]
        outGT = torch.from_numpy(outGT).float()
        outGT.unsqueeze_(0)

        sample = {"Images":inpData,"T1Map":outGT}
        return sample

class Normalise(object):

    def __init__(self):

        mean = np.load("mean_7Ch.npy")
        std = np.load("std_7Ch.npy")

        self.normImg = transforms.Normalize(mean,std,inplace=True)
        self.normT1 = transforms.Normalize([362.66540459],[501.85027392],inplace=True)

    def __call__(self,sample):
        inpData = sample["Images"]
        outGT = sample["T1Map"]

        sample = {"Images":self.normImg(inpData),"T1Map":self.normT1(outGT)}
        return sample

def collate_fn(sampleBatch):
    eid = [item['eid'] for item in sampleBatch]

    inpData = [item['Images'].unsqueeze_(0) for item in sampleBatch]
    try:
        inpData = torch.cat(inpData)
    except RuntimeError:
        for i,(item,e) in enumerate(zip(inpData,eid)):
            if item.size() != torch.Size([1,7,288,384]):
                inpData[i] = torch.randn((1,7,288,384))
        inpData = torch.cat(inpData)

    outGT = [item['T1Map'].unsqueeze_(0) for item in sampleBatch]
    try:
        outGT = torch.cat(outGT)
    except RuntimeError:
        for i,(item,e) in enumerate(zip(outGT,eid)):
            if item.size() != torch.Size([1,1,288,384]):
                outGT[i] = torch.randn((1,1,288,384))
        outGT = torch.cat(outGT)

    invTimes = [item['InvTime'] for item in sampleBatch]
    invTimes = torch.tensor(invTimes)

    metaData = [item["Meta"] for item in sampleBatch]
    metaData = torch.tensor(metaData)

    sample = {"Images":inpData,"T1Map":outGT,"InvTime":invTimes,"eid":eid,"Meta":metaData}
    return sample
