## Script for generating dicom header meta values for whole of biobank
## Charles Hill
## 28/04/2021

import pydicom
import os
import csv
import re
import json
import sys
import time

molliDir = r"D:\UKB_Liver\20204_2_0"

subjList = [os.path.join(molliDir,x) for x in os.listdir(molliDir) if x.endswith("_0")]

subjDict = {}
t0 = time.time()
subjList = subjList[:]
for i,fol in enumerate(subjList):
    t_elaps = time.time() - t0
    eta = (len(subjList) - (i+1)) * (t_elaps/(i+1)) 
    sys.stdout.write("\r[{}/{}] Time Elapsed = {:.0f} s, Till End = {:.0f}:{:.0f}".format(i,len(subjList),t_elaps,eta/60,eta % 60))
    molliList = [os.path.join(fol,x) for x in os.listdir(fol) if x.endswith(".dcm")]
    instDict = {}
    for f in molliList:
        ds = pydicom.dcmread(f,stop_before_pixels=True)
        if "M" in ds.ImageType:
            dsDict = {}
            for elem in ds:
                
                if type(elem.value) == str:
                    dsDict[elem.name] = str(elem.value)
                elif type(elem.value) in [float,int,pydicom.valuerep.DSfloat,pydicom.valuerep.IS]:
                    dsDict[elem.name] = float(elem.value)
                elif type(elem.value) in [pydicom.multival.MultiValue,list]:
                    dsDict[elem.name] = list(elem.value)

            instDict[int(ds.InstanceNumber)] = dsDict
    subjDict[fol[-17:]] = instDict

with open("biobank_meta_full.json","w") as f:
    json.dump(subjDict,f)

