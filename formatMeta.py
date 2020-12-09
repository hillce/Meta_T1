# import os,sys
# from os.path import isdir

# import pydicom
# import numpy as np
# import matplotlib.pyplot as plt

# dataDir = "D:/UKB_Liver/20204_2_0/"

# subjList = [x for x in os.listdir(dataDir) if os.path.isdir(os.path.join(dataDir,x))]


# metaOI = {"ImagingFrequency":(0x0018,0x0084)
#             }


# freqHist = []
# for i,subj in enumerate(subjList):
#     sys.stdout.write("\r[{}/{}]".format(i,len(subjList)))
#     fullSubjPath = os.path.join(dataDir,subj)
#     dcmList = [x for x in os.listdir(fullSubjPath) if x[-4:] in [".IMA",".dcm"]]

#     for dcm in dcmList:
#         dcmFullPath = os.path.join(fullSubjPath,dcm)
#         ds = pydicom.dcmread(dcmFullPath)
#         if "M" in ds.ImageType:
#             for k in metaOI.keys():
#                 elem = ds[metaOI[k][0],metaOI[k][1]]
#                 freqHist.append(elem.value)
#             break

# plt.hist(freqHist)
# plt.show()
