import numpy as np
import sys

npFiles = np.load("ownDataset_float.npz",allow_pickle=True)

ownDataset = {}
for i,fN in enumerate(npFiles.files):
    sys.stdout.write("\r[{}/{}]".format(i,len(npFiles.files)))
    ownDataset[fN] = (npFiles[fN][0],npFiles[fN][1][i,:])
    del npFiles[fN]

np.savez_compressed("ownDataset_float.npz",**ownDataset)

