import json, sys, os

import numpy as np


def meta_loading(fullDataset = True,jsonDir = "./jsonFiles", numpyDir = "./numpyFiles"):

    errMSE = np.load(os.path.join(numpyDir,"ERROR_MAE_3D_T1.npy"))

    with open(os.path.join(jsonDir,"file_list_3D_MAE.json"),"r") as f:
        fileList = json.load(f)

    fileList = [x[:-8] for x in fileList]

    print("Number of Files: ", len(fileList))

    # Flattening Error volume 8x7x7 for RF

    errFlat = np.zeros((errMSE.shape[0],errMSE.shape[1]*errMSE.shape[2]*errMSE.shape[3]))

    for i in range(errMSE.shape[0]):
        errFlat[i] = errMSE[i,:,:,:].flatten()

    print("Pre Flatten {} vs. Flattened {}".format(errMSE.shape,errFlat.shape))

    # Load in tags and additional meta data:

    with open(os.path.join(jsonDir,"./reasons_split.json"),"r") as f: # Tags
        tagDict = json.load(f)

    if not fullDataset:
        with open(os.path.join(jsonDir,"biobank_meta_float.json"),"r") as f: # Float meta data from dcm headers
            metaDict = json.load(f)
    else:
        with open(os.path.join(jsonDir,"biobank_meta_full_one_hot.json"),"r") as f: # All one hot encoded meta
            metaDict = json.load(f)

    with open(os.path.join(jsonDir,"Biobank_Bounding_Boxes.json"),"r") as f: # Bounding box meta data
        bBoxes = json.load(f)

    # Sort through subj to make sure all meta data present

    if fullDataset:
        keys = list(metaDict['eid'].values())
    else:
        keys = list(metaDict.keys())

    keys = [k for k in keys if k in bBoxes.keys()]
    keys = [k for k in keys if k in fileList]

    print("Number of Files with complete meta data: ", len(keys))

    ###### Find out the keys present in every single case:
    if not fullDataset:
        allMetaKeys = []
        instTime = ["1","2","3","4","5","6","7"]
        for k in keys:
            for i in instTime:
                allMetaKeys.extend(list(metaDict[k][i].keys()))

        allMetaKeysSet = set(allMetaKeys)

        keysOI = []
        for k in allMetaKeysSet:
            if allMetaKeys.count(k) == (len(keys)*7):
                keysOI.append(k)
    else:
        keysOI = list(metaDict.keys())

    print("Meta values to use: \n\n", keysOI)
    # Create dataset (full one hot):
    if fullDataset:
        subjLength = len(keys)
        dataLength = len(keysOI)
        bBoxesLength = 16
        errLength = errFlat.shape[1]

        ownDataset = {}

        k0 = list(tagDict.keys())[0]

        for i,k in enumerate(keys):
            sys.stdout.write("\r[{}/{}]".format(i,len(keys)))
            metaList = []
            tags = np.zeros((len(tagDict[k0])))

            for kOI in keysOI:
                metaList.append(metaDict[kOI][str(i)])

            try:
                metaList.extend(bBoxes[k]["Body"])
            except KeyError as e:
                metaList.extend([0,0,0,0])

            try:
                metaList.extend(bBoxes[k]["Liver"])
            except KeyError as e:
                metaList.extend([0,0,0,0])

            try:
                metaList.extend(bBoxes[k]["Lungs"])
            except KeyError as e:
                metaList.extend([0,0,0,0])

            try:
                metaList.extend(bBoxes[k]["Heart"])
            except KeyError as e:
                metaList.extend([0,0,0,0])

            charArr = np.char.find(fileList,k)
            charIdx = np.argwhere(charArr == 0)[0,0]
            
            assert(type(charIdx) == np.int64)
            errMeta = list(errFlat[charIdx,:])

            metaList.extend(errMeta)

            subDataset = np.array(metaList)
            if k in tagDict.keys():
                tags[i] = np.array(tagDict[k])

            ownDataset[k] = (subDataset,tags)

    # Create dataset (float only):
    else:
        subjLength = len(keys)
        dataLength = len(keysOI)*len(instTime)
        print("Data Length: {}\n".format(dataLength))
        bBoxesLength = 16
        errLength = errFlat.shape[1]

        ownDataset = np.zeros((subjLength,dataLength + bBoxesLength + errLength))

        k0 = list(tagDict.keys())[0]
        tags = np.zeros((subjLength,len(tagDict[k0])))

        for i,k in enumerate(keys):
            sys.stdout.write("\r[{}/{}]".format(i,len(keys)))
            metaList = []
            for kOI in keysOI:
                for inst in instTime:
                    metaList.append(metaDict[k][inst][kOI])
            try:
                metaList.extend(bBoxes[k]["Body"])
            except KeyError as e:
                metaList.extend([0,0,0,0])

            try:
                metaList.extend(bBoxes[k]["Liver"])
            except KeyError as e:
                metaList.extend([0,0,0,0])

            try:
                metaList.extend(bBoxes[k]["Lungs"])
            except KeyError as e:
                metaList.extend([0,0,0,0])

            try:
                metaList.extend(bBoxes[k]["Heart"])
            except KeyError as e:
                metaList.extend([0,0,0,0])

            charArr = np.char.find(fileList,k)
            charIdx = np.argwhere(charArr == 0)[0,0]
            
            assert(type(charIdx) == np.int64)
            errMeta = list(errFlat[charIdx,:])

            metaList.extend(errMeta)

            ownDataset[i,:] = np.array(metaList)
            if k in tagDict.keys():
                tags[i] = np.array(tagDict[k])

        ownDataset = (ownDataset,tags)

        print("\n Meta Data for Subj0: {} \n Tag for Subj0: {}".format(ownDataset[0][0][:10],ownDataset[1][0]))

    return ownDataset