import json, sys
from textwrap import wrap

import matplotlib
matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

with open("biobank_meta.json","r") as f:
    metaDict = json.load(f)

subj = list(metaDict.keys())
instances = list(range(1,8))

metaClasses = list(metaDict[subj[0]]["1"].keys())

binNum = 50
valLimit = 15
for mC in metaClasses:
    try:
        if "Time" not in mC:
            if "Date" not in mC:

                x0 = [metaDict[sJ][str(1)][mC] for sJ in subj]
                x1 = [metaDict[sJ][str(2)][mC] for sJ in subj]

                if type(x0[0]) == float:
                    if x0 == x1:
                        continue
                        fig = plt.figure()
                        ax = fig.add_subplot()
                        hist, bins = np.histogram(x0,bins=binNum)
                        width = bins[1] - bins[0]
                        ax.bar(bins[:-1],hist,width=width)

                        ax.set_xlabel(mC)
                        ax.set_ylabel("Value")

                        plt.show()
                    else:
                        if "Window" in mC:
                            fig = plt.figure()
                            ax = fig.add_subplot(projection='3d')

                            for inst in instances:
                                x = [metaDict[sJ][str(inst)][mC] for sJ in subj]
                                hist, bins = np.histogram(x,bins=binNum)
                                width = bins[1] - bins[0]
                                ax.bar(bins[:-1],hist,zs=inst, zdir="y",width=width)

                            ax.set_xlabel(mC)
                            ax.set_ylabel("Instance Number")
                            ax.set_zlabel("Value")

                            plt.show()

                elif type(x0[0]) == str:
                    continue
                    if x0 == x1:
                        fig = plt.figure()
                        ax = fig.add_subplot()

                        bins = np.unique(x0)
                        if bins.shape[0] < 150:
                            vals = [x0.count(itm) for itm in bins]

                            zipped = [(bin,val) for bin,val in zip(bins,vals) if val > valLimit]
                            
                            bins = [x[0] for x in zipped]
                            bins = [ '\n'.join(wrap(l, 20)) for l in bins]
                            vals = [x[1] for x in zipped]

                            ax.barh(bins,vals)

                            ax.set_xlabel("Value")
                            ax.set_ylabel(mC)

                            print(bins,vals)

                        else:
                            print("{}:   {} len {}".format(mC,bins,bins.shape))

                        mng = plt.get_current_fig_manager()
                        mng.window.showMaximized()

                        plt.show()

    except KeyError as e:
        print(e)
