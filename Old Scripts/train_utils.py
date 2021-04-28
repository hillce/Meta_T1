import matplotlib.pyplot as plt
import numpy as np

def plot_images(inpData,outImg,invTimes,figDir,epoch,index,vmaxDiff=None,val=False,test=False):

    _, ax = plt.subplots(1,3)

    print(invTimes)
    for i in range(inpData.shape[0]):
        ax[0].imshow(inpData[i])
        # print(invTimes[i])
        # ax[0,i].set_title("{:.0f}".format("_".join(map(str,invTimes[i]))))
        ax[0].axis('off')
        ax[1].imshow(outImg[i])
        # ax[1,i].set_title("{:.0f}".format("_".join(map(str,invTimes[i]))))
        ax[1].axis('off')
        ax[2].imshow(abs(outImg[i]-inpData[i]),cmap="jet",vmax=vmaxDiff)
        # # ax[2,i].set_title("{:.0f}".format("_".join(map(str,invTimes[i]))))
        ax[2].axis('off')

    if val:
        plt.savefig("{}Epoch_{}_i_{}_img_val.png".format(figDir,epoch+1,index+1))
    elif test:
        plt.savefig("{}i_{}_img.png".format(figDir,index+1))
    else:
        plt.savefig("{}Epoch_{}_i_{}_img.png".format(figDir,epoch+1,index+1))

def plot_images_meta(inpData,outImg,meta,figDir,epoch,index,vmaxDiff=None,val=False,test=False):

    if inpData.shape[0] > 1:


        _, ax = plt.subplots(3,inpData.shape[0])

        if outImg.shape[0] == 1:
            outImg = np.array([outImg[0]]*inpData.shape[0])

        for i in range(inpData.shape[0]):
            ax[0,i].imshow(inpData[i])
            try:
                ax[0,i].set_title("{:.3f}".format(meta[0,i]))
            except:
                pass
            ax[0,i].axis('off')
            ax[1,i].imshow(outImg[i])
            try:
                ax[1,i].set_title("{:.3f}".format(meta[1,i]))
            except:
                pass
            ax[1,i].axis('off')
            ax[2,i].imshow(abs(outImg[i]-inpData[i]),cmap="jet",vmax=vmaxDiff)
            try:
                ax[2,i].set_title("{:.3f}".format(meta[0,i]-meta[1,i]))
            except:
                pass
            ax[2,i].axis('off')

    else:
        _, ax = plt.subplots(1,3)


        ax[0].imshow(inpData[0])
        try:
            ax[0].set_title("{:.3f}".format(meta[0][0]))
        except:
            pass
        ax[0].axis('off')
        ax[1].imshow(outImg[0])
        try:
            ax[1].set_title("{:.3f}".format(meta[1][0]))
        except:
            pass
        ax[1].axis('off')
        ax[2].imshow(abs(outImg[0]-inpData[0]),cmap="jet",vmax=vmaxDiff)
        try:
            ax[2].set_title("{:.3f}".format(meta[0][0]-meta[1][0]))
        except:
            pass
        ax[2].axis('off')

    if val:
        plt.savefig("{}Epoch_{}_i_{}_img_val.png".format(figDir,epoch+1,index+1))
    elif test:
        plt.savefig("{}i_{}_img.png".format(figDir,index+1))
    else:
        plt.savefig("{}Epoch_{}_i_{}_img.png".format(figDir,epoch+1,index+1))