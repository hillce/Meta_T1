import matplotlib.pyplot as plt

def plot_images(inpData,outImg,invTimes,figDir,epoch,index,vmaxDiff=None,val=False,test=False):

    _, ax = plt.subplots(3,7)

    for i in range(inpData.shape[0]):
        ax[0,i].imshow(inpData[i])
        ax[0,i].set_title("{:.0f}".format(invTimes[0,i]))
        ax[0,i].axis('off')
        ax[1,i].imshow(outImg[i])
        ax[1,i].set_title("{:.0f}".format(invTimes[1,i]))
        ax[1,i].axis('off')
        ax[2,i].imshow(abs(outImg[i]-inpData[i]),cmap="jet",vmax=vmaxDiff)
        ax[2,i].set_title("{:.0f}".format(invTimes[0,i]-invTimes[1,i]))
        ax[2,i].axis('off')

    if val:
        plt.savefig("{}Epoch_{}_i_{}_img_val.png".format(figDir,epoch+1,index+1))
    elif test:
        plt.savefig("{}i_{}_img.png".format(figDir,index+1))
    else:
        plt.savefig("{}Epoch_{}_i_{}_img.png".format(figDir,epoch+1,index+1))
