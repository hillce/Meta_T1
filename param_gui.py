import sys,os,json,argparse

from PyQt5.QtCore import QLine, QTimer, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QComboBox, QLabel, QGroupBox, QMainWindow, QTableWidgetItem, QWidget, QComboBox, QLineEdit, QVBoxLayout, QHBoxLayout, QApplication, QPushButton, QTableWidget

class Param_GUI(QMainWindow):
    """
    GUI for choosing training parameters
    """
    def __init__(self):
        super().__init__()
        self.main_w()

    def  main_w(self):
        self.main_widget = QWidget(self)

        self.l_fileDir = QLabel("File directory: ")
        self.l_t1MapDir = QLabel("T1 map directory: ")
        self.l_modelName = QLabel("Model name: ")
        self.l_load = QLabel("Load preset data splits: ")
        self.l_lr = QLabel("Learning rate (e-6): ")
        self.l_b1 = QLabel("Beta 1 for Adam (e-2): ")
        self.l_batchSize = QLabel("Batch size: ")
        self.l_numEpochs = QLabel("Number of epochs: ")
        self.l_stepSize = QLabel("Step size for lr scheduler: ")
        self.l_norm = QLabel("Normalise: ")
        self.l_trainSize = QLabel("Train set size: ")
        self.l_valSize = QLabel("Validation set size: ")
        self.l_testSize = QLabel("Test set size: ")
        self.l_inImgC = QLabel("Input Image Channels: ")
        self.l_inMetaC = QLabel("Input Meta Channels: ")
        self.l_outImgC = QLabel("Output Image Channels: ")
        self.l_outMetaC = QLabel("Output Meta Channels: ")
        self.l_minMeta = QLabel("Minimise Meta: ")
        self.l_zeroMeta = QLabel("Zero Meta: ")

        self.le_fileDir = QLineEdit("C:/fully_split_data/")
        self.le_t1MapDir = QLineEdit("C:/T1_Maps/")
        self.le_modelName = QLineEdit("MU")
        self.cb_load = QComboBox()
        self.cb_load.addItems(["True","False"])
        self.s_lr = QLineEdit("1e-3")
        self.s_b1 = QLineEdit("0.5")
        self.s_batchSize = QLineEdit("24")
        self.s_numEpochs =  QLineEdit("50")
        self.s_stepSize =  QLineEdit("20")
        self.cb_norm = QComboBox()
        self.cb_norm.addItems(["False","True"])
        self.le_trainSize = QLineEdit("10000")
        self.le_valSize = QLineEdit("1000")
        self.le_testSize = QLineEdit("1000")
        self.le_inImgC = QLineEdit("7")
        self.le_inMetaC = QLineEdit("7")
        self.le_outImgC = QLineEdit("1")
        self.le_outMetaC = QLineEdit("1")
        self.cb_minMeta = QComboBox()
        self.cb_minMeta.addItems(["True","False"])
        self.cb_zeroMeta = QComboBox()
        self.cb_zeroMeta.addItems(["True","False"])

        self.pb_confirm = QPushButton("Confirm and Exit")
        self.pb_confirm.clicked.connect(self.confirm_n_close)

        self.v_box_1 = QVBoxLayout()
        self.v_box_1.addWidget(self.l_fileDir)
        self.v_box_1.addWidget(self.l_t1MapDir)
        self.v_box_1.addWidget(self.l_modelName)
        self.v_box_1.addWidget(self.l_load)
        self.v_box_1.addWidget(self.l_lr)
        self.v_box_1.addWidget(self.l_b1)
        self.v_box_1.addWidget(self.l_batchSize)
        self.v_box_1.addWidget(self.l_numEpochs)
        self.v_box_1.addWidget(self.l_stepSize)
        self.v_box_1.addWidget(self.l_norm)
        self.v_box_1.addWidget(self.l_trainSize)
        self.v_box_1.addWidget(self.l_valSize)
        self.v_box_1.addWidget(self.l_testSize)
        self.v_box_1.addWidget(self.l_inImgC)
        self.v_box_1.addWidget(self.l_inMetaC)
        self.v_box_1.addWidget(self.l_outImgC)
        self.v_box_1.addWidget(self.l_outMetaC)
        self.v_box_1.addWidget(self.l_minMeta)
        self.v_box_1.addWidget(self.l_zeroMeta)

        self.v_box_2 = QVBoxLayout()
        self.v_box_2.addWidget(self.le_fileDir)
        self.v_box_2.addWidget(self.le_t1MapDir)
        self.v_box_2.addWidget(self.le_modelName)
        self.v_box_2.addWidget(self.cb_load)
        self.v_box_2.addWidget(self.s_lr)
        self.v_box_2.addWidget(self.s_b1)
        self.v_box_2.addWidget(self.s_batchSize)
        self.v_box_2.addWidget(self.s_numEpochs)
        self.v_box_2.addWidget(self.s_stepSize)
        self.v_box_2.addWidget(self.cb_norm)
        self.v_box_2.addWidget(self.le_trainSize)
        self.v_box_2.addWidget(self.le_valSize)
        self.v_box_2.addWidget(self.le_testSize)
        self.v_box_2.addWidget(self.le_inImgC)
        self.v_box_2.addWidget(self.le_inMetaC)
        self.v_box_2.addWidget(self.le_outImgC)
        self.v_box_2.addWidget(self.le_outMetaC)
        self.v_box_2.addWidget(self.cb_minMeta)
        self.v_box_2.addWidget(self.cb_zeroMeta)

        self.h_box = QHBoxLayout()
        self.h_box.addLayout(self.v_box_1)
        self.h_box.addLayout(self.v_box_2)
        self.h_box.addWidget(self.pb_confirm)

        self.main_widget.setLayout(self.h_box)
        self.setCentralWidget(self.main_widget)
        self.show()

    def confirm_n_close(self):
        hParamDict = {}
        hParamDict["fileDir"] = self.le_fileDir.text()
        hParamDict["t1MapDir"] = self.le_t1MapDir.text()
        hParamDict["modelName"] = self.le_modelName.text()
        
        if self.cb_load.currentText() == "False":
            hParamDict["load"] = False
        else:
            hParamDict["load"] = True
        
        hParamDict["lr"] = float(self.s_lr.text())
        hParamDict["b1"] = float(self.s_b1.text())
        hParamDict["batchSize"] = int(self.s_batchSize.text())
        hParamDict["numEpochs"] = int(self.s_numEpochs.text())
        hParamDict["stepSize"] = int(self.s_stepSize.text())
        
        if self.cb_norm.currentText() == "False":
            hParamDict["normalise"] = False
        else:
            hParamDict["normalise"] = True
        
        hParamDict["trainSize"] = int(self.le_trainSize.text())
        hParamDict["valSize"] = int(self.le_valSize.text())
        hParamDict["testSize"] = int(self.le_testSize.text())
        hParamDict["inImgC"] = int(self.le_inImgC.text())
        hParamDict["inMetaC"] = int(self.le_inMetaC.text())
        hParamDict["outImgC"] = int(self.le_outImgC.text())
        hParamDict["outMetaC"] = int(self.le_outMetaC.text())
        
        if self.cb_minMeta.currentText() == "False":
            hParamDict["minMeta"] = False
        else:
            hParamDict["minMeta"] = True

        if self.cb_zeroMeta.currentText() == "False":
            hParamDict["zeroMeta"] = False
        else:
            hParamDict["zeroMeta"] = True

        os.makedirs("./TrainingLogs/{}/".format(hParamDict["modelName"]))

        with open("./TrainingLogs/{}/hparams.json".format(hParamDict["modelName"]),"w") as f:
            json.dump(hParamDict,f)
        self.close()

class Training_GUI(QMainWindow):

    def __init__(self,modelName):
        QMainWindow.__init__(self, None, Qt.WindowStaysOnTopHint)
        self.modelName = modelName
        self.epoch = 1
        self.main_w()

    def main_w(self):
        self.main_widget = QWidget(self)

        ################ h param section ###############################
        self.h_param_load()

        self.table_hParams = QTableWidget()
        self.table_hParams.setRowCount(1)
        self.table_hParams.setColumnCount(len(self.hParamDict.keys()))
        self.table_hParams.setHorizontalHeaderLabels(list(self.hParamDict.keys()))

        for i,k in enumerate(self.hParamDict.keys()):
            self.table_hParams.setItem(0,i,QTableWidgetItem(str(self.hParamDict[k])))

        self.vbox = QVBoxLayout()
        self.vbox.addWidget(self.table_hParams)

        ################ Images ############################################
        self.l_trainImg = QLabel()
        self.l_trainImgText = QLabel("Epoch_1_i_1_img.png")
        self.l_valImg = QLabel()
        self.l_valImgText = QLabel("Epoch_1_i_1_img_val.png")

        self.imgFiles = os.listdir("./TrainingLogs/{}/Training_Figures/".format(self.modelName))

        self.trainImg = QPixmap("./TrainingLogs/{}/Training_Figures/Epoch_1_i_1_img.png".format(self.modelName))
        self.l_trainImg.setPixmap(self.trainImg)

        self.valImg = QPixmap("./TrainingLogs/{}/Training_Figures/Epoch_1_i_1_img_val.png".format(self.modelName))
        self.l_valImg.setPixmap(self.valImg)

        self.hbox0 = QHBoxLayout()
        self.hbox0.addWidget(self.l_trainImgText)
        self.hbox0.addWidget(self.l_valImgText)

        self.hbox1 = QHBoxLayout()
        self.hbox1.addWidget(self.l_trainImg)
        self.hbox1.addWidget(self.l_valImg)

        self.vbox.addLayout(self.hbox0)
        self.vbox.addLayout(self.hbox1)
        self.main_widget.setLayout(self.vbox)
        self.setCentralWidget(self.main_widget)
        self.show()

        ######################### Timer Functions ##############################
        self.check0 = QTimer()
        self.check0.setInterval(10000)
        self.check0.timeout.connect(self.check_for_images)
        self.check0.start()

    def check_for_images(self):
        tempImgFiles = os.listdir("./TrainingLogs/{}/Training_Figures/".format(self.modelName))
        for f in self.imgFiles:
            tempImgFiles.remove(f)
        tempTrainFiles = [x for x in tempImgFiles if "val" not in x and "loss" not in x]
        tempValFiles = [x for x in tempImgFiles if "val" in x and "InvTime" not in x]

        if len(tempTrainFiles) >= 1:
            self.trainImg = QPixmap("./TrainingLogs/{}/Training_Figures/{}".format(self.modelName,tempTrainFiles[-1]))
            self.l_trainImg.setPixmap(self.trainImg)
            self.l_trainImg.update()
            self.l_trainImgText.setText(tempTrainFiles[-1])
            self.l_trainImgText.update()

        if len(tempValFiles) >= 1:
            self.valImg = QPixmap("./TrainingLogs/{}/Training_Figures/{}".format(self.modelName,tempValFiles[-1]))
            self.l_valImg.setPixmap(self.valImg)
            self.l_valImg.update()
            self.l_valImgText.setText(tempValFiles[-1])
            self.l_valImgText.update()

        self.imgFiles = os.listdir("./TrainingLogs/{}/Training_Figures/".format(self.modelName))

    def check_for_loss(self):
        pass

    def h_param_load(self):
        self.hParamDict = {}
        with open("./TrainingLogs/{}/hparams.json".format(self.modelName),"r") as f:
            self.hParamDict = json.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",help="Name for saving the model",type=str,dest="modelName",required=True)
    args = parser.parse_args()

    app = QApplication(sys.argv)
    main_win = Training_GUI(modelName=args.modelName)
    main_win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()