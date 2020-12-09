import sys
import os

from PyQt5.QtGui import QImage, QPixmap
import numpy as np
from PyQt5.QtWidgets import QApplication, QComboBox, QHBoxLayout, QLabel, QMainWindow, QPushButton, QVBoxLayout, QWidget
from numpy.lib.type_check import real


class Testing_GUI(QMainWindow):

    def __init__(self):
        super().__init__()
        self.nC = 7
        self.dir = "./TrainingLogs/"
        self.firstTime = True
        self.main_w()

    def main_w(self):
        self.main_widget = QWidget(self)

        self.modelList = os.listdir(self.dir)

        self.l_model = QLabel("Model: ")
        self.cb_model = QComboBox()
        self.cb_model.addItems(self.modelList)

        self.pb_load = QPushButton("Load")
        self.pb_load.clicked.connect(self.load)

        self.hbox0 = QHBoxLayout()
        self.hbox0.addWidget(self.l_model)
        self.hbox0.addWidget(self.cb_model)
        self.hbox0.addWidget(self.pb_load)

        # Load First image:
        self.imgDict = {}
        self.hbox1 = QHBoxLayout()
        self.hbox2 = QHBoxLayout()
        for i in range(self.nC):
            self.imgDict["Inst {} Input".format(i+1)] = QLabel()
            self.imgDict["Inst {} Output".format(i+1)] = QLabel()

            self.hbox1.addWidget(self.imgDict["Inst {} Input".format(i+1)])
            self.hbox2.addWidget(self.imgDict["Inst {} Output".format(i+1)])

        self.pb_next = QPushButton("Next Subject")
        self.pb_prev = QPushButton("Prev Subject")

        self.hbox3 = QHBoxLayout()
        self.hbox3.addWidget(self.pb_prev)
        self.hbox3.addWidget(self.pb_next)

        self.vbox = QVBoxLayout()
        self.vbox.addLayout(self.hbox0)
        self.vbox.addLayout(self.hbox1)
        self.vbox.addLayout(self.hbox2)
        self.vbox.addLayout(self.hbox3)
        self.main_widget.setLayout(self.vbox)
        self.setCentralWidget(self.main_widget)
        self.show()

    def load(self):
        if self.firstTime:
            self.model = self.cb_model.currentText()
            self.imgList = [ x for x in os.listdir("{}{}/Test_Figures/".format(self.dir,self.model)) if x.endswith(".npy")]
            self.imgList.sort()
            
            self.reals = [x for x in self.imgList if "Real" in x]
            self.fakes = [x for x in self.imgList if "Fake" in x]

            realImg = np.load("{}{}/Test_Figures/{}".format(self.dir,self.model,self.reals[0]))
            fakeImg = np.load("{}{}/Test_Figures/{}".format(self.dir,self.model,self.fakes[0]))

            realImg = realImg.astype(np.int16)
            fakeImg = fakeImg.astype(np.int16)

            for i in range(self.nC):
                qImg = QImage(realImg[i,:,:].data, realImg.shape[2], realImg.shape[1], 2*realImg.shape[1], QImage.Format_Grayscale16)
                pixmap = QPixmap(qImg)
                self.imgDict["Inst {} Input".format(i+1)].setPixmap(pixmap)
                self.imgDict["Inst {} Input".format(i+1)].update()

                qImg = QImage(fakeImg[i,:,:].data, fakeImg.shape[2], fakeImg.shape[1], 2*fakeImg.shape[1], QImage.Format_Grayscale16)
                pixmap = QPixmap(qImg)
                self.imgDict["Inst {} Output".format(i+1)].setPixmap(pixmap)
                self.imgDict["Inst {} Output".format(i+1)].update()

            print("Loaded Images!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = Testing_GUI()
    main_win.show()
    sys.exit(app.exec_())