#!/usr/bin/python3

import sys
from os.path import join, dirname, abspath
from qtpy import uic
from PyQt5.QtWidgets import *
from pyqtgraph import PlotWidget
import qtmodern.styles
import qtmodern.windows
from collections import deque
import serial
import serial.tools.list_ports as port_list

from mimic import Mimic
from comparison_chart import CompareChartWidget
import os

os.environ["XDG_SESSION_TYPE"] = "xcb"
# _UI5 = join(dirname(abspath(__file__)), 'charttabs.ui')
_UI_TOP = join(dirname(abspath(__file__)), 'top.ui')

class MainWindow(QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self)
        self.widget = uic.loadUi(_UI_TOP, self)
        self.mimic = Mimic(self.customa)
        # verticalLayout_4 = QVBoxLayout()
        # self.verticalLayout_4.addWidget(self.mimic)
        self.comparison_chart = None
        self.UiComponents()
        print(self.verticalLayout_4.children())
        self.show()

    def UiComponents(self):
        self.actionOpen.triggered.connect(self.OpenFile)
        self.actionOpen_Folder.triggered.connect(self.OpenFolder)

    def OpenFile(self):
        print("Menu -> Open File")
        location = dirname(abspath(__file__)) + '\\'
        fname = QFileDialog.getOpenFileName(self, 'Open file', location, "json files (*.json *.txt)")

    def OpenFolder(self):
        print("Menu -> Open Folder")
        location = dirname(abspath(__file__)) + '\\'
        foldername = QFileDialog.getExistingDirectory(self, "Select Folder", location)
        self.comparison_chart = CompareChartWidget(foldername)
        self.comparison_chart.showNormal()

        # self.mimic.showNormal()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    # qtmodern.styles.dark(app)
    qtmodern.styles.light(app)
    mw_class_instance = MainWindow()
    mw = qtmodern.windows.ModernWindow(mw_class_instance)
    # mw.showFullScreen()
    mw.showNormal()
    sys.exit(app.exec_())

