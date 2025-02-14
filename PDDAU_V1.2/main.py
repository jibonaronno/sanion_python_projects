#!/usr/bin/python3

import sys
from os.path import join, dirname, abspath
from qtpy import uic
from qtpy.QtCore import Slot
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
import json
from pddsrvr import PddSrvr, ServerThread
import threading

os.environ["XDG_SESSION_TYPE"] = "xcb"
# _UI5 = join(dirname(abspath(__file__)), 'charttabs.ui')
_UI_TOP = join(dirname(abspath(__file__)), 'top.ui')

class Configs(object):
    def __init__(self):
        super().__init__()
        self.settings = None
        self.local_ip = ""
        self.local_port = ""
        self.remote_ip = ""
        self.remote_port = ""

    def loadJson(self, json_file):
        try:
            with open(json_file, 'r') as f:
                self.settings = json.load(f)
                self.local_ip = self.settings.get("local_ip", "")
                self.local_port = self.settings.get("local_port", "")
        except Exception as e:
            print(f"Error Loading JSON File main.py : {str(e)}")

class MainWindow(QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self)
        self.widget = uic.loadUi(_UI_TOP, self)
        self.mimic = Mimic(self.customa)
        # verticalLayout_4 = QVBoxLayout()
        # self.verticalLayout_4.addWidget(self.mimic)
        self.comparison_chart = None
        self.UiComponents()
        self.configs = Configs()
        self.configs.loadJson("settings.json")
        self.event_pddthread_stop = threading.Event()
        self.send_samples = threading.Event()

        # self.pdsrvr = PddSrvr(self.event_pddthread_stop, self.send_samples)
        # self.server_thread = threading.Thread(target=self.pdsrvr.run_server)
        self.server_thread = ServerThread(host='192.168.246.147', port = 5000)
        self.server_thread.received.connect(self.on_ServerThraedSignalCallback)

        try:
            self.qlist.addItem('Local IP : ' + self.configs.local_ip)
            self.qlist.addItem('Local Port : ' + self.configs.local_port)
            self.lineEdit.setText(self.configs.local_ip)
        except Exception as e:
            print(f'Error main.py : {str(e)}')
        # print(self.verticalLayout_4.children())
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

    def on_ServerThraedSignalCallback(self, message):
        print(message)


    @Slot()
    def on_btnInit_clicked(self):
        self.server_thread.start()
        pass
        # self.server_thread.start()
        # self.pdsrvr.run_server()

    @Slot()
    def on_btnProcD_clicked(self):
        print("Set send_samples .. \n")
        self.send_samples.set()


    @Slot()
    def on_btnStop_clicked(self):
        self.event_pddthread_stop.set()

    @Slot()
    def on_btnchk128_clicked(self):
        pass

    def closeEvent(self, event):
        """Stop the server thread when closing the window."""
        self.server_thread.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # qtmodern.styles.dark(app)
    qtmodern.styles.light(app)
    mw_class_instance = MainWindow()
    mw = qtmodern.windows.ModernWindow(mw_class_instance)
    # mw.showFullScreen()
    mw.showNormal()
    sys.exit(app.exec_())

