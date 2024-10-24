#!/usr/bin/python3
import sys
import enum
from os.path import join, dirname, abspath
import queue
import serial
import serial.tools.list_ports as port_list
from qtpy import uic
from qtpy.QtCore import Slot, QTimer, QThread, Signal, QObject, Qt
from qtpy.QtWidgets import QApplication, QMainWindow, QMessageBox, QAction, QDialog, QTableWidgetItem, QLabel
from pyqtgraph import PlotWidget
import pyqtgraph as pg
from collections import deque
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtGui import QPainter
from PyQt5 import QtCore, QtSvg
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton, QListWidget, QListWidgetItem

import math
import os
import numpy as np
import random
import qtmodern.styles
import qtmodern.windows
import time
import json
import pprint
from portdetection import DetectDevices

from dispatchers import PrimaryThread

from mimic import Mimic
from crud import CRUD
from dataview import DataView
from datetime import datetime

''' 
Database codes are in crud.py file. here the object name is db. Accessed by self.db.
Implemented in sensordata(...) callback function. database file is flow.db . 
'''

_UI = join(dirname(abspath(__file__)), 'top.ui')
_UI2 = join(dirname(abspath(__file__)), 'dashboard.ui')
_UI3 = join(dirname(abspath(__file__)), 'commands.ui')

'''
dev_address function_code reg_addr_high reg_addr-LOW, reg_qntt_high red_qntt_low crc_low crc_high
'''
#08  04  00  00  00  02  71  52
_CMD_1 = [0x08, 0x04, 0x00, 0x00, 0x00, 0x02, 0x71, 0x52]
_CMD_2 = [0x08, 0x04, 0x00, 0x00, 0x00, 0x02, 0x71, 0x52]
_CMD_3 = [0x08, 0x04, 0x00, 0x22, 0x00, 0x02, 0xD1, 0x58]
_CMD_4 = [0x08, 0x04, 0x00, 0x04, 0x00, 0x02, 0x30, 0x93]
_CMD_5 = [0x08, 0x04, 0x00, 0x00, 0x00, 0x02, 0x71, 0x52]
_CMD_6 = [0x08, 0x04, 0x00, 0x22, 0x00, 0x02, 0xD1, 0x58]
_CMD_7 = [0x08, 0x04, 0x00, 0x00, 0x00, 0x02, 0x71, 0x52]
_CMD_8 = [0x08, 0x04, 0x00, 0x04, 0x00, 0x02, 0x30, 0x93]
_CMD_9 = [0x09, 0x04, 0x00, 0x00, 0x00, 0x02, 0x70, 0x83]
_CMD_10 = [0x09, 0x04, 0x00, 0x00, 0x00, 0x02, 0x70, 0x83]
_CMD_11 = [0x09, 0x04, 0x00, 0x22, 0x00, 0x02, 0xD0, 0x89]
_CMD_12 = [0x09, 0x04, 0x00, 0x04, 0x00, 0x02, 0x31, 0x42]
_CMD_13 = [0x09, 0x04, 0x00, 0x00, 0x00, 0x02, 0x31, 0x42]
_CMD_14 = [0x09, 0x04, 0x00, 0x22, 0x00, 0x02, 0xD0, 0x89]
_CMD_15 = [0x09, 0x04, 0x00, 0x00, 0x00, 0x02, 0x70, 0x83]
_CMD_16 = [0x09, 0x04, 0x00, 0x04, 0x00, 0x02, 0x31, 0x42]
_CMD_17 = [0x0A, 0x04, 0x00, 0x00, 0x00, 0x02, 0x70, 0xB0]
_CMD_18 = [0x0A, 0x04, 0x00, 0x22, 0x00, 0x02, 0xD0, 0xBA]
_CMD_19 = [0x0A, 0x04, 0x00, 0x04, 0x00, 0x02, 0x31, 0x71]
class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.widget = uic.loadUi(_UI, self)
        self.mimic = Mimic()
        self.window_title = "top"
        self.mimic = Mimic()
        self.ports = DetectDevices()
        self.selectedPort = ""
        self.lst = QListWidget()

        self.sensor = ''
        self.sensorThread = ''
        self.sensorThreadCreated = False
        self.sensorPortOpen = False
        self.sensorDataString = ""

        self.serialSensor = ""
        self.selectedPort = ""

        self.cmdlist = []
        self.cmdlist.append(_CMD_1)
        self.cmdlist.append(_CMD_2)
        self.cmdlist.append(_CMD_3)
        self.cmdlist.append(_CMD_4)
        self.cmdlist.append(_CMD_5)
        self.cmdlist.append(_CMD_6)
        self.cmdlist.append(_CMD_7)
        self.cmdlist.append(_CMD_8)
        self.cmdlist.append(_CMD_9)
        self.cmdlist.append(_CMD_10)
        self.cmdlist.append(_CMD_11)
        self.cmdlist.append(_CMD_12)
        self.cmdlist.append(_CMD_13)
        self.cmdlist.append(_CMD_14)
        self.cmdlist.append(_CMD_15)
        self.cmdlist.append(_CMD_16)
        self.cmdlist.append(_CMD_17)
        self.cmdlist.append(_CMD_18)
        self.cmdlist.append(_CMD_19)

        #List only usb-ttl ports in self.portListBox QListWidget
        self.ports = list(port_list.comports())
        for p in self.ports:
            if "USB" in p[1]:
                self.portListBox.addItem(p[0])

        self.btn1.setEnabled(False)
        self.btn2.setEnabled(True)

        #self.lst.selectedItems()
        # getting item changed signal
        self.portListBox.currentItemChanged.connect(self.portLstItemChanged)

        self.db = CRUD("flow.db")
        self.db.openDBHard()

        self.dtv = DataView()

        # renderer =  QtSvg.QSvgRenderer('ico1.svg')
        # painter = QPainter(self.btn1)
        # painter.restore()
        # renderer.render(painter)
        # self.btn1.show()

    def portLstItemChanged(self, tm):
            print("Port Item Changed " + tm.text())
            self.selectedPort = tm.text()
            if tm.text() != "":
                #if "USB" in tm.text():
                self.btn1.setEnabled(True)

    def startSensorThread(self):
        if self.sensorPortOpen:
            if not self.sensorThreadCreated:
                self.sensor = PrimaryThread(self.serialSensor, self.cmdlist)
                self.sensorThread = QThread()
                self.sensorThread.started.connect(self.sensor.run)
                self.sensor.signal.connect(self.sensorData)
                self.sensor.moveToThread(self.sensorThread)
                self.sensorThread.start()
                self.sensorThreadCreated = True
                print("Starting Sensor Thread")

    def extractFlowData(self, starData=""):
        parts = starData.split(" ")
        res = "0000.00"
        if(len(parts) >= 18):
            #val = int('0x' + parts[15]+parts[16]+parts[17]+parts[18], base=16)
            val = int(parts[12]+parts[13], base=16)
            if val > 0:
                res = str(val/1000)
            else:
                res = 0
        return res

    def extractSumData(self, starData=""):
        parts = starData.split(" ")
        res = "0000.00"
        if(len(parts) >= 18):
            val = int(parts[12]+parts[13], base=16)
            if val > 0:
                res = str(val/1000)
            else:
                res = 0
        return res

    def extractPercentData(self, starData=""):
        parts = starData.split(" ")
        res = "0000.00"
        if (len(parts) >= 18):
            #val = int(parts[12] + parts[13] + parts[14] + parts[15], base=16)
            val = int(parts[12] + parts[13], base=16)
            if val > 0:
                res = str(val/1000)
            else:
                res = 0
        return res

#Data Received from thread. parts[12] is dev id. Not Applicable now.
#12-13-2021 23:40:37 - [8, 4, 0, 0, 0, 2, 113, 82] - 08 04 04 00 1A 00 00 43 43 . Terminal data is shown as below
#08 04 00 00 00 02 71 52 - 08 04 04 00 A1 00 00 43 43
# return data: dev id - F.code - Bytes Count - B3 B2 B1 B0 - CRC - CRC
#                 08      04        04         00 1A 00 00 - 43 - 43
    def sendMeterDataFromSensorString(self, sensorString:str):
        parts = sensorString.split(" ")
        devid = 0
        if(len(parts) >= 18):
            #print(parts[0] + " " +parts[9] + " " +parts[10] + " " +parts[11] + " " + parts[12])
            if(int(parts[9], base=16) == 8):
                devid = 8
                if(int(parts[3], base=16) == 0):
                    self.mimic.meterFlow1 = self.extractFlowData(sensorString)
                    self.mimic.AppendFlow1(float(self.extractFlowData(sensorString)))
                if(int(parts[3], base=16) == 34):
                    self.mimic.meterSum1 = self.extractSumData(sensorString)
                if (int(parts[3], base=16) == 4):
                    #self.mimic.meterSum1 = self.extractSumData(sensorString)
                    print("PERCENT 1: " + sensorString)
            if (int(parts[9], base=16) == 9):
                devid = 9
                if (int(parts[3], base=16) == 0):
                    self.mimic.meterFlow2 = self.extractFlowData(sensorString)
                if (int(parts[3], base=16) == 34):
                    self.mimic.meterSum2 = self.extractSumData(sensorString)
                if (int(parts[3], base=16) == 4):
                    # self.mimic.meterSum1 = self.extractSumData(sensorString)
                    print("PERCENT 2: " + sensorString)
            if (int(parts[9], base=16) == 10):
                devid = 10
                if (int(parts[3], base=16) == 0):
                    self.mimic.meterFlow3 = self.extractFlowData(sensorString)
                if (int(parts[3], base=16) == 34):
                    self.mimic.meterSum3 = self.extractSumData(sensorString)
                if (int(parts[3], base=16) == 4):
                    # self.mimic.meterSum1 = self.extractSumData(sensorString)
                    print("PERCENT 3: " + sensorString)
        return devid

    def sensorData(self, data_stream):
        self.sensorDataString = data_stream
        strdatetime = datetime.today().strftime('%m-%d-%Y %H:%M:%S')                #Collect Present Date Time
        print(strdatetime + " - " +self.sensorDataString)                           #
        #print(self.sensorDataString)
        self.msgListBox.addItem(strdatetime + " - " +self.sensorDataString)         #Insert incomming data to local List Box
        devid = self.sendMeterDataFromSensorString(self.sensorDataString)
        self.db.insert_meter_data([strdatetime, self.sensorDataString, str(devid)])  # Inserting data to database
        self.mimic.repaint()
        if(self.msgListBox.count() > 10):
            self.msgListBox.clear()

    @Slot()
    def on_btn1_clicked(self):
        if self.selectedPort != "":
            if not self.sensorPortOpen:
                try:
                    self.serialSensor = serial.Serial(self.selectedPort, baudrate=9600, timeout=0)
                    self.sensorPortOpen = True
                except serial.SerialException as ex:
                    self.sensorPortOpen = False
                    print(ex.strerror)
                    print("Error Opening Serial Port..........................................")
                finally:
                    print("Serial Port Connected..........................")
                    self.btn2.setEnabled(True)
        # self.mim = Mimic()
        # self.mim.setFixedHeight(100)
        # self.VL0.addWidget(self.mim)
        # self.setWindowTitle(self.window_title)

        #Show svg file svgwidget
        #self.svgwidget = QtSvg.QSvgWidget('ico1.svg')
        #comment self.VL1 = QVBoxLayout()
        #self.VL0.addWidget(self.svgwidget)
        #comment self.dash.show()

    @Slot()
    def on_btn2_clicked(self):
        #self.mimic.show() # Enable This Line to show the Mimic without starting sensorThread.
        if self.sensorPortOpen:
            if not self.sensorThreadCreated:
                self.startSensorThread()
            self.mimic.show()

    @Slot()
    def on_btn3_clicked(self):
        ''' Example code to insert data in database
        #self.db.insert_meter_data_hard()
        '''
        self.dtv.summery = True
        self.dtv.showNormal()


    @Slot()
    def on_btn4_clicked(self):
        self.dtv.summery = None
        self.dtv.showNormal()

    @Slot()
    def on_btnPause_clicked(self):
        if self.btnPause.text() == "Pause":
            self.btnPause.setText("Start")
            self.sensor.pause = True
        else:
            self.btnPause.setText("Pause")
            self.sensor.pause = False

    @Slot()
    def on_btnGetStream_clicked(self):
        print("btnGetStream or Connect PD Clicked")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    #qtmodern.styles.dark(app)
    qtmodern.styles.light(app)

    mw_class_instance = MainWindow()
    mw = qtmodern.windows.ModernWindow(mw_class_instance)
    #mw.showFullScreen()
    mw.showNormal()
    sys.exit(app.exec_())
