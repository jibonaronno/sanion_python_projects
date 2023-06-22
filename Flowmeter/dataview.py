
from os.path import join, dirname, abspath

import pandas
from qtpy import uic
from qtpy.QtCore import Slot, QTimer, QThread, Signal, QObject, Qt
from PyQt5.QtGui import *
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from crud import CRUD
import pandas as pd

_UI3 = join(dirname(abspath(__file__)), 'dataview.ui')

class DataView(QWidget):
    def __init__(self, parent=None, summery=None):
        QWidget.__init__(self)  # self, *args, **kwargs
        self.setParent(parent)
        self.widget = uic.loadUi(_UI3, self)
        self.startdte = QDateTimeEdit()
        self.crud = CRUD("flow.db")
        self.crud.openDBHard()
        self.summery = None
        if summery:
            self.summery = summery
        self.initUI()
        #self.tableWidget = QTableWidget()


    def initUI(self):
        self.startdte = self.startDate
        self.startdte.setDateTime(QtCore.QDateTime.currentDateTime())
        self.stopDate.setDateTime(QtCore.QDateTime.currentDateTime())

    def extractData(self, starData=""):
        parts = starData.split(" ")
        res = "0000.00"
        if(len(parts) >= 18):
            #val = int('0x' + parts[15]+parts[16]+parts[17]+parts[18], base=16)
            val = int(parts[12]+parts[13], base=16)
            if val > 0:
                res = str(val/1000)
            else:
                res = "0"
        return res

    def ModbusData(self, sensorString:str):
        parts = sensorString.split(" ")
        flowd = '0'
        if(len(parts) >= 18):
            #print(parts[0] + " " +parts[9] + " " +parts[10] + " " +parts[11] + " " + parts[12])
            #if(int(parts[9], base=16) == 8):
                #devid = 8
                #if(int(parts[3], base=16) == 0):
            flowd = self.extractData(sensorString)
        return flowd

    def getListByFieldType(self, Data:list, fieldType:str):
        res = []
        crit = "00"
        if fieldType == "Flow":
            crit = "00"
        elif fieldType == "Total":
            crit = "22"
        elif fieldType == "Percent":
            crit = "04"
        print("Field Type : " + crit)
        for dat in Data:
            if dat[3] == crit:
                res.insert(dat)
        return res

    def GenerateSummery(self):
        idx = 0
        flowd = '0'
        devid = int(self.cmbDevid.itemText(self.cmbDevid.currentIndex()))
        data = self.crud.getListByDateRange(self.startdte.dateTime(), self.stopDate.dateTime(), devid)
        print(len(data))
        df = pd.DataFrame(data, columns=["Date", "Time", "Dev Id", "MBus Data"])
        print(df)
        with pandas.ExcelWriter("summery.xlsx") as writer:
            df.to_excel(writer)

    @Slot()
    def on_btnQuery_clicked(self):
        #self.cmbDevid = QComboBox()
        if self.summery:
            self.GenerateSummery()
        else:
            idx = 0
            flowd = '0'
            devid = int(self.cmbDevid.itemText(self.cmbDevid.currentIndex()))
            data = self.crud.getListByDateRange(self.startdte.dateTime(), self.stopDate.dateTime(), devid)
            #data = self.getListByFieldType(data, self.cmbField.itemText(self.cmbDevid.currentIndex()))
            print(len(data))
            #print(data[0])
            self.tableWidget.clear()
            self.tableWidget.setHorizontalHeaderLabels(["Date", "Time", "Dev ID", "Value"])
            self.tableWidget.setRowCount(len(data))
            for dat in data:
                flowd = self.ModbusData(dat[3])
                self.tableWidget.setItem(idx, 0, QTableWidgetItem(dat[0]))
                self.tableWidget.setItem(idx, 1, QTableWidgetItem(dat[1]))
                self.tableWidget.setItem(idx, 2, QTableWidgetItem(dat[2]))
                self.tableWidget.setItem(idx, 3, QTableWidgetItem(flowd))
                #self.tableWidget.setItem(idx, 3, QTableWidgetItem(dat[3]))
                idx += 1

