
'''
Project builds on PyQt5 Matplotlib . Codes coming from Flowmeter, filereader.dat projects.

FAQ: What is self.flowdata ?
    self.flowdata is a queue object. for easy pop and easy append.
'''
from os.path import join, dirname, abspath
from qtpy import uic
from qtpy.QtCore import Slot, QTimer, QThread, Signal, QObject, Qt
from PyQt5.QtGui import *
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
from pyqtgraph import PlotWidget
import pyqtgraph as pg
from collections import deque

_UI5 = join(dirname(abspath(__file__)), 'charttabs.ui')
# from charttab import ChartTab

class ChartTab(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self)  # self, *args, **kwargs
        self.setParent(parent)
        self.widget = uic.loadUi(_UI5, self)
        self.flowdata = deque()
        self.flowsum = deque()
        self.plotter = PlotWidget()
        self.plotter.showGrid(x=True, y=True, alpha=None)
        self.plotter.setLabel('left', 'Flow : m3/M')
        self.plotter.getViewBox().setYRange(-65, 40)
        self.curve1 = self.plotter.plot(0, 0, "flow", 'b')
        self.ttm = 0.0
        self.tfdata = deque()
        self.maxLen = 7680
        self.vlay01.addWidget(self.plotter)
        self.initUI()

    def Append(self, data=0):
        self.flowdata.append(data)
        if len(self.flowdata) > self.maxLen:
            self.flowdata.popleft()
        if len(self.tfdata) < self.maxLen:
            self.ttm += 0.032
            self.tfdata.append(self.ttm)
        self.curve1.setData(self.tfdata, self.flowdata)

    def initUI(self):
        pass
