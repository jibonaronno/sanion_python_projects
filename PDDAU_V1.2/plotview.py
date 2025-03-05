

from PyQt5.QtWidgets import QWidget # QApplication, QTreeView, QFileDialog, QVBoxLayout, QPushButton
from qtpy import uic
from os.path import join, dirname, abspath
from qtpy.QtCore import Slot
from pathlib import Path
from PyQt5.QtGui import QStandardItemModel, QStandardItem
from PyQt5.QtCore import QModelIndex
from charttab import ChartTab
import os
from binaryfilereader import BinaryFileReader
import numpy as np
from kalmanfilter import KalmanFilter

_UI_PLOT_VIEW = join(dirname(abspath(__file__)), 'comparison_chart.ui')

class PlotView(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.widget = uic.loadUi(_UI_PLOT_VIEW, self)
        self.charts01 = ChartTab(self)
        self.charts02 = ChartTab(self)
        # self.horizontalLayout_ = QHBoxLayout()
        self.horizontalLayout_4.addWidget(self.charts01)
        self.horizontalLayout_5.addWidget(self.charts02)

    def injectDataStreamToGraph_16bit(self, data):
        for i in range(0, len(data), 2):
            two_bytes = data[i:i + 2]
            self.charts01.Append(int(two_bytes))
