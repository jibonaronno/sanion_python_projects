'''
 List all files in the given folder in a tree view. It is hardcoded that selected file is a raw binary file.
 When click an item (i.e. file) in the tree view, it will read the file and show 2 graph chart on the right
 section. First one is the
'''
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
import struct
# import sys
# from PyQt5.QtGui import QIcon
# from PyQt5.QtCore import QDir
# from PyQt5.QtWidgets import QFileSystemModel, QHBoxLayout

_UI_COMPARISON_CHART = join(dirname(abspath(__file__)), 'comparison_chart.ui')

class CompareChartWidget(QWidget):
    def __init__(self, folder_path:str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.widget = uic.loadUi(_UI_COMPARISON_CHART, self)
        self.model = QStandardItemModel()
        self.file_path_dictionary = dict()
        self.parent_item = QStandardItem(folder_path)
        self.InvokeFileModel(folder_path)
        self.treeView.clicked.connect(self.on_tree_item_click)
        self.binfil = BinaryFileReader()

        # self.model = QFileSystemModel()
        # self.InvokeFileModel(QDir.rootPath())
        # self.InvokeFileModel(QDir.currentPath())
        # self.InvokeFileModel("wavegenA_PD_FILTER_GRAPH")
        # self.treeView.setModel(self.model)
        self.charts01 = ChartTab(self)
        self.charts02 = ChartTab(self)
        #self.horizontalLayout_ = QHBoxLayout()
        self.horizontalLayout_4.addWidget(self.charts01)
        self.horizontalLayout_5.addWidget(self.charts02)
        self.kalmann = None

    def getFilesInFolder(self, folder_path):
        file_names = []
        if Path.exists(folder_path):
            for file_name in os.listdir(folder_path):
                full_path = os.path.join(folder_path, file_name)
                if os.path.isfile(full_path):
                    file_names.append(file_name)
                    self.file_path_dictionary[file_name] = os.path.join(folder_path, file_name)
                    #return file_names
        return file_names

    def makeSingleTreeViewItem(self, parentItem, itemList:[]):
        # parentItem = QStandardItem("Parent Item")
        for item in itemList:
            child_item = QStandardItem(f"{item}")
            parentItem.appendRow(child_item)
        return parentItem

    def InvokeFileModel(self, folder_path):
        folder_path_object = Path(folder_path)
        if folder_path_object.exists() and folder_path_object.is_dir():
            file_list = self.getFilesInFolder(folder_path_object)
            treeViewItem = self.makeSingleTreeViewItem(self.parent_item, file_list)
            self.model.appendRow(treeViewItem)
            self.treeView.setModel(self.model)
            self.treeView.setColumnWidth(0, 250)
            print(folder_path)

            # self.model.setRootPath("C:\\Users\\jibon\\PycharmProjects\\")
            # self.treeView.setRootIndex(self.model.index("C:\\Users\\jibon\\PycharmProjects\\"))
            # self.treeView.setModel(self.model)

    def injectDataStreamToGraph(self):
        bytesArray = self.binfil.getArray()
        predictions = self.kalmann.filterB(bytesArray)
        for byt in bytesArray:
            self.charts01.Append(int(byt))
        for byt in predictions:
            self.charts02.Append(byt[0])

    def injectRawDataStreamToGraph(self):
        bytesArray = self.binfil.getArray()
        for byt in bytesArray:
            self.charts01.Append(int(byt))

    def injectRawDataStreamToGraphSize(self, _size):
        bytesArray = self.binfil.getArray()
        sz = 0
        self.charts01.Clear()
        for byt in bytesArray:
            self.charts01.Append(int(byt))
            sz += 1
            if(sz >= _size):
                return

    def showKalmannPlotlib(self):
        bytesArray = self.binfil.getArray()
        dt = 1000.0 / 7680.0
        F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
        H = np.array([1, 0, 0]).reshape(1, 3)
        Q = np.array([[0.09, 0.09, 0.0], [0.09, 0.09, 0.0], [0.0, 0.0, 0.0]])
        R = np.array([1.0]).reshape(1, 1)
        self.kalmann = KalmanFilter(F = F, H = H, Q = Q, R = R)
        self.kalmann.filterA(bytesArray)

    def showKalmann(self):
        bytesArray = self.binfil.getArray()
        dt = 1000.0 / 7680.0
        F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
        H = np.array([1, 0, 0]).reshape(1, 3)
        Q = np.array([[0.09, 0.09, 0.0], [0.09, 0.09, 0.0], [0.0, 0.0, 0.0]])
        R = np.array([1]).reshape(1, 1)
        self.kalmann = KalmanFilter(F=F, H=H, Q=Q, R=R)
        self.injectDataStreamToGraph()

    def on_tree_item_click(self, index: QModelIndex):
        # Retrieve the item that was clicked
        item = self.model.itemFromIndex(index)
        print(f"Clicked on: {item.text()} : {self.file_path_dictionary[item.text()]}")
        self.binfil.printFilContentSize(self.file_path_dictionary[item.text()])
        # self.injectDataStreamToGraph()
        # self.showKalmannPlotlib()

        self.injectRawDataStreamToGraphSize(4000)

        self.printHexToConsoleShort(bytearray(self.charts01.flowdata)[500:], 20, 128)
        # self.showKalmann()

    def prepareExport(self):
        if len(self.file_path_dictionary) > 0:
            for key, name in self.file_path_dictionary.items():
                print(key, name)

    def printHexToConsole(self, _data: bytes, nbytes_perLine=0, tot_bytes=0):
        _counter = 0
        _line_counter = 0
        _tot_counter = 0
        for _byte in _data:
            # print(f'_byte', end='')
            # print(' 0X'.join('{:02X}'.format(_byte)), end='')
            print(f' 0X{_byte:02X}', end='')
            _counter += 1
            if nbytes_perLine != 0:
                if _counter == nbytes_perLine:
                    print(' ')
                    if _tot_counter >= tot_bytes:
                        return
                    else:
                        _tot_counter += _counter
                        _counter = 0

    def printHexToConsoleShort(self, _data: bytes, nwords_perLine=0, tot_words=0):
        _counter = 0
        _tot_counter = 0

        # Convert the byte data to 16-bit integers (Little Endian)
        num_shorts = len(_data) // 2  # Each short is 2 bytes
        shorts = struct.unpack('<' + 'H' * num_shorts, _data[:num_shorts * 2])  # Convert to 16-bit words

        for short in shorts:
            print(f' 0X{short:04X}', end='')  # Print each 16-bit word in hex format

            _counter += 1
            if nwords_perLine != 0 and _counter == nwords_perLine:
                print()  # New line
                if _tot_counter >= tot_words:
                    return
                _tot_counter += _counter
                _counter = 0
        print()  # Ensure final newline

    @Slot()
    def on_btnParse01_clicked(self):
        pass
        # self.InvokeFileModel()

    @Slot()
    def on_btnExport_clicked(self):
        self.prepareExport()
        pass
