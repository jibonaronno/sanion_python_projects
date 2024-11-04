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
        for byt in bytesArray:
            self.charts01.Append(int(byt))

    def on_tree_item_click(self, index: QModelIndex):
        # Retrieve the item that was clicked
        item = self.model.itemFromIndex(index)
        print(f"Clicked on: {item.text()} : {self.file_path_dictionary[item.text()]}")
        self.binfil.printFilContentSize(self.file_path_dictionary[item.text()])
        self.injectDataStreamToGraph()


    @Slot()
    def on_btnParse01_clicked(self):
        pass
        # self.InvokeFileModel()
