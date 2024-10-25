import sys
from PyQt5.QtWidgets import QApplication, QTreeView, QFileDialog, QVBoxLayout, QPushButton, QWidget
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QDir
from PyQt5.QtWidgets import QFileSystemModel
from qtpy import uic
from os.path import join, dirname, abspath
from qtpy.QtCore import Slot

_UI_PAINT_TABS = join(dirname(abspath(__file__)), 'comparison_chart.ui')

class CompareChartWidget(QWidget):
    def __init__(self, folder_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.widget = uic.loadUi(_UI_PAINT_TABS, self)
        self.model = QFileSystemModel()
        # self.InvokeFileModel(QDir.rootPath())
        #self.InvokeFileModel(QDir.currentPath())
        self.InvokeFileModel(folder_path)
        # self.InvokeFileModel("wavegenA_PD_FILTER_GRAPH")
        self.treeView.setModel(self.model)


    def InvokeFileModel(self, folder_path):
        #self.model.setRootPath("C:\\Users\\jibon\\PycharmProjects\\")
        self.treeView.setRootIndex(self.model.index("C:\\Users\\jibon\\PycharmProjects\\"))
        # self.treeView.setModel(self.model)
        self.treeView.setColumnWidth(0, 250)
        print(folder_path)

    @Slot()
    def on_btnParse01_clicked(self):
        pass
        # self.InvokeFileModel()
