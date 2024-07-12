#!/usr/bin/python3
import sys
from os.path import join, dirname, abspath
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from qtpy import uic
from paint import Paint

_UI_PAINT_TABS = join(dirname(abspath(__file__)), 'charttabs.ui')

class Mimic(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.widget = uic.loadUi(_UI_PAINT_TABS, self)
        self.setAcceptDrops(True)
        self.paint = Paint()
        self.initUI()


    def initUI(self):
        # self.setGeometry(0, 0, 1500, 1200)
        self.vlay01.addWidget(self.paint)
