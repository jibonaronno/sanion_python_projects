'''
Reminder : mimic.py contains multiple tabs and the base UI created from QTDesigner app 'charttab.ui'.
In Flowmeter app mimic.py itself handles paint event for a tab. Here we are trying to use QGraphicsScene
class for drawing ops.
'''

import sys
from os.path import join, dirname, abspath
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from qtpy import uic
from paint import Paint
from graphicsscene import GraphicsScene
_UI_CHART_TABS = join(dirname(abspath(__file__)), 'charttabs.ui')

class Mimic(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.widget = uic.loadUi(_UI_CHART_TABS, self)
        self.setAcceptDrops(True)
        self.scene = GraphicsScene()
        self.scene.addText("Hello, world!")
        self.scene.addEllipse(10, 10, 10, 10)
        self.paint = Paint()
        self.initUI()


    def initUI(self):
        # self.setGeometry(0, 0, 1500, 1200)
        self.gfxvu.setScene(self.scene)
        self.vlay01.addWidget(self.paint)
        for itm in self.scene.items():
            itm.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
