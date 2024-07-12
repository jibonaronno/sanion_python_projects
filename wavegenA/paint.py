#!/usr/bin/python3
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import json

class ControlPoint(QObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.coord = QPoint(10, 10)

class Paint(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.setAcceptDrops(True)
        self.initUI()
        self.controlPoints = []
        self.addControlPoint(10, 10)
        self.addControlPoint(30, 30)
        self.addControlPoint(50, 50)
        self.repaint()


    def initUI(self):
        self.setGeometry(0, 0, 1500, 1200)

    def paintEvent(self, event):
        qpainter = QPainter(self)
        qpainter.setPen(QPen(Qt.green, 8, Qt.SolidLine))
        # Qt.GlobalColor.green
        qpainter.setBrush(QBrush(Qt.red, Qt.SolidPattern))
        for cpoint in self.controlPoints:
            qpainter.drawEllipse(cpoint.coord.x(), cpoint.coord.y(), 5, 5)

    def addControlPoint(self, _x, _y):
        coord = QPoint(_x, _y)
        cpoint = ControlPoint()
        cpoint.coord = coord
        self.controlPoints.append(cpoint)