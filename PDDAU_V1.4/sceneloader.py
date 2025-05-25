from paint import ControlPoint
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
class SceneLoader(QObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.controlpoints = []
        self.addControlPoint(10, 10)
        self.addControlPoint(30, 30)
        self.addControlPoint(50, 50)

    def addControlPoint(self, _x, _y):
        coord = QPoint(_x, _y)
        cpoint = ControlPoint()
        cpoint.coord = coord
        self.controlpoints.append(cpoint)
