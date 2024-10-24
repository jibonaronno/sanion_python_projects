#!/usr/bin/python3

'''
Drag and drop concept came from https://www.youtube.com/watch?v=s1QZIwg3x3o excelent video.
'''

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

class Sticker(QWidget):
    def __init__(self, *args, png:str):
        super().__init__(*args)
        self.pix = QPixmap()
        self.selected = False;
        self.png = png
        self.initUI()

    def initUI(self):
        self.pix.load(self.png)
        self.setGeometry(self.pix.rect())

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        qp.drawPixmap(20, 10, self.pix)
        qp.setPen(QPen(Qt.darkMagenta, 10, Qt.SolidLine))
        if self.selected:
            qp.drawRect(self.pix.rect())
        qp.end()

    def mousePressEvent(self, ev):
        if self.selected:
            self.selected = False;
        else:
            self.selected = True
        self.repaint()

    def mouseMoveEvent(self, ev:QMouseEvent):
        if self.selected:
            mimedata = QMimeData()
            drag = QDrag(self)
            drag.setMimeData(mimedata)
            dropaction = drag.exec_(Qt.MoveAction)