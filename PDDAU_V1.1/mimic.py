#!/usr/bin/python3
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import json
# from charttab import ChartTab
#

class RectFromJson(QObject):
    def __init__(self, qwidget:QWidget):
        super().__init__()
        self.widget = qwidget
        self.rectangles = []

    def loadShapes(self, json_file):
        try:
            with open(json_file, 'r') as f:
                self.rectangles = json.load(f)
        except Exception as e:
            print("Error Loading JSON File")

    def paintEvent(self, event):
        try:
            painter = QPainter(self.widget)
            for rect in self.rectangles:
                x = rect.get("x", 0)
                y = rect.get("y", 0)
                width = rect.get("width", 50)
                height = rect.get("height", 50)
                color = rect.get("color", "#FFA500")

                pen = QPen(QColor(color))
                pen.setWidth(12)
                painter.setPen(pen)
                painter.drawRect(x, y, width, height)

            pass
        except Exception as e:
            print("RectFromJson class paintEvent() Error")
            print(str(e))

class Mimic(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)
        # self.charts = ChartTab(self)
        self.rects_json = RectFromJson(self)
        self.rects_json.loadShapes("rects.json")
        self.lwidth = 0
        self.lheight = 0
        self.img_loaded = False
        self.initUI()
        self.show()

    def AppendFlow1(self, data=0):
        pass
        #self.charts.Append(data)

    def dragEnterEvent(self, ev:QDragLeaveEvent):
        ev.accept()

    def dragMoveEvent(self, ev:QDragMoveEvent):
        qp = QPainter()
        pass

    def dropEvent(self, ev:QDropEvent):
        pos = ev.pos()
        ##self.st.move(pos)
        ev.accept()
        print("POS : " + str(pos.x()) + " " + str(pos.y()))
        ##self.st.selected = False
        ##self.st.repaint()


    def initUI(self):
        self.text = "hello world"
        self.setGeometry(0, 0, 1500, 1200)
        # self.charts.setGeometry(10, 400, 1400, 500)
        # self.charts.show()


        self.setWindowTitle('Draw Demo')
        self.meterFlow1 = "000.00"
        self.meterFlow2 = "000.00"
        self.meterFlow3 = "000.00"
        self.meterSum1 = "000.00"
        self.meterSum2 = "000.00"
        self.meterSum3 = "000.00"

        '''
        Code below displays a Sticker object on mimic window.
        '''
        #self.st.show()

    def paintEvent(self, event):

        qp = QPainter()
        #qp.setPen(QPen(Qt.black, 6, Qt.SolidLine))
        #font = qp.font()
        #font.setPixelSize(48)
        #qp.setFont(font)
        #qp.setPen(QColor(Qt.red))

        #font_db = QFontDatabase()
        #font_id = font_db.addApplicationFont("Seven Segment.ttf")

        qp.begin(self)

        #qp.setFont(QFont('Courier New', 20))

        qp.setPen(QColor(Qt.white))
        #font = qp.font()
        font = QFont('Seven Segment', 18)
        #Sfont.setPixelSize(48)
        qp.setFont(font)
        #qp.drawRect(10, 150, 150, 100)
        #qp.setPen(QColor(Qt.yellow))
        #qp.drawEllipse(100, 50, 100, 50)
        pxmp = QPixmap("LU_FRONT_2D.jpg")
        qp.drawPixmap(20, 10, pxmp)
        qp.drawText(175, 170, "FLOW:" + str(self.meterFlow1))
        qp.drawText(175, 200, " SUM:" + str(self.meterSum1))

        if not self.img_loaded:
            print(f'Width = {pxmp.width()} Height = {pxmp.height()}')
            self.img_loaded = True

        self.rects_json.paintEvent(event)

        # qp.drawPixmap(480, 10, QPixmap("meter.jpg"))
        # qp.drawText(635, 170, "FLOW:" + str(self.meterFlow2))
        # qp.drawText(635, 200, " SUM:" + str(self.meterSum2))
        #
        # qp.drawPixmap(940, 10, QPixmap("meter.jpg"))
        # qp.drawText(1095, 170, "FLOW:" + str(self.meterFlow3))
        # qp.drawText(1095, 200, " SUM:" + str(self.meterSum3))
        #htmlDoc1.drawContents(qp, rect2)

        #qp.drawText(390, 152, "FLOW:" + self.meterFlow)
        #qp.fillRect(20, 175, 130, 70, QBrush(Qt.SolidPattern))
        qp.end()
