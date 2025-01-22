#!/usr/bin/python3
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
# from charttab import ChartTab
#
class Mimic(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)
        # self.charts = ChartTab(self)
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
        qp.drawPixmap(20, 10, QPixmap("LU_FRONT_2D.jpg"))
        qp.drawText(175, 170, "FLOW:" + str(self.meterFlow1))
        qp.drawText(175, 200, " SUM:" + str(self.meterSum1))

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
