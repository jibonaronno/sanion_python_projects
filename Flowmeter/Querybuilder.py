
'''
Custom Date Selector.
Code came from following page.
'''
#https://stackoverflow.com/questions/21674060/qt-pyqt4-pyside-qdateedit-calendar-popup-falls-off-screen

'''
#Following code block is a solution from a question by the coder. It was about positioning the 
#calender block.
class Widget(QWidget):
    def __init__(self):
        super().__init__()
        self.setupUi()

    def setupUi(self):
        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.dateEdit = QDateEdit()
        self.dateEdit.setCalendarPopup(True)
        self.dateEdit.calendarWidget().installEventFilter(self)
        layout.addWidget(self.dateEdit, 0, 0)
        layout.addItem(QSpacerItem(20, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))
        self.setLayout(layout)

    def eventFilter(self, obj, event):
        if obj == self.dateEdit.calendarWidget() and event.type() == QEvent.Show:
            pos = self.dateEdit.mapToGlobal(self.dateEdit.geometry().bottomRight())
            width = self.dateEdit.calendarWidget().window().width()
            self.dateEdit.calendarWidget().window().move(pos.x() - width, pos.y())
        return False
'''

from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
import datetime
from os import path
import decimal

headers = [('BA Number', int), ('Applicant Name', str), ('Site Address', str), ('Application Type', str), ('Applicant Address', str), ('Applicant Tel.', str), ('Applicant Fax', str), ('Applicant Email', str), ('Applicant Status', str), ('Agent Name', str), ('Agent Address', str), ('Agent Tel.', str), ('Agent Fax', str), ('Agent Email', str), ('Property No.', str), ('Proposed Works', str), ('Landuse', str), ('Application Date', datetime.datetime), ('Application Received', datetime.datetime), ('Estimated Costs', float), ('EIA', str), ('Inspectors Initials', str), ('Landlord Name', str), ('Landlord Address', str), ('Landlord Tel.', str), ('Landlord Fax', str), ('Landlord Email', str), ('Decision', str), ('Plan Permit No.', str), ('Plan Permit Date', datetime.datetime), ('Build Notice No.', str), ('Build Notice Date', datetime.datetime), ('Build Notice Comments', str), ('Plan Permit Expiry Date', datetime.datetime), ('Build Notice Expiry Date', datetime.datetime), ('Fee Building', float), ('Fee Planning', float), ('Fee Inspection', float), ('Works Completed', datetime.datetime), ('Works Started', datetime.datetime), ('Appeal', str), ('Notes', str), ('Completion Certificate', str), ('Fitness Certificate Issued', str), ('DPC Decision', str), ('DPC Date', datetime.datetime), ('DPC Approved by', str), ('GOG', str)]

numbercomparisonoperators = [('=', 'Equal to'), ('>', 'Greater than'), ('<', 'Less than'), ('>=', 'Greater than or equal to'), ('<=', 'Less than or equal to')]
datecomparisonoperators = [('=', 'On'), ('>', 'More recent than'), ('<', 'Earlier than'), ('>=', 'More recent than or on'), ('<=', 'Earlier than or on')]
stringcomparisonoperators = [('=', 'Equal to'), ('in', 'Contains'), ('any', 'Match any sequence of character in')]
logicaloperators = [('and', 'Match both criteria'), ('or', 'Match either or both criteria'), ('not', 'Don\'t match the following criteria')]

iconpath = ''


class QueryBuilderWidget(QWidget):
    """
    Display a widget to enable filtering multiple columns in a QTreeWidget
    """
    def __init__(self, parent=None):
        super(QueryBuilderWidget, self).__init__()
        self.parent = parent
        self.setMinimumSize(300, 100)

        self.centralLayout = QGridLayout()
        self.centralLayout.setContentsMargins(0, 0, 0, 0)
        self.centralLayout.setSpacing(2)
        self.centralLayout.setColumnStretch(1, 1)
        self.centralLayout.setRowStretch(3, 1)
        self.setLayout(self.centralLayout)

        AddFieldLabel = QLabel()
        AddFieldLabel.setText('Add field:')
        self.centralLayout.addWidget(AddFieldLabel, 0, 0)

        self.FieldSelectorCombo = FieldSelectorCombo(self)
        self.centralLayout.addWidget(self.FieldSelectorCombo, 0, 1)
        self.FieldSelectorCombo.activated.connect(self.enable_add_item)

        self.QueryTreeWidget = QueryTreeWidget()
        self.centralLayout.addWidget(self.QueryTreeWidget, 1, 0, 3, 2)
        self.QueryTreeWidget.itemSelectionChanged.connect(self.enable_buttons)

        self.AddItemButton = AddItemButton(self)
        self.AddItemButton.setEnabled(False)
        self.centralLayout.addWidget(self.AddItemButton, 0, 2)

        self.RemoveItemButton = RemoveItemButton(self.QueryTreeWidget)
        self.RemoveItemButton.setEnabled(False)
        self.centralLayout.addWidget(self.RemoveItemButton, 1, 2)

        self.AndButton = AndButton(self)
        self.AndButton.setEnabled(False)
        self.centralLayout.addWidget(self.AndButton, 2, 2)

    def enable_buttons(self):
        """
        Enable certain UI buttons as required when items are selected in a
        QTreeWidget
        """
        if self.QueryTreeWidget.currentItem():
            self.RemoveItemButton.setEnabled(True)
            self.AndButton.setEnabled(True)
        else:
            self.RemoveItemButton.setEnabled(False)
            self.AndButton.setEnabled(False)

    def enable_add_item(self, index):
        """
        Enable certain UI buttons as required when items are selected in a
        QComboBox
        """
        if index != -1:
            self.AddItemButton.setEnabled(True)
        else:
            self.AddItemButton.setEnabled(False)


class RemoveItemButton(QPushButton):
    """docstring for RemoveItemButton"""
    def __init__(self, parent=None):
        super(RemoveItemButton, self).__init__()
        self.parent = parent

        self.setMaximumSize(24, 24)
        minusicon = QtGui.QIcon()
        minusicon.addPixmap(QtGui.QPixmap(path.join(iconpath, 'minus.png')),
                            QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setIconSize(QtCore.QSize(16, 16))
        self.setIcon(minusicon)
        self.setToolTip('Remove selected item')

        self.clicked.connect(self.remove_selected_item)

    def remove_selected_item(self):
        currentitem = self.parent.currentItem()
        currentindex = self.parent.indexOfTopLevelItem(currentitem)
        self.parent.takeTopLevelItem(currentindex)


class AndButton(QPushButton):
    """docstring for AndButton"""
    def __init__(self, parent=None):
        super(AndButton, self).__init__()
        self.parent = parent

        self.setMaximumSize(24, 24)
        self.setText('and')
        self.setToolTip('Match both criteria')

        self.clicked.connect(self.add_and_item)

    def add_and_item(self):
        treewidget = self.parent.QueryTreeWidget
        treewidget.add_logic_item('and')


class AddItemButton(QPushButton):
    """docstring for AddItemButton"""
    def __init__(self, parent=None):
        super(AddItemButton, self).__init__()
        self.parent = parent

        self.setMaximumSize(24, 24)
        addicon = QtGui.QIcon()
        addicon.addPixmap(QtGui.QPixmap(path.join(iconpath, 'plus.png')),
                          QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setIconSize(QtCore.QSize(16, 16))
        self.setIcon(addicon)
        self.setToolTip('Add selected item')

        self.clicked.connect(self.add_selected_item)

    def add_selected_item(self):
        FieldCombo = self.parent.FieldSelectorCombo
        index = FieldCombo.currentIndex()
        fieldname = FieldCombo.currentText()
        #fieldtype = FieldCombo.itemData(index, QtCore.Qt.UserRole).toPyObject()
        #fieldtype = type(FieldCombo.itemData(index, QtCore.Qt.UserRole))
        fieldtype = QtCore.QVariant(FieldCombo.itemData(index, QtCore.Qt.UserRole))
        if fieldname:
            self.parent.QueryTreeWidget.addItem(fieldname, fieldtype)


class FieldSelectorCombo(QComboBox):
    """docstring for FieldSelectorCombo"""
    def __init__(self, parent=None):
        super(FieldSelectorCombo, self).__init__()
        self.parent = parent

        count = 0
        for field, fieldtype in headers:
            self.addItem(field)
            self.setItemData(count, fieldtype, QtCore.Qt.UserRole)
            count += 1
        self.setCurrentIndex(-1)

    def keyPressEvent(self, event):
        """ """
        if event.key() == QtCore.Qt.Key_Return:
            self.parent.AddItemButton.add_selected_item()
        return QtGui.QComboBox.keyPressEvent(self, event)

    def wheelEvent(self, event):
        event.ignore()


class OperatorComboBox(QComboBox):
    """docstring for OperatorComboBox"""
    def __init__(self, parent=None, operatorhint=int):
        super(OperatorComboBox, self).__init__()
        self.parent = parent

        if operatorhint in (int, float):
            operatorlist = numbercomparisonoperators
        elif operatorhint == str:
            operatorlist = stringcomparisonoperators
        elif operatorhint == datetime.datetime:
            operatorlist = datecomparisonoperators

        count = 0
        for comparisonoperator, tooltip in operatorlist:
            self.addItem(comparisonoperator)
            self.setItemData(count, comparisonoperator)
            self.setItemData(count, tooltip, QtCore.Qt.ToolTipRole)
            count += 1

    def wheelEvent(self, event):
        event.ignore()


class OperatorLineEdit(QLineEdit):
    """docstring for OperatorLineEdit"""
    def __init__(self, parent=None):
        super(OperatorLineEdit, self).__init__()
        self.parent = parent


class DateWidget(QDateEdit):
    """docstring for DateWidget"""
    def __init__(self, parent=None):
        super(DateWidget, self).__init__()
        self.parent = parent

        self.setDate(QtCore.QDate.currentDate())
        self.setCalendarPopup(True)
        self.setDisplayFormat('dd/MM/yyyy')
        self.cal = self.calendarWidget()
        self.cal.setFirstDayOfWeek(QtCore.Qt.Monday)
        self.cal.setHorizontalHeaderFormat(QCalendarWidget.SingleLetterDayNames)
        self.cal.setVerticalHeaderFormat(QCalendarWidget.NoVerticalHeader)
        self.cal.setGridVisible(True)


class QueryTreeWidget(QTreeWidget):
    """docstring for QueryTreeWidget"""
    def __init__(self, parent=None):
        super(QueryTreeWidget, self).__init__()
        self.parent = parent
        self.setAlternatingRowColors(True)
        headerlabels = ('Field', 'Operator', 'Value')
        self.setHeaderLabels(headerlabels)
        self.setColumnCount(len(headerlabels))
        ####self.header().setResizeMode(QHeaderView.ResizeToContents)
        ####self.header().setMovable(False)
        self.setSelectionMode(QAbstractItemView.ContiguousSelection)

    def addItem(self, itemtext, itemtype):
        treewidgetitem = QTreeWidgetItem()
        treewidgetitem.setText(0, itemtext)

        OperatorSelectorBox = OperatorComboBox(operatorhint=itemtype)

        if itemtype in (str, int, float):
            valueWidget = OperatorLineEdit(self)
        elif itemtype == datetime.datetime:
            valueWidget = DateWidget(self)

        self.insertTopLevelItem(self.insert_index(), treewidgetitem)
        self.setItemWidget(treewidgetitem, 1, OperatorSelectorBox)
        self.setItemWidget(treewidgetitem, 2, valueWidget)
        self.setCurrentItem(treewidgetitem)

    def add_logic_item(self, logic):
        treewidgetitem = QTreeWidgetItem()
        treewidgetitem.setText(0, 'and')
        treewidgetitem.setFont(0, QtGui.QFont('', -1, QtGui.QFont.Bold))
        treewidgetitem.setData(0, QtCore.Qt.UserRole, 'and')
        self.insertTopLevelItem(self.insert_index(), treewidgetitem)

    def insert_index(self):
        if self.currentItem():
            currentindex = self.indexOfTopLevelItem(self.currentItem()) + 1
        else:
            currentindex = 0
        return currentindex


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    w = QueryBuilderWidget(app)
    w.show()
    app.exec_()
