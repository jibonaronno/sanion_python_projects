#!/usr/bin/python3
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

'''
This class will show and set parameters for a selected object. A json structure will guide
the parameter structure. Let the object name should be params. 
'''

class Paraview(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
