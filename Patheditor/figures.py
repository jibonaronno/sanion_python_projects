#  Because we are using TkInter gui for viewing plots so conventional document resources
#  are not applicable sometimes. Our helper object is FigureCanvasTkAgg .
#
import pprint

import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class FIGS(object):
    def __init__(self, parent):
        self.dash = parent
        self.fig = Figure(figsize=(10, 2), dpi=90)
        self.axis = self.fig.add_subplot(1,1,1)
        self.canvases = []

    def addCanvas(self, fig, frame):
        canvas = FigureCanvasTkAgg(fig, master=frame)
        self.canvases.append(canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def updatePlot(self, new_x:[], new_y:[], style_string:str):
        x = []
        if len(new_y) > 0:
            if len(new_x) != len(new_y):
                xarr = np.linspace(1, len(new_y), len(new_y))
                x = np.array(xarr)
            else:
                x = np.array(new_x)
            y = np.array(new_y)
            print(f"LEN x:{len(x)} LEN y:{len(y)}")
            #  self.axis.clear()
            self.axis.plot(x, y, style_string)
            for canvas in self.canvases:
                canvas.draw()

    def addSampleCanvas(self, frame):
        new_x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        new_y = [5, 3, 1, 2, 4, 2, 6, 10, 1, 11, 11, 10, 9, 8]
        self.axis.clear()
        self.axis.plot(new_x, new_y)
        self.addCanvas(self.fig, frame)
        #  pprint.pprint(self.axis.get_paths())