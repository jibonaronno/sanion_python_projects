

import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class POLARBLOCK(object):
    def __init__(self):
        self.fig = Figure(figsize=(10, 2), dpi=90)
        self.plot = self.fig.add_subplot(1, 1, 1)
        self.canvases = []

    def addCanvas(self, fig, frame):
        canvas = FigureCanvasTkAgg(fig, master=frame)
        self.canvases.append(canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def updatePlot(self, new_x:[], new_y:[]):
        x = []
        if len(new_y) > 0:
            if len(new_x) != len(new_y):
                xarr = np.linspace(1, len(new_y), len(new_y))
                x = np.array(xarr)
            else:
                x = np.array(new_x)
            y = np.array(new_y)
            print(f"LEN x:{len(x)} LEN y:{len(y)}")
            #  self.plot.clear()
            self.plot.plot(x, y)
            for canvas in self.canvases:
                canvas.draw()
