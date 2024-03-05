import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from tkinter import ttk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


class FIGS(object):
    def __init__(self):
        self.fig = Figure(figsize=(10, 2), dpi=50)
        self.plot = self.fig.add_subplot(1,1,1)
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
                x = new_x
            self.plot.clear()
            self.plot(x, new_y)
            for canvas in self.canvases:
                canvas.draw()

    def addSampleCanvas(self, frame):
        new_x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        new_y = [5, 3, 1, 2, 4, 2, 6, 10, 1, 11]
        self.plot.clear()
        self.plot.plot(new_x, new_y)
        self.addCanvas(self.fig, frame)